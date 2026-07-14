#!/usr/bin/env python3
"""gen_qwopus_ffn.py — 8-tile NPU FFN matvec (Phase 3a, Re-stream-h + BF16 MMUL).

M=1 MXFP4 matvec, 8-tile N-split.  ONE uint8 stream per tile.  Packet =
[32 B header(row_base,kc) | h_chunk(2 KB BF16) | M_n*G*17 weight bytes] = 10784 B
(< 16 KB => 51 GB/s DMA side of the on-tile-array / packet-buffer wall).  h
streams as BF16 (the model's h IS bf16) and is re-streamed every packet (never
held on-tile — a 20 KB held array is 1470× slow).  Accumulation is into the held
S2MM output `o` (N_TILE fp32 = 8.7 KB, < 16 KB => fast RMW, host pre-zeros it);
each packet does o[row_base+r] = (kc==0 ? partial : o[row_base+r] + partial).

The C kernel is K-agnostic (processes G=32 groups/packet); the host controls the
packet count (n_chunks × k_chunks) and per-packet row_base.  This build targets
the gate/up shape N=17408, K=5120 (N_TILE=2176, k_chunks=5, n_chunks=136,
NPKT=680/tile).  The same kernel serves down [5120,17408] with different host
packing (k_chunks=17, n_chunks=40) — recompile only if N_TILE changes (held
output size).

Emits TWO xclbins sharing the geometry so the probe measures COMPUTE-vs-DMA:
  fst_qwopus_ffn.xclbin      — ffn_matvec_restream      (full dequant + MMUL)
  fst_qwopus_ffn_noop.xclbin — ffn_matvec_noop_stream   (same DMA, no compute)
If the matvec is much slower than the no-op, dequant compute dominates and the
nibble extract must be vectorised before wiring; if they match, the kernel is
DMA-bound at 51 GB/s (the win).  Phase 3a swaps the scalar fp32 MAC for native
aie::mmul<4,8,8> bf16×bf16->fp32-acc (reuse fst_fused_dequant_gemm_16x64x64);
target ~2 ms/matrix (the DMA floor).
"""
import os, sys, shutil
from pathlib import Path

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
IRON_PATH = str(PROJ_ROOT.parent / "Source" / "IRON-devel")
if IRON_PATH not in sys.path:
    sys.path.insert(0, IRON_PATH)
os.environ.setdefault("PATH", os.environ["PATH"] + ":" + os.path.expanduser("~/.local/bin"))
os.environ.setdefault("PEANO_INSTALL_DIR",
                       os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

import numpy as np
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

# ── Packet geometry (must match fst_qwopus_ffn_kernel.cc) ────────────────────
M_n       = 16
G         = 32
HDR       = 32
MH = 4
H_BYTES   = MH * G * 32 * 2        # 8192 (4 h-vectors)
BLK_BYTES = 17
W_BYTES   = M_n * G * BLK_BYTES   # 8704
PKT_BYTES = HDR + H_BYTES + W_BYTES   # 16928 (4 h's + W)

# ── Shape: gate/up [17408, 5120] ─────────────────────────────────────────────
HV_K    = 5120
GROUPS  = HV_K // 32              # 160
KCHUNKS = GROUPS // G             # 5 k_chunks per n_chunk
N       = 17408
NT      = 8
N_TILE  = N // NT                 # 2176
NCHUNKS = N_TILE // M_n           # 136 n_chunks per tile
NPKT    = NCHUNKS * KCHUNKS      # 680 packets per tile
# BD repeat <= 255: 680 = 8 × 85
NPKT_O  = 8
NPKT_I  = NPKT // NPKT_O          # 85

# Env-parameterized so arbitrary kernel variants (const/dq=1.0 isolation,
# no-transpose experiment, future M=4) can be built without touching the
# defaults.  FST_FFN_SRC overrides the source file; FST_FFN_FN / _XCLBIN /
# _OBJ override the names.  Defaults reproduce the original build exactly.
SRC = os.environ.get("FST_FFN_SRC", str(PROJ_ROOT / "fst_qwopus_ffn_kernel.cc"))
INC = [aie_config.cxx_header_path()]

PKT_T = np.ndarray[(PKT_BYTES,), np.dtype[np.uint8]]   # [hdr | h_chunk | weights]
OUT_T = np.ndarray[(N_TILE,),   np.dtype[np.float32]]  # held, drained once


def _build(fn_name, xclbin_name, obj_name):
    @iron.jit
    def ffn_op(inp: In, out: Out):
        kM = ExternalFunction(fn_name, source_file=SRC,
            arg_types=[PKT_T, OUT_T], include_dirs=INC, object_file_name=obj_name)

        fins = [ObjectFifo(PKT_T, name=f"in{t}", depth=1) for t in range(NT)]
        fos  = [ObjectFifo(OUT_T, name=f"o{t}",  depth=1) for t in range(NT)]

        def core(fin, fo, kM):
            o = fo.acquire(1)                       # held output, drained once
            for _p in range_(NPKT):
                p = fin.acquire(1)
                kM(p, o)
                fin.release(1)
            fo.release(1)

        ws = [Worker(core, [fins[t].cons(), fos[t].prod(), kM]) for t in range(NT)]

        rt = Runtime()
        with rt.sequence(np.ndarray[(NT * NPKT * PKT_BYTES,), np.dtype[np.uint8]],
                         np.ndarray[(NT * N_TILE,),        np.dtype[np.float32]]) as (inp, out):
            rt.start(*ws)
            tg = rt.task_group()
            for t in range(NT):
                rt.fill(fins[t].prod(), inp,
                        tap=TensorAccessPattern(tensor_dims=(1, NT * NPKT * PKT_BYTES),
                                                offset=t * NPKT * PKT_BYTES,
                                                sizes=[NPKT_O, NPKT_I, PKT_BYTES],
                                                strides=[NPKT_I * PKT_BYTES, PKT_BYTES, 1]),
                        task_group=tg)
                rt.drain(fos[t].cons(), out,
                         tap=TensorAccessPattern(tensor_dims=(NT * N_TILE,), offset=t * N_TILE,
                                                 sizes=[1, N_TILE], strides=[0, 1]),
                         task_group=tg, wait=(t == NT - 1))
            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()

    xclbin, insts = ffn_op.compile()
    # Always write into kernels/ (PROJ_ROOT), regardless of CWD, so xclbins/insts
    # land in their canonical home (never the repo root — repo-cleanliness rule).
    out_xclbin = str(PROJ_ROOT / f"{xclbin_name}.xclbin")
    out_insts  = str(PROJ_ROOT / f"{xclbin_name}_insts.bin")
    shutil.copy(xclbin, out_xclbin)
    shutil.copy(insts, out_insts)
    print(f"{xclbin_name}: {out_xclbin} ({os.path.getsize(out_xclbin)}B)  "
          f"pkt={PKT_BYTES}B  {NPKT} pkts/tile  {NT} tiles  N_TILE={N_TILE} k_chunks={KCHUNKS}")


if __name__ == "__main__":
    # Build each variant in a SEPARATE process: @iron.jit caches the resolved
    # program in-process and fn_name is not part of the cache key, so building
    # both in one process yields two identical xclbins (both = the first build).
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    # Env-override path: build a single arbitrary variant (one process, one xclbin).
    if os.environ.get("FST_FFN_FN"):
        _build(os.environ["FST_FFN_FN"],
               os.environ.get("FST_FFN_XCLBIN", "fst_qwopus_ffn"),
               os.environ.get("FST_FFN_OBJ", "fst_qwopus_ffn.o"))
        print(f"OK FFN variant — fn={os.environ['FST_FFN_FN']} src={SRC} "
              f"pkt={PKT_BYTES}B<16KB N_TILE={N_TILE} k_chunks={KCHUNKS}")
    elif which in ("both", "full"):
        _build("ffn_matvec_restream", "fst_qwopus_ffn",      "fst_qwopus_ffn.o")
    if which in ("both", "noop") and not os.environ.get("FST_FFN_FN"):
        _build("ffn_matvec_noop_stream", "fst_qwopus_ffn_noop", "fst_qwopus_ffn_noop.o")
    print(f"OK re-stream-h FFN [{which}] — pkt={PKT_BYTES}B<16KB, h_chunk={H_BYTES}B, "
          f"M_n={M_n}, G={G}, NPKT={NPKT}/tile, N_TILE={N_TILE} (gate/up [17408,5120])")