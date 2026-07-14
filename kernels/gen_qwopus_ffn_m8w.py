#!/usr/bin/env python3
"""gen_qwopus_ffn_m8.py — M=8 SPECULATIVE-DECODING FFN, 32-tile N-split.

M=8 (8 h-vectors), NT=16 compute tiles (the NPU2 has 32 compute tiles but only 8
shims, each with 2 MM2S + 2 S2MM DMA channels => 16-tile ceiling for a 1-fifo-in
+ 1-fifo-out-per-tile design; NT=32 does NOT place — "no ShimNOCTile has
sufficient DMA capacity").  N_TILE=1088 -> M=8 held output = 8x1088x4 = 34.8 KB
(fits a 64 KB tile; M=8 on NT=8 was 69 KB = the wall).  G_pkt=16 (not 32) so
the packet = 32 + 8x(16x32x2) + 16x16x17 = 12576 B < 16 KB (stays on the fast
DMA side of the packet-buffer wall; G_pkt=16 is the only G dividing BOTH up
GROUPS=160 and down GROUPS=544 that keeps M=8 < 16 KB).

Architecture: 2x-dequant, 1-live MMUL (n4v structure), 8 h-vectors in the 8
A-rows of <8,8,4>.  NOT dequant-once/2-live — the n4_live2 probe proved
dequant-once is correct but 7 ms SLOWER (2-live overhead > saved dequant).
The win comes from NT=16 (2x parallelism over the 8-tile M=1 n4v 85 ms), not
dequant amortization.  Projected M=8 ~ 85/2 ~ 42.5 ms full for 8 tokens.

Targets gate/up [17408,5120]: N_TILE=1088, KCHUNKS=10 (160/16), NCHUNKS=68
(1088/16), NPKT=680/tile.  Down [5120,17408] is served by the SAME xclbin with
host packing N_TILE_USED=320 (down's per-tile @ NT=16); the compiled 1088-wide
held output wastes the upper 768/tile (probe-only, fine).
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

# ── Packet geometry (must match fst_qwopus_ffn_m8w_kernel.cc) ──────────────────
M_n       = 32
MH        = 8
G         = 16          # G_pkt = 16 (keeps M=8 packet < 16 KB; divides 160 & 544)
HDR       = 32
H_BYTES   = MH * G * 32 * 2        # 8192 (8 bf16 h-vectors)
BLK_BYTES = 17
W_BYTES   = M_n * G * BLK_BYTES   # 8704
PKT_BYTES = HDR + H_BYTES + W_BYTES   # 16928 (M_n=32 wide; M=4-proven DMA-safe)

# ── Shape: gate/up [17408, 5120] ─────────────────────────────────────────────
HV_K    = 5120
GROUPS  = HV_K // 32              # 160
KCHUNKS = GROUPS // G             # 10 k_chunks per n_chunk
N       = 17408
NT      = 16
N_TILE  = N // NT                 # 1088
NCHUNKS = N_TILE // M_n           # 34 n_chunks per tile (M_n=32)
NPKT    = NCHUNKS * KCHUNKS      # 340 packets per tile
# BD repeat <= 255: 340 = 10 x 34
NPKT_O  = 10
NPKT_I  = NPKT // NPKT_O          # 34

SRC = os.environ.get("FST_FFN_SRC", str(PROJ_ROOT / "fst_qwopus_ffn_m8w_kernel.cc"))
INC = [aie_config.cxx_header_path()]

PKT_T = np.ndarray[(PKT_BYTES,),       np.dtype[np.uint8]]   # [hdr | 8 h's | weights]
OUT_T = np.ndarray[(MH * N_TILE,),     np.dtype[np.float32]]  # 8x held output, drained once


def _build(fn_name, xclbin_name, obj_name):
    @iron.jit
    def ffn_op(inp: In, out: Out):
        kM = ExternalFunction(fn_name, source_file=SRC,
            arg_types=[PKT_T, OUT_T], include_dirs=INC, object_file_name=obj_name)

        # depth=1: the wide 16928 B packet x depth=2 + 34.8 KB held output = 68.7
        # KB > 64 KB tile (overflow).  depth=1 fits (51.7 KB).  depth=2 only bought
        # ~4 ms on the M_n=16 variant anyway (the floor is per-packet dispatch,
        # not pipelining) — m8w's win comes from halving the packet count.
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
                         np.ndarray[(NT * MH * N_TILE,),    np.dtype[np.float32]]) as (inp, out):
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
                         tap=TensorAccessPattern(tensor_dims=(NT * MH * N_TILE,), offset=t * MH * N_TILE,
                                                 sizes=[1, MH * N_TILE], strides=[0, 1]),
                         task_group=tg, wait=(t == NT - 1))
            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()

    xclbin, insts = ffn_op.compile()
    out_xclbin = str(PROJ_ROOT / f"{xclbin_name}.xclbin")
    out_insts  = str(PROJ_ROOT / f"{xclbin_name}_insts.bin")
    shutil.copy(xclbin, out_xclbin)
    shutil.copy(insts, out_insts)
    print(f"{xclbin_name}: {out_xclbin} ({os.path.getsize(out_xclbin)}B)  "
          f"pkt={PKT_BYTES}B  {NPKT} pkts/tile  {NT} tiles  N_TILE={N_TILE} k_chunks={KCHUNKS}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if os.environ.get("FST_FFN_FN"):
        _build(os.environ["FST_FFN_FN"],
               os.environ.get("FST_FFN_XCLBIN", "fst_qwopus_ffn_m8w"),
               os.environ.get("FST_FFN_OBJ", "fst_qwopus_ffn_m8w.o"))
        print(f"OK M=8 variant — fn={os.environ['FST_FFN_FN']} src={SRC} "
              f"pkt={PKT_BYTES}B<16KB N_TILE={N_TILE} NT={NT} k_chunks={KCHUNKS}")
    elif which in ("both", "full"):
        _build("ffn_matvec_restream", "fst_qwopus_ffn_m8w",      "fst_qwopus_ffn_m8w.o")
    if which in ("both", "noop") and not os.environ.get("FST_FFN_FN"):
        _build("ffn_matvec_noop_stream", "fst_qwopus_ffn_m8w_noop", "fst_qwopus_ffn_m8w_noop.o")
    print(f"OK M=8 FFN [{which}] — pkt={PKT_BYTES}B<16KB, 8 h's, G_pkt={G}, "
          f"M_n={M_n}, NPKT={NPKT}/tile, NT={NT} tiles, N_TILE={N_TILE} (gate/up [17408,5120])")