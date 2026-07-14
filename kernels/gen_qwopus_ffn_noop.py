#!/usr/bin/env python3
"""gen_qwopus_ffn_probe.py — No-op diagnostic (no h-array access) — isolates weight-DMA
FFN lever (Qwopus3.6 dense SwiGLU FFN, M=1).

Measures the make-or-break unknown: NPU sustained weight-read bandwidth with
large BDs + a 20 KB read-only held input vector h.  One tile streams weight packets with NO h-array access (isolates weight-DMA from the 20KB held-h cost)
gate-projection matvec (out[n] = sum_k W[n,k]·h[k], W MXFP4 17-byte blocks
padded to 18) over N_TILE rows of the gate weight.  The host probe
(tools/qwopus_ffn_probe.cpp) packs synthetic MXFP4 + h, dispatches, times the
steady-state, and compares to the host fp32 matvec reference.

Layout:
  f_h   : MM2S, HV_K=5120 fp32, depth=1, HELD across all weight packets (20 KB
          read-only on-tile — tests the on-tile-array risk class).
  f_w   : MM2S, RPB × 2880 B (= RPB×2880/4 float), depth=2, streamed.  Typed
          float (reinterpret bytes) so it DMAs — a uint8 fifo alongside a float
          fifo does NOT DMA on this IRON build (fst_fused_dequant_gemm contract).
  f_out : S2MM, N_TILE fp32, depth=1, HELD, written incrementally, drained ONCE
          at the end (one S2MM BD -> isolates MM2S weight bandwidth).

Emits fst_qwopus_ffn_noop.xclbin + _insts.bin.
"""
import os, sys, shutil
from pathlib import Path

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
IRON_PATH = str(PROJ_ROOT.parent / "Source" / "IRON-devel")
if not Path(IRON_PATH).exists() and IRON_PATH not in sys.path:
    pass
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

HV_K      = 5120
GROUPS    = HV_K // 32          # 160
PAD_BYTES = 18
ROW_BYTES = GROUPS * PAD_BYTES  # 2880
RPB       = 4                   # rows per weight packet
N_TILE    = 2176                # 17408 / 8 rows on this tile
WPKT      = RPB * ROW_BYTES     # 11520 B = 2880 float
NPKT      = N_TILE // RPB       # 544 weight packets

SRC = str(PROJ_ROOT / "fst_qwopus_ffn_probe_kernel.cc")  # ffn_noop_rows lives here
INC = [aie_config.cxx_header_path()]
OBJ = "fst_qwopus_ffn_noop.o"

H_T   = np.ndarray[(HV_K,),      np.dtype[np.float32]]
W_T   = np.ndarray[(WPKT // 4,), np.dtype[np.float32]]   # bytes as float
OUT_T = np.ndarray[(N_TILE,),    np.dtype[np.float32]]


@iron.jit
def ffn_probe_op(inp_h: In, inp_w: In, out: Out):
    kM = ExternalFunction("ffn_noop_rows", source_file=SRC,
        arg_types=[W_T, H_T, OUT_T, np.int32], include_dirs=INC, object_file_name=OBJ)

    f_h   = ObjectFifo(H_T,   name="h",   depth=1)   # held read-only
    f_w   = ObjectFifo(W_T,   name="w",   depth=2)   # streamed
    f_out = ObjectFifo(OUT_T, name="out", depth=1)   # held, drained once

    def core(fh, fw, fo, kM):
        h   = fh.acquire(1)                       # held across all weight packets
        o   = fo.acquire(1)                       # held, written incrementally
        for _p in range_(NPKT):
            w = fw.acquire(1)
            kM(w, h, o, RPB)                      # writes RPB results at o[_p*RPB : +RPB]
            fw.release(1)
        fo.release(1)                             # drain all N_TILE results in one S2MM BD
        fh.release(1)

    w = Worker(core, [f_h.cons(), f_w.cons(), f_out.prod(), kM])

    rt = Runtime()
    with rt.sequence(np.ndarray[(HV_K,),           np.dtype[np.float32]],   # inp_h
                     np.ndarray[(NPKT * WPKT // 4,), np.dtype[np.float32]], # inp_w (bytes as float)
                     np.ndarray[(N_TILE,),         np.dtype[np.float32]]) as (inp_h, inp_w, out):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_h.prod(), inp_h,
                tap=TensorAccessPattern(tensor_dims=(1, HV_K), offset=0,
                                        sizes=[1, HV_K], strides=[0, 1]),
                task_group=tg)
        # 544 weight packets × 2880 float (11520 B).  BD repeat ≤255 -> split
        # 544 = 4 × 136.
        NPKT_O, NPKT_I = 4, NPKT // 4
        rt.fill(f_w.prod(), inp_w,
                tap=TensorAccessPattern(tensor_dims=(1, NPKT * WPKT // 4), offset=0,
                                        sizes=[NPKT_O, NPKT_I, WPKT // 4],
                                        strides=[NPKT_I * (WPKT // 4), WPKT // 4, 1]),
                task_group=tg)
        rt.drain(f_out.cons(), out,
                 tap=TensorAccessPattern(tensor_dims=(N_TILE,), offset=0,
                                         sizes=[1, N_TILE], strides=[0, 1]),
                 task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def _emit(jitfn, name):
    xclbin, insts = jitfn.compile()
    shutil.copy(xclbin, f"{name}.xclbin")
    print(f"{name}: {os.path.abspath(name+'.xclbin')} ({os.path.getsize(name+'.xclbin')}B)")
    try:
        shutil.copy(insts, f"{name}_insts.bin")
        print(f"  insts: {os.path.abspath(name+'_insts.bin')}")
    except Exception:
        pass


if __name__ == "__main__":
    _emit(ffn_probe_op, "fst_qwopus_ffn_noop")
    print(f"OK: 1-tile M=1 MXFP4 matvec probe (N_TILE={N_TILE} rows, {NPKT} weight packets)")