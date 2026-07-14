#!/usr/bin/env python3
"""gen_qwopus_ffn_noop8.py — 8-TILE no-op diagnostic: measures NPU AGGREGATE
weight-read DDR bandwidth (the number that sets the NPU-FFN ceiling).

8 workers on 8 tiles, each streaming ~6.27 MB of weight packets (total ~50 MB)
with NO h-array access (just a vector load per row to force the DMA read).  The
host probe (tools/qwopus_ffn_noop8.cpp) dispatches once and reports
total_GB / time = aggregate GB/s.  This is THE decisive number for B:
  - aggregate >= ~20 GB/s => NPU can drive DDR near peak => FFN ~4x over host
    (4.7 GB/s) => ~0.3-0.4 tok/s ceiling.
  - aggregate ~5-10 GB/s   => marginal/no win over host.
Each tile also holds an f_h slot (depth-1, unread) to match the real design's
fifo footprint.  Emits fst_qwopus_ffn_noop8.xclbin + _insts.bin.
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
GROUPS    = HV_K // 32
PAD_BYTES = 18
ROW_BYTES = GROUPS * PAD_BYTES
RPB       = 4
N_TILE    = 2176
NPKT      = N_TILE // RPB       # 544 packets per tile
WPKT      = RPB * ROW_BYTES     # 11520 B
H_BYTES   = HV_K * 4
NT        = 8                   # 8 tiles

SRC = str(PROJ_ROOT / "fst_qwopus_ffn_probe_kernel.cc")  # ffn_noop_rows lives here
INC = [aie_config.cxx_header_path()]
OBJ = "fst_qwopus_ffn_noop8.o"

H_T   = np.ndarray[(HV_K,),      np.dtype[np.float32]]
W_T   = np.ndarray[(WPKT // 4,), np.dtype[np.float32]]
OUT_T = np.ndarray[(N_TILE,),    np.dtype[np.float32]]


@iron.jit
def ffn_noop8_op(inp_h: In, inp_w: In, out: Out):
    kM = ExternalFunction("ffn_noop_rows", source_file=SRC,
        arg_types=[W_T, H_T, OUT_T, np.int32], include_dirs=INC, object_file_name=OBJ)

    # One fifo per tile (8 × (1 MM2S h + 1 MM2S w + 1 S2MM out)).  h is held but
    # unread (matches real fifo footprint); w is streamed and vector-loaded.
    fhs = [ObjectFifo(H_T,   name=f"h{t}",   depth=1) for t in range(NT)]
    fws = [ObjectFifo(W_T,   name=f"w{t}",   depth=2) for t in range(NT)]
    fos = [ObjectFifo(OUT_T, name=f"o{t}",   depth=1) for t in range(NT)]

    def core(fh, fw, fo, kM):
        h = fh.acquire(1); o = fo.acquire(1)
        for _p in range_(NPKT):
            w = fw.acquire(1); kM(w, h, o, RPB); fw.release(1)
        fo.release(1); fh.release(1)

    ws = [Worker(core, [fhs[t].cons(), fws[t].cons(), fos[t].prod(), kM]) for t in range(NT)]

    rt = Runtime()
    ONE_W = NPKT * WPKT // 4
    with rt.sequence(np.ndarray[(NT * HV_K,),        np.dtype[np.float32]],  # inp_h (8 × h)
                     np.ndarray[(NT * ONE_W,),       np.dtype[np.float32]],  # inp_w (8 × 6.27MB)
                     np.ndarray[(NT * N_TILE,),      np.dtype[np.float32]]) as (inp_h, inp_w, out):
        rt.start(*ws)
        tg = rt.task_group()
        for t in range(NT):
            rt.fill(fhs[t].prod(), inp_h,
                    tap=TensorAccessPattern(tensor_dims=(1, NT * HV_K), offset=t * HV_K,
                                            sizes=[1, HV_K], strides=[0, 1]), task_group=tg)
            NPKT_O, NPKT_I = 4, NPKT // 4
            rt.fill(fws[t].prod(), inp_w,
                    tap=TensorAccessPattern(tensor_dims=(1, NT * ONE_W), offset=t * ONE_W,
                                            sizes=[NPKT_O, NPKT_I, WPKT // 4],
                                            strides=[NPKT_I * (WPKT // 4), WPKT // 4, 1]),
                    task_group=tg)
            rt.drain(fos[t].cons(), out,
                     tap=TensorAccessPattern(tensor_dims=(NT * N_TILE,), offset=t * N_TILE,
                                             sizes=[1, N_TILE], strides=[0, 1]),
                     task_group=tg, wait=(t == NT - 1))
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
    _emit(ffn_noop8_op, "fst_qwopus_ffn_noop8")
    print(f"OK: 8-tile no-op (each streams {NPKT} pkts = {NPKT*WPKT/1e6:.2f} MB; total {NT*NPKT*WPKT/1e6:.1f} MB)")