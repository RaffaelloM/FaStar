#!/usr/bin/env python3
"""gen_ffn_nibprobe.py — A3 vector-nibble correctness probe generator.

1-tile kernel.  Input is a FLOAT-typed ObjectFifo (32 floats = 128 B); the
first 16 B are 16 raw MXFP4 weight bytes.  The kernel dequants them to 32 FP4
floats ([16 low-nibble, 16 high-nibble]) and writes them to a float output fifo.

Emits TWO xclbins sharing the geometry so the probe compares vector vs scalar
dequant on the SAME float fifo:
  fst_ffn_nibprobe_vec.xclbin — nib_vec  (load_v + bit_and + lut<4,float>)
  fst_ffn_nibprobe_sca.xclbin — nib_sca  (scalar reads; expected garbage)
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

PKT_FLOATS = 32        # 128 B in; first 16 B = 16 weight bytes
OUT_FLOATS = 32        # 32 nibbles dequanted to float
NPKT       = 16
SRC = str(PROJ_ROOT / "fst_ffn_nibprobe_kernel.cc")
INC = [aie_config.cxx_header_path()]

PKT_T = np.ndarray[(PKT_FLOATS,), np.dtype[np.float32]]
OUT_T = np.ndarray[(OUT_FLOATS,), np.dtype[np.float32]]


def _build(fn_name, xclbin_name, obj_name):
    @iron.jit
    def nib_op(inp: In, out: Out):
        k = ExternalFunction(fn_name, source_file=SRC,
            arg_types=[PKT_T, OUT_T], include_dirs=INC, object_file_name=obj_name)

        f_in  = ObjectFifo(PKT_T, name="in",  depth=2)
        f_out = ObjectFifo(OUT_T, name="out", depth=2)

        def core(fin, fo, k):
            for _ in range_(NPKT):
                p = fin.acquire(1)
                o = fo.acquire(1)
                k(p, o)
                fin.release(1)
                fo.release(1)

        w = Worker(core, [f_in.cons(), f_out.prod(), k])

        rt = Runtime()
        with rt.sequence(np.ndarray[(NPKT * PKT_FLOATS,), np.dtype[np.float32]],
                         np.ndarray[(NPKT * OUT_FLOATS,), np.dtype[np.float32]]) as (inp, out):
            rt.start(w)
            tg = rt.task_group()
            rt.fill(f_in.prod(), inp,
                    tap=TensorAccessPattern(tensor_dims=(1, NPKT * PKT_FLOATS),
                                            offset=0,
                                            sizes=[NPKT, PKT_FLOATS],
                                            strides=[PKT_FLOATS, 1]),
                    task_group=tg)
            rt.drain(f_out.cons(), out,
                     tap=TensorAccessPattern(tensor_dims=(NPKT * OUT_FLOATS,),
                                              offset=0,
                                              sizes=[NPKT, OUT_FLOATS],
                                              strides=[OUT_FLOATS, 1]),
                     task_group=tg, wait=True)
            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()

    x, i = nib_op.compile()
    shutil.copy(x, f"{xclbin_name}.xclbin")
    shutil.copy(i, f"{xclbin_name}_insts.bin")
    print(f"{xclbin_name}: {os.path.abspath(xclbin_name + '.xclbin')} "
          f"({os.path.getsize(xclbin_name + '.xclbin')}B)")


if __name__ == "__main__":
    # Build each variant in a SEPARATE process — @iron.jit caches the resolved
    # program in-process and the fn_name closure is NOT part of the cache key, so
    # building vec + sca in one process yields TWO IDENTICAL xclbins (both = the
    # first build).  Selecting one per invocation busts the cache correctly.
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "vec"):
        _build("nib_vec", "fst_ffn_nibprobe_vec", "fst_ffn_nibprobe_vec.o")
    if which in ("both", "sca"):
        _build("nib_sca", "fst_ffn_nibprobe_sca", "fst_ffn_nibprobe_sca.o")
    print(f"OK nibprobe [{which}] — pkt={PKT_FLOATS} floats (first 16 B = weights), "
          f"{NPKT} pkts")