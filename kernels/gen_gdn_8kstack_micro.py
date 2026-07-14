#!/usr/bin/env python3
"""gen_gdn_8kstack_micro.py — single-call 8 KB-stack INTERNAL-LOOP test.
One dispatch, one shim MM2S + one shim S2MM, ONE call to the C function which
loops the 48 v-heads INTERNALLY (the full chunkwise kernel structure: the 8 KB
stack frame is entered once, so the SP does not restore between v-heads —
avoids the repeated-large-frame-call breakage).  PKT = 48*8 = 384, NV = 1.
Expected partial == 264.0 for all 48 v-heads (gdec=0.5, K=8, bit-exact).
"""
import os, sys, shutil
from pathlib import Path
PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PATH", os.environ["PATH"] + ":" + os.path.expanduser("~/.local/bin"))
os.environ.setdefault("PEANO_INSTALL_DIR",
                       os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config
set_current_device(NPU2())

NVH = 48; PKT = 8; TOTAL = NVH * PKT   # one 384-float packet
SRC = str(PROJ_ROOT / "fst_gdn_8kstack_micro_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_8kstack_micro.o"
T = np.ndarray[(TOTAL,), np.dtype[np.float32]]

@iron.jit
def gdn_8k_op(inp: iron.In, outp: iron.Out):
    k = ExternalFunction("gdn_8k_vhead", source_file=SRC, arg_types=[T, T],
                          include_dirs=INC, object_file_name=OBJ)
    f_in  = ObjectFifo(T, name="in",  depth=2)
    f_out = ObjectFifo(T, name="out", depth=2)
    def core(fin, fout, k):
        for _ in range_(1):
            pi = fin.acquire(1); po = fout.acquire(1)
            k(pi, po)
            fin.release(1); fout.release(1)
    w = Worker(core, [f_in.cons(), f_out.prod(), k], stack_size=0x3000)
    rt = Runtime()
    with rt.sequence(np.ndarray[(TOTAL,), np.dtype[np.float32]],
                     np.ndarray[(TOTAL,), np.dtype[np.float32]]) as (inp, outp):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(1, TOTAL), offset=0, sizes=[1, TOTAL], strides=[TOTAL, 1]),
                task_group=tg)
        rt.drain(f_out.cons(), outp,
                 tap=TensorAccessPattern(tensor_dims=(TOTAL,), offset=0, sizes=[1, TOTAL], strides=[0, 1]),
                 task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()

def _emit(jitfn, name):
    xclbin, insts = jitfn.compile()
    shutil.copy(xclbin, f"{name}.xclbin"); print(f"{name}: xclbin {os.path.getsize(name+'.xclbin')}B")
    try: shutil.copy(insts, f"{name}_insts.bin")
    except Exception: pass

if __name__ == "__main__":
    _emit(gdn_8k_op, "fst_gdn_8kstack_micro")
    print("OK: single-call 8KB-stack internal-loop (48 v-heads in one call)")