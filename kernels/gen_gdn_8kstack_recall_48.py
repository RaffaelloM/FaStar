#!/usr/bin/env python3
"""gen_gdn_8kstack_recall.py — DECISIVE re-call test for the chunkwise kernel.

The full chunkwise GDN kernel has a PER-TOKEN recurrent cross-tile reduction
(passA a,b = S^T@kn, S^T@qn reduced across tiles; delta broadcast back).  That
per-token handshaking requires the worker core to drive per-token ObjectFifo
ops, i.e. call the 8 KB-frame C function ONCE PER TOKEN (K=8 re-calls).  The
single-call internal-loop pattern (PROVEN in gen_gdn_8kstack_micro.py at N=1)
does NOT extend to the per-token recurrent case: batching all K tokens into one
acquire deadlocks (token t+1's passA needs passB(t)'s S, which needs delta(t)
from the chain, which can't arrive until the chain processes token t).

So the full kernel's viability hinges on whether the 8 KB-frame function
SURVIVES K=8 re-calls.  We know N=1 works (single-call internal-loop test) and
N=48 breaks (the original repeated-call bug, SP/restore fails after iter 0).
The threshold is unknown.  K=8 is the plan's value.

This test = gen_gdn_8kstack_micro.py with range_(1) -> range_(8) and 8 packets:
the worker core calls gdn_8k_vhead (8 KB stack frame, 48 v-heads internally)
EIGHT TIMES.  If all 8 outputs == 264.0, the 8 KB frame survives 8 re-calls and
the per-token worker-driven chunkwise kernel is viable.  If later calls
corrupt (the N=48 signature), the chunkwise path is dead at K=8 and a different
approach (raw-MLIR aie.buffer, or accept the floor) is needed.
"""
import os, shutil
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

NVH = 48; PKT = 8; TOTAL = NVH * PKT   # one 384-float packet per call
NCALL = 48                              # K = 8 re-calls (the plan's value)
SRC = str(PROJ_ROOT / "fst_gdn_8kstack_micro_kernel.cc")   # reuse the proven kernel
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_8kstack_recall.o"
T = np.ndarray[(TOTAL,), np.dtype[np.float32]]

@iron.jit
def gdn_8k_recall_op(inp: iron.In, outp: iron.Out):
    k = ExternalFunction("gdn_8k_vhead", source_file=SRC, arg_types=[T, T],
                          include_dirs=INC, object_file_name=OBJ)
    f_in  = ObjectFifo(T, name="in",  depth=NCALL+2)
    f_out = ObjectFifo(T, name="out", depth=NCALL+2)
    def core(fin, fout, k):
        for _ in range_(NCALL):                       # 8 re-calls of the 8 KB-frame fn
            pi = fin.acquire(1); po = fout.acquire(1)
            k(pi, po)
            fin.release(1); fout.release(1)
    w = Worker(core, [f_in.cons(), f_out.prod(), k], stack_size=0x3000)
    rt = Runtime()
    with rt.sequence(np.ndarray[(NCALL * TOTAL,), np.dtype[np.float32]],
                     np.ndarray[(NCALL * TOTAL,), np.dtype[np.float32]]) as (inp, outp):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(NCALL, TOTAL), offset=0,
                                        sizes=[NCALL, TOTAL], strides=[TOTAL, 1]),
                task_group=tg)
        rt.drain(f_out.cons(), outp,
                 tap=TensorAccessPattern(tensor_dims=(NCALL, TOTAL), offset=0,
                                         sizes=[NCALL, TOTAL], strides=[TOTAL, 1]),
                 task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()

def _emit(jitfn, name):
    xclbin, insts = jitfn.compile()
    shutil.copy(xclbin, f"{name}.xclbin"); print(f"{name}: xclbin {os.path.getsize(name+'.xclbin')}B")
    try: shutil.copy(insts, f"{name}_insts.bin")
    except Exception: pass

if __name__ == "__main__":
    _emit(gdn_8k_recall_op, "fst_gdn_8kstack_recall_48")
    print("OK: 8 KB-frame fn re-called 8x (K=8 re-call viability test)")