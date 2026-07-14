#!/usr/bin/env python3
"""gen_gdn_scratch_test.py — Stage 1.1d: persistent held-scratchpad test.

The full chunkwise kernel needs S to PERSIST across the K=8 per-token C-fn
re-calls (the recurrence state).  A STACK array does NOT persist (the frame is
freed on return; the toy fill re-init each call masked this).  This test uses a
NAMED on-tile memory region (aie.iron.Buffer) as S, passed to the C fn as a
POINTER and RMW'd with load_v/store_v — same plain tile memory as the fast
stack array (NOT the Python-level Buffer RMW that was 600x slow in M=1).

Tests:
  (a) PERSISTENCE: S[0] = 0 (zeroed once), then N re-calls each do
      S[0] = S[0]*gdec + addend with addend=1, gdec=1.0  ->  S[0] = N.  If S
      did NOT persist (re-zeroed each call), the last out[0] would be 1, not N.
  (b) SPEED: per-call latency should be ~comparable to the stack-array RMW
      (1.6 ms), NOT 600x slow.  If slow, the C-kernel-direct-access Buffer is
      still the slow path and the chunkwise kernel is blocked.

Emits fst_gdn_scratch_test.xclbin (1 worker, 1 MM2S + 1 S2MM + 1 persistent Buffer).
"""
import os, shutil
from pathlib import Path
PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PATH", os.environ["PATH"] + ":" + os.path.expanduser("~/.local/bin"))
os.environ.setdefault("PEANO_INSTALL_DIR",
                       os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker, Buffer
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config
set_current_device(NPU2())

SLEN = 8192
N    = 8               # K = 8 re-calls
INPKT  = 2             # [addend, gdec]
OUTPKT = 1             # [S[0]]
SRC = str(PROJ_ROOT / "fst_gdn_scratch_test_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_scratch_test.o"
S_T   = np.ndarray[(SLEN,),    np.dtype[np.float32]]
IN_T  = np.ndarray[(INPKT,),   np.dtype[np.float32]]
OUT_T = np.ndarray[(OUTPKT,),  np.dtype[np.float32]]

@iron.jit
def gdn_scratch_op(inp: iron.In, outp: iron.Out):
    kzero = ExternalFunction("gdn_scratch_zero", source_file=SRC,
        arg_types=[S_T], include_dirs=INC, object_file_name=OBJ)
    kstep = ExternalFunction("gdn_scratch_step", source_file=SRC,
        arg_types=[S_T, IN_T, OUT_T], include_dirs=INC, object_file_name=OBJ)
    S    = Buffer(S_T, name="ScratchS")
    f_in  = ObjectFifo(IN_T,  name="in",  depth=N+2)
    f_out = ObjectFifo(OUT_T, name="out", depth=N+2)
    def core(S, fin, fout, kzero, kstep):
        kzero(S)                                  # init persistent S = 0 (once)
        for _ in range_(N):                       # K=8 re-calls; S persists
            pi = fin.acquire(1); po = fout.acquire(1)
            kstep(S, pi, po)
            fin.release(1); fout.release(1)
    w = Worker(core, [S, f_in.cons(), f_out.prod(), kzero, kstep], stack_size=0x800)
    rt = Runtime()
    with rt.sequence(np.ndarray[(N * INPKT,),  np.dtype[np.float32]],
                     np.ndarray[(N * OUTPKT,), np.dtype[np.float32]]) as (inp, outp):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(N, INPKT), offset=0,
                                        sizes=[N, INPKT], strides=[INPKT, 1]),
                task_group=tg)
        rt.drain(f_out.cons(), outp,
                 tap=TensorAccessPattern(tensor_dims=(N, OUTPKT), offset=0,
                                         sizes=[N, OUTPKT], strides=[OUTPKT, 1]),
                 task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()

def _emit(jitfn, name):
    xclbin, insts = jitfn.compile()
    shutil.copy(xclbin, f"{name}.xclbin"); print(f"{name}: xclbin {os.path.getsize(name+'.xclbin')}B")
    try: shutil.copy(insts, f"{name}_insts.bin")
    except Exception: pass

if __name__ == "__main__":
    _emit(gdn_scratch_op, "fst_gdn_scratch_32k")
    print("OK: persistent held-scratchpad test (Buffer S, K=8 re-calls)")