#!/usr/bin/env python3
"""gen_gdn_chunkwise_4tile_micro.py — Stage 1.1c 4-TILE 8 KB-stack
INTERNAL-LOOP micro-test (decisive 4-tile reduction test).

Emits `fst_gdn_chunkwise_4tile_micro.xclbin`: ONE @iron.jit op, FOUR Workers on
FOUR compute tiles joined by a 3-hop CORE↔CORE ObjectFifo chain, where EACH
worker is a SINGLE-CALL INTERNAL-LOOP worker (calls its C function ONCE; the C
function loops the 48 v-heads internally with an 8 KB stack-local S frame
entered only once per tile).

This combines the three proven pieces:
  (a) 8 KB stack-local bfloat16 S RMW is FAST + CLEAN across 48 internal
      iterations (PROVEN: 8kstack single-tile test, 1.6 ms, all 48 == 264.0).
  (b) the SINGLE-CALL INTERNAL-LOOP IRON pattern (the worker calls the external
      fn once; the 8 KB frame is entered once — avoids the repeated-large-frame-
      call breakage that killed the per-v-head 3-hop chain).
  (c) the 3-hop core↔core chain (1 hop PROVEN PASS in the minimal cross test;
      here it carries ONE 96-float packet per tile pair, not per-v-head).

Shim discipline = the PROVEN minimal-cross pattern: ONE shim MM2S (f_in0 -> T0)
+ ONE shim S2MM (T3 -> f_out3).  T1/T2 are compute-only (core↔core fifos, no
shim).  This avoids the 4-separate-MM2S shim overload that threw
"qds_device::wait() unexpected command state".

Cross packet layout (96 floats):  [ gdec(48) | partials(48) ]
  gdec rides the chain (each tile reads gdec[v]); the running partial sum
  accumulates one tile's contribution per hop.  T0 seeds from the shim input
  (gdec per v-head); T3 drains the 48 totals to the shim output.

Expected total == 4128.0 for all 48 v-heads (bf16-exact, gdec=0.5, K=8 RMW;
partial0=264, +776, +1288, +1800 = 4128.0).
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

INPKT  = 384          # shim input  = 48 v-heads x 8 floats  (gdec per v-head)
OUTPKT = 384          # shim output = 48 v-heads x 8 floats  (total  per v-head)
CROSS  = 96           # core<->core = [gdec(48) | partials(48)]
DEPTH  = 4

SRC = str(PROJ_ROOT / "fst_gdn_chunkwise_4tile_micro_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_chunkwise_4tile_micro.o"

IN_T   = np.ndarray[(INPKT,),  np.dtype[np.float32]]
OUT_T  = np.ndarray[(OUTPKT,), np.dtype[np.float32]]
CROSS_T= np.ndarray[(CROSS,),  np.dtype[np.float32]]


@iron.jit
def gdn_4tile_op(in0: iron.In, out3: iron.Out):
    k0 = ExternalFunction("gdn_micro_t0", source_file=SRC,
        arg_types=[IN_T, CROSS_T], include_dirs=INC, object_file_name=OBJ)
    k1 = ExternalFunction("gdn_micro_t1", source_file=SRC,
        arg_types=[CROSS_T, CROSS_T], include_dirs=INC, object_file_name=OBJ)
    k2 = ExternalFunction("gdn_micro_t2", source_file=SRC,
        arg_types=[CROSS_T, CROSS_T], include_dirs=INC, object_file_name=OBJ)
    k3 = ExternalFunction("gdn_micro_t3", source_file=SRC,
        arg_types=[CROSS_T, OUT_T], include_dirs=INC, object_file_name=OBJ)

    f_in0  = ObjectFifo(IN_T,    name="in0",  depth=DEPTH)   # shim MM2S -> T0
    f_c01  = ObjectFifo(CROSS_T, name="c01",  depth=DEPTH)   # T0.prod -> T1.cons
    f_c12  = ObjectFifo(CROSS_T, name="c12",  depth=DEPTH)   # T1.prod -> T2.cons
    f_c23  = ObjectFifo(CROSS_T, name="c23",  depth=DEPTH)   # T2.prod -> T3.cons
    f_out3 = ObjectFifo(OUT_T,   name="out3", depth=DEPTH)   # T3.prod -> shim S2MM

    # ONE call per tile (range_(1)); the C function loops 48 v-heads internally.
    def core_t0(fin, fc, k0):
        for _ in range_(1):
            pi = fin.acquire(1); px = fc.acquire(1)
            k0(pi, px)
            fin.release(1); fc.release(1)

    def core_t1(fci, fco, k1):
        for _ in range_(1):
            pc = fci.acquire(1); px = fco.acquire(1)
            k1(pc, px)
            fci.release(1); fco.release(1)

    def core_t2(fci, fco, k2):
        for _ in range_(1):
            pc = fci.acquire(1); px = fco.acquire(1)
            k2(pc, px)
            fci.release(1); fco.release(1)

    def core_t3(fci, fout, k3):
        for _ in range_(1):
            pc = fci.acquire(1); po = fout.acquire(1)
            k3(pc, po)
            fci.release(1); fout.release(1)

    w0 = Worker(core_t0, [f_in0.cons(),  f_c01.prod(),  k0], stack_size=0x3000)
    w1 = Worker(core_t1, [f_c01.cons(),   f_c12.prod(),  k1], stack_size=0x3000)
    w2 = Worker(core_t2, [f_c12.cons(),   f_c23.prod(),  k2], stack_size=0x3000)
    w3 = Worker(core_t3, [f_c23.cons(),   f_out3.prod(), k3], stack_size=0x3000)

    rt = Runtime()
    with rt.sequence(np.ndarray[(INPKT,), np.dtype[np.float32]],
                     np.ndarray[(OUTPKT,), np.dtype[np.float32]]) as (in0, out3):
        rt.start(w0, w1, w2, w3)
        tg = rt.task_group()
        rt.fill(f_in0.prod(), in0,
                tap=TensorAccessPattern(tensor_dims=(1, INPKT), offset=0,
                                        sizes=[1, INPKT], strides=[INPKT, 1]),
                task_group=tg)
        rt.drain(f_out3.cons(), out3,
                 tap=TensorAccessPattern(tensor_dims=(OUTPKT,), offset=0,
                                         sizes=[1, OUTPKT], strides=[0, 1]),
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
    _emit(gdn_4tile_op, "fst_gdn_chunkwise_4tile_micro")
    print("OK: 4-tile 8KB-stack INTERNAL-LOOP micro-test xclbin (1 MM2S + 1 S2MM, 3-hop chain)")