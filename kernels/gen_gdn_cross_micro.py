#!/usr/bin/env python3
"""gen_gdn_cross_micro.py — MINIMAL core↔core ObjectFifo IRON generator.

Emits `fst_gdn_cross_micro.xclbin`: ONE @iron.jit op, TWO Workers on TWO
compute tiles joined by a CORE↔CORE ObjectFifo (W0.prod -> W1.cons) — the
mechanism the 2-tile GDN split hinges on and that is UNPROVEN on this build.

This is the CHEAP, stack-free isolation of that risk: the kernels
(gdn_cross_w0/w1) use only tiny register loops (NO 16 KB stack array), so the
compile cannot hit the AIE2P bank-allocation wall that blocks the full 16 KB
2-tile micro-test.  The ONLY thing this verifies is:

  (a) a worker->worker ObjectFifo actually transfers on NPU2 here, AND
  (b) the per-tile shim pattern (1 MM2S + <=1 S2MM, no 2-S2MM race) is sound.

If this passes (total == 4128.0 for all 48 v-heads), the core↔core mechanism
is PROVEN and the remaining obstacle is the 16 KB stack array alone (to be
solved separately — likely a 4-tile 8 KB split).  If this fails, EVERY tile
split (2- or 4-tile) is dead and the structural floor stands.

Layout (mirrors gen_gdn_chunkwise_2tile_micro.py minus the big stacks):
  f_in0  shim MM2S -> W0 ; f_in1 shim MM2S -> W1
  f_cross core<->core (W0.prod -> W1.cons)  <-- UNDER TEST
  f_out1 W1.prod -> shim S2MM
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
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

PKT = 8                 # [gdec(1)|row_offset(1)|pad(6)] in; [partial/total(1)|pad(7)] cross/out — 8-aligned
NV  = 48                # v-heads per dispatch (both tiles loop them)

SRC = str(PROJ_ROOT / "fst_gdn_cross_micro_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_cross_micro.o"

PKT_T = np.ndarray[(PKT,), np.dtype[np.float32]]


@iron.jit
def gdn_cross_op(in_w0: iron.In, in_w1: iron.In, out_w1: iron.Out):
    k0 = ExternalFunction("gdn_cross_w0", source_file=SRC,
        arg_types=[PKT_T, PKT_T], include_dirs=INC, object_file_name=OBJ)
    k1 = ExternalFunction("gdn_cross_w1", source_file=SRC,
        arg_types=[PKT_T, PKT_T, PKT_T], include_dirs=INC, object_file_name=OBJ)

    f_in0  = ObjectFifo(PKT_T, name="in0",   depth=2)   # shim->W0 (gdec,rowoff=0)
    f_in1  = ObjectFifo(PKT_T, name="in1",   depth=2)   # shim->W1 (gdec,rowoff=64)
    f_cross = ObjectFifo(PKT_T, name="cross", depth=2)  # W0.prod -> W1.cons (core<->core, UNDER TEST)
    f_out1 = ObjectFifo(PKT_T, name="out1",  depth=2)  # W1.prod -> shim (total)

    def core_w0(fin, fxp, k0):
        for _v in range_(NV):
            p_in = fin.acquire(1)
            p_x  = fxp.acquire(1)
            k0(p_in, p_x)
            fin.release(1); fxp.release(1)

    def core_w1(fin, fxc, fout, k1):
        for _v in range_(NV):
            p_in  = fin.acquire(1)
            p_xc  = fxc.acquire(1)
            p_out = fout.acquire(1)
            k1(p_in, p_xc, p_out)
            fin.release(1); fxc.release(1); fout.release(1)

    w0 = Worker(core_w0, [f_in0.cons(), f_cross.prod(), k0])
    w1 = Worker(core_w1, [f_in1.cons(), f_cross.cons(), f_out1.prod(), k1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(NV * PKT,), np.dtype[np.float32]],   # in_w0
                     np.ndarray[(NV * PKT,), np.dtype[np.float32]],   # in_w1
                     np.ndarray[(NV * PKT,), np.dtype[np.float32]]) as (in_w0, in_w1, out_w1):
        rt.start(w0, w1)
        tg = rt.task_group()
        rt.fill(f_in0.prod(), in_w0,
                tap=TensorAccessPattern(tensor_dims=(1, NV * PKT), offset=0,
                                        sizes=[NV, PKT], strides=[PKT, 1]),
                task_group=tg)
        rt.fill(f_in1.prod(), in_w1,
                tap=TensorAccessPattern(tensor_dims=(1, NV * PKT), offset=0,
                                        sizes=[NV, PKT], strides=[PKT, 1]),
                task_group=tg)
        rt.drain(f_out1.cons(), out_w1,
                 tap=TensorAccessPattern(tensor_dims=(NV * PKT,), offset=0,
                                         sizes=[1, NV, PKT], strides=[0, PKT, 1]),
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
    _emit(gdn_cross_op, "fst_gdn_cross_micro")
    print("OK: minimal core<->core ObjectFifo micro-test xclbin")