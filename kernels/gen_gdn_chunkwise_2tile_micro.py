#!/usr/bin/env python3
"""gen_gdn_chunkwise_2tile_micro.py — Stage 1.1b 2-TILE micro-test IRON generator.

Emits ONE xclbin `fst_gdn_chunkwise_2tile_micro.xclbin` with a single @iron.jit
op containing TWO Workers on TWO compute tiles, joined by a CORE↔CORE
ObjectFifo (W0.prod -> W1.cons) — the mechanism the 2-tile GDN split hinges on
and that is UNPROVEN on this build (every reachable multi-worker example uses
independent per-worker shim DMA, never worker->worker).

  W0 (tile 0): 1 shim MM2S (gdec+rowoff), 0 shim S2MM -> RMWs 16 KB stack,
              emits partial0 via f_cross.prod
  W1 (tile 1): 1 shim MM2S (gdec+rowoff) + 1 shim S2MM (total) -> RMWs 16 KB
              stack, recvs partial0 via f_cross.cons, drains total
  f_cross: core<->core ObjectFifo, depth 2, [8] fp32 (partial0 at [0])

Per-tile shim stays 1 MM2S + <=1 S2MM -> no 2-S2MM shim race (the death of M=1
fusion).  stack_size bumped to 0x9000 (36 KB) to hold the 16 KB stack array +
frame; the 16 KB array itself is well within the [-32768,-64] load/store
immediate (unlike the 32 KB single-tile array that crashed at -32896).

See fst_gdn_chunkwise_2tile_micro_kernel.cc for the test rationale + the
expected total (4128.0, bf16-exact with gdec=0.5).
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

PKT = 8                 # [gdec(1)|row_offset(1)|pad(6)] in; [partial/total(1)|pad(7)] out/cross — 8-aligned
NV  = 48                # v-heads per dispatch (both tiles loop them)
STACK = 0x5000          # 20 KB stack per worker (16 KB Shalf + ~1 KB frame; leaves
                        # ~44 KB of the 64 KB tile for the tiny ObjectFifo buffers)

SRC = str(PROJ_ROOT / "fst_gdn_chunkwise_2tile_micro_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_chunkwise_2tile_micro.o"

PKT_T = np.ndarray[(PKT,), np.dtype[np.float32]]


@iron.jit(aiecc_flags=["--dump-intermediates"])
def gdn_2tile_op(in_w0: iron.In, in_w1: iron.In, out_w1: iron.Out):
    k0 = ExternalFunction("gdn_micro_w0", source_file=SRC,
        arg_types=[PKT_T, PKT_T], include_dirs=INC, object_file_name=OBJ)
    k1 = ExternalFunction("gdn_micro_w1", source_file=SRC,
        arg_types=[PKT_T, PKT_T, PKT_T], include_dirs=INC, object_file_name=OBJ)

    f_in0  = ObjectFifo(PKT_T, name="in0",  depth=2)   # shim->W0 (gdec,rowoff=0)
    f_in1  = ObjectFifo(PKT_T, name="in1",  depth=2)   # shim->W1 (gdec,rowoff=64)
    f_cross = ObjectFifo(PKT_T, name="cross", depth=2) # W0.prod -> W1.cons (core<->core, UNDER TEST)
    f_out1 = ObjectFifo(PKT_T, name="out1", depth=2)   # W1.prod -> shim (total)

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

    w0 = Worker(core_w0, [f_in0.cons(), f_cross.prod(), k0], stack_size=STACK)
    w1 = Worker(core_w1, [f_in1.cons(), f_cross.cons(), f_out1.prod(), k1], stack_size=STACK)

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
    _emit(gdn_2tile_op, "fst_gdn_chunkwise_2tile_micro")
    print("OK: 2-tile micro-test xclbin (W0 tile0 + W1 tile1 + core<->core ObjectFifo)")