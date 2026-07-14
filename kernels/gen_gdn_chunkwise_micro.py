#!/usr/bin/env python3
"""gen_gdn_chunkwise_micro.py — Stage 1.1 MICRO-TEST IRON generator.

Emits ONE xclbin `fst_gdn_chunkwise_micro.xclbin` with a single @iron.jit op:
1 Worker loops the 48 v-heads, calling the C kernel `gdn_micro_vhead` once per
v-head with a tiny [8] fp32 input packet (gdec) and a tiny [8] fp32 output
packet (checksum).  1 MM2S (f_in) + 1 S2MM (f_out), depth-2 each.

This is deliberately minimal: it exists to answer the #1 risk of the chunkwise
M=K GDN kernel — is a stack-local bfloat16 Sbuf[128*128] (32 KB) RMW'd via
aie::load_v/store_v FAST (~30 ms, NOT 60 s) and CLEAN — BEFORE authoring the
full kernel.  See fst_gdn_chunkwise_micro_kernel.cc for the test's rationale.
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

NV   = 48          # v-heads processed in ONE dispatch
PKT  = 8           # [gdec(1)|pad(7)] in, [checksum(1)|pad(7)] out — 8-aligned

SRC = str(PROJ_ROOT / "fst_gdn_chunkwise_micro_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_chunkwise_micro.o"

PKT_T = np.ndarray[(PKT,), np.dtype[np.float32]]


@iron.jit
def gdn_micro_op(inp: In, outp: Out):
    kM = ExternalFunction("gdn_micro_vhead", source_file=SRC,
        arg_types=[PKT_T, PKT_T], include_dirs=INC, object_file_name=OBJ)

    f_in  = ObjectFifo(PKT_T, name="in",  depth=2)   # shim->core, [gdec|pad]
    f_out = ObjectFifo(PKT_T, name="out", depth=2)   # core->shim, [checksum|pad]

    def core_micro(fin, fout, kM):
        for _v in range_(NV):                        # 48 v-heads, 1 dispatch
            p_in  = fin.acquire(1)
            p_out = fout.acquire(1)
            kM(p_in, p_out)
            fin.release(1)
            fout.release(1)

    w = Worker(core_micro, [f_in.cons(), f_out.prod(), kM])

    rt = Runtime()
    with rt.sequence(np.ndarray[(NV * PKT,), np.dtype[np.float32]],
                     np.ndarray[(NV * PKT,), np.dtype[np.float32]]) as (inp, outp):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(1, NV * PKT), offset=0,
                                        sizes=[NV, PKT], strides=[PKT, 1]),
                task_group=tg)
        rt.drain(f_out.cons(), outp,
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
    _emit(gdn_micro_op, "fst_gdn_chunkwise_micro")
    print("OK: 1 micro-test xclbin, 48 v-heads per dispatch, stack bf16 RMW speed test")