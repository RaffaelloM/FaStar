#!/usr/bin/env python3
"""gen_gdn_passB_delta.py — passB with DELTA BROADCAST (Fork B / Phase 1).

Drop the 3.1 MB of replicated delta from the per-row passB packet; instead hold
the per-v-head delta (128 fp32 = 512 B, under the 8 KB fast threshold) on-tile
via a 2nd MM2S shim, fed ONCE per v-head.  Per-row packet shrinks 264 -> 136
fp32 ([S0(128)|kn_i|gdec|pad(6)], %8==0 stride).  Math byte-identical to the
shipped gdn_passB_block (s2 = gdec*S + kn_i*delta); only delta's source changes.

Channels (1 tile, 1 worker): 2 MM2S (f_in + f_delta) + 1 S2MM (f_out) — within
the 2+2 shim budget.  f_delta is a HELD input: acquired once per v-head and
held across that v-head's 16 row-blocks (the proven passA held-accumulator
pattern, but for an input).  depth=2 on all fifos.

Emits fst_gdn_passB_delta.xclbin + fst_gdn_passB_delta_insts.bin.
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

HV = 128
NROWS = 128                      # rows per v-head
NV = 48                          # v-heads per layer (all in 1 dispatch)
ROWS_PER_PKT = 16                # 16 rows/block (delta-bcast shrinks row 264->136
                                  #  so 16×136=2176 fits the 4096-word BD cap; halves
                                  #  the packet count 768->384 vs shipped 8-row/264)
PKT_BD = 136                     # [S0(128)|kn_i@128|gdec@129|pad(6)]  (%8==0)
NPKT_V = NROWS // ROWS_PER_PKT   # 8 blocks per v-head
NPKT = NV * NPKT_V               # 384 blocks total
NPKT_O = 6                       # split 384 = 6 × 64 (BD repeat ≤255)
NPKT_I = NPKT // NPKT_O          # 64
PKT_B_BLK = PKT_BD * ROWS_PER_PKT  # 136*8 = 1088 (8 rows)
OUT_BLK = HV * ROWS_PER_PKT      # 128*8 = 1024 (proven silu drain)

SRC = str(PROJ_ROOT / "fst_gdn_passB_delta_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_passB_delta.o"

BLK_IN_T   = np.ndarray[(PKT_B_BLK,), np.dtype[np.float32]]
DELTA_T    = np.ndarray[(HV,),      np.dtype[np.float32]]
BLK_OUT_T  = np.ndarray[(OUT_BLK,),  np.dtype[np.float32]]


@iron.jit
def gdn_passB_delta_op(Spkt: In, delta_in: In, s2_out: Out):
    kB = ExternalFunction("gdn_passB_block_delta", source_file=SRC,
        arg_types=[BLK_IN_T, BLK_OUT_T, DELTA_T], include_dirs=INC, object_file_name=OBJ)

    f_in    = ObjectFifo(BLK_IN_T,  name="in",    depth=2)  # 1088/pkt (8 rows), 768 pkts
    f_delta = ObjectFifo(DELTA_T,   name="delta", depth=2)  # 128/pkt, 48 pkts (held/v-head)
    f_out   = ObjectFifo(BLK_OUT_T, name="out",   depth=2)  # 1024/pkt, 768 pkts

    def core_passBd(fin, fdelta, fo, kB):
        for _v in range_(NV):                          # 48 v-heads
            delta = fdelta.acquire(1)                  # held across this v-head's 16 blocks
            for _b in range_(NPKT_V):                  # 16 blocks
                p = fin.acquire(1); s2 = fo.acquire(1)
                kB(p, s2, delta)
                fin.release(1); fo.release(1)
            fdelta.release(1)

    w = Worker(core_passBd, [f_in.cons(), f_delta.cons(), f_out.prod(), kB])

    rt = Runtime()
    with rt.sequence(np.ndarray[(NV * NROWS * PKT_BD,), np.dtype[np.float32]],  # Spkt (136-stride)
                     np.ndarray[(NV * HV,),          np.dtype[np.float32]],     # delta_in (48 × 128)
                     np.ndarray[(NV * NROWS * HV,),  np.dtype[np.float32]]) as (Spkt, delta_in, s2_out):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), Spkt,
                tap=TensorAccessPattern(tensor_dims=(1, NV * NROWS * PKT_BD), offset=0,
                                        sizes=[NPKT_O, NPKT_I, PKT_B_BLK],
                                        strides=[NPKT_I * PKT_B_BLK, PKT_B_BLK, 1]),
                task_group=tg)
        # 48 delta packets of 128 (linear; 2-level like gdn_delta's din fill,
        # 48 ≤ 255 BD repeat).
        rt.fill(f_delta.prod(), delta_in,
                tap=TensorAccessPattern(tensor_dims=(1, NV * HV), offset=0,
                                        sizes=[NV, HV],
                                        strides=[HV, 1]),
                task_group=tg)
        rt.drain(f_out.cons(), s2_out,
                 tap=TensorAccessPattern(tensor_dims=(NV * NROWS * HV,), offset=0,
                                         sizes=[NPKT_O, NPKT_I, OUT_BLK],
                                         strides=[NPKT_I * OUT_BLK, OUT_BLK, 1]),
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
    _emit(gdn_passB_delta_op, "fst_gdn_passB_delta")
    print(f"OK: passB delta-broadcast — 48 v-heads, held delta (512 B), 136-stride rows, 1 dispatch")