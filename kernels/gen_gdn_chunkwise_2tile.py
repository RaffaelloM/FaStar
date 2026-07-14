#!/usr/bin/env python3
"""gen_gdn_chunkwise_2tile.py — chunkwise M=K GDN scan for XDNA2 (NPU2),
2-TILE COLUMN-SPLIT + SINGLE-CALL STACK-S design (Stage 1.2-v3).

THE FIX for the persistent-aie.iron.Buffer dead-end (proven x2: 1-tile 32KB and
8-tile 4KB Buffer both ~340x slow — the Buffer's register-addressed access does
NOT pipeline at any size).  The only proven-fast S access is a STACK-LOCAL array
inside a single-call C fn (the 4-tile micro: 5.87ms).  This kernel uses that.

DESIGN — 2 INDEPENDENT tiles (no chain), each 1 MM2S + 1 S2MM = 2+2 shim budget:
  Tile 0 (cols 0..63), Tile 1 (cols 64..127).  Column-split ⇒ recurrence
  INDEPENDENT per tile ⇒ NO cross-tile reduction, NO chain, NO forwarding.
  S[128,64] bf16 = 16KB STACK-LOCAL (fits the stack-immediate limit; no Buffer).

STREAMING — PER-V-HEAD unified packets (the 4-tile-micro acquire(1) pattern
scaled): the worker loops 48 v-heads; per v-head acquires ONE in-packet
(10768 fp32 = [S0(128×64) | par(8×322)]) + ONE out-packet (8704 fp32 =
[y(8×64) | sf(128×64)]), calls gdn_2tile_vhead ONCE (stack-S lives for the
call: 128 load + 8 recur + 128 store), releases.  All accesses are ObjectFifo
tile-local data or stack — both FAST.  No persistent Buffer anywhere.

2× compute parallelism: 18.8M MACs/dispatch ÷ 2 = 9.4M MACs/tile × 2.4cyc =
22.5ms < 30ms syncobj ⇒ syncobj-bound ~30ms ⇒ the 18× premise holds IF the
stack-S pipelines (the de-risk this kernel resolves).

Emits fst_gdn_chunkwise_2tile.xclbin + fst_gdn_chunkwise_2tile_insts.bin.
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
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

HV     = 128
K      = 8
NVH    = 48
COLS   = 64
IN_PAR = 322
IN_PKT = HV * COLS + K * IN_PAR     # 8192 + 2576 = 10768
OUT_PKT = K * COLS + HV * COLS      # 512 + 8192 = 8704

SRC = str(PROJ_ROOT / "fst_gdn_chunkwise_2tile_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_chunkwise_2tile.o"

# bf16 unified packets: in 21KB + out 17KB + 20KB stack = 58KB fits the ~64KB
# tile data memory (fp32 packets overflowed at 96KB).  S stays stack bf16,
# recur stays fp32; kn/qn/v/gdec/beta are bf16 on DMA, converted on-tile.
IN_T  = np.ndarray[(IN_PKT,),  np.dtype[bfloat16]]
OUT_T = np.ndarray[(OUT_PKT,), np.dtype[bfloat16]]

# AIE BD max length is 4096 words; split each >4096-float packet into ≤4096
# chunks so the TAP lowers to legal BDs.  10768 = 4×2692; 8704 = 4×2176.
IN_CHUNK  = IN_PKT  // 4     # 2692
OUT_CHUNK = OUT_PKT // 4     # 2176


@iron.jit
def gdn_2tile_op(inp: iron.In, out: iron.Out):
    kv = ExternalFunction("gdn_2tile_vhead", source_file=SRC,
                          arg_types=[IN_T, OUT_T], include_dirs=INC, object_file_name=OBJ)

    # 2 independent tiles: each 1 MM2S (in) + 1 S2MM (out).  depth=1: the
    # 42KB in / 34KB out per-v-head packets + 20KB stack = 96KB fits the ~128KB
    # tile data memory (depth-2 = 172KB overflows).  DMA fill is ~5µs/packet so
    # no-overlap costs only ~0.24ms/48 v-heads — compute dominates anyway.
    f_in0  = ObjectFifo(IN_T,  name="in0",  depth=1)
    f_out0 = ObjectFifo(OUT_T, name="out0", depth=1)
    f_in1  = ObjectFifo(IN_T,  name="in1",  depth=1)
    f_out1 = ObjectFifo(OUT_T, name="out1", depth=1)

    def core(fin, fout, kv):
        for _v in range_(NVH):                  # 48 v-heads, 1 dispatch
            ip = fin.acquire(1);  op = fout.acquire(1)
            kv(ip, op)                          # one call: stack-S load→8 recur→store
            fin.release(1); fout.release(1)

    # stack_size: gdn_2tile_vhead holds S[128*64] bf16 = 16KB + recur_core's
    # parf[328](1.3KB) + a/b/delta(768B) ≈ 18KB live.  Tile data mem ~64KB total,
    # fifos take 21KB(in)+17KB(out)=38KB → ~26KB for stack.  0x6000 (24KB) gives
    # 6KB spill headroom over the 18KB live data and still fits; 0x10000 overflows.
    w0 = Worker(core, [f_in0.cons(), f_out0.prod(), kv], stack_size=0x6000)
    w1 = Worker(core, [f_in1.cons(), f_out1.prod(), kv], stack_size=0x6000)

    rt = Runtime()
    HALF = NVH * IN_PKT                       # 48 × 10768 bf16 per tile (in)
    HALF_O = NVH * OUT_PKT                    # 48 × 8704  bf16 per tile (out)
    with rt.sequence(np.ndarray[(2 * HALF,),   np.dtype[bfloat16]],   # inp [tile0 | tile1]
                     np.ndarray[(2 * HALF_O,), np.dtype[bfloat16]]) as (inp, out):
        rt.start(w0, w1)
        tg = rt.task_group()
        # in fill: per tile 48 packets × 10768, split as [48, 4, 2692] (≤4096).
        rt.fill(f_in0.prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(1, 2 * HALF), offset=0,
                                        sizes=[NVH, 4, IN_CHUNK],
                                        strides=[IN_PKT, IN_CHUNK, 1]),
                task_group=tg)
        rt.fill(f_in1.prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(1, 2 * HALF), offset=HALF,
                                        sizes=[NVH, 4, IN_CHUNK],
                                        strides=[IN_PKT, IN_CHUNK, 1]),
                task_group=tg)
        # out drain: per tile 48 packets × 8704, split as [48, 4, 2176].
        rt.drain(f_out0.cons(), out,
                 tap=TensorAccessPattern(tensor_dims=(2 * HALF_O,), offset=0,
                                         sizes=[NVH, 4, OUT_CHUNK],
                                         strides=[OUT_PKT, OUT_CHUNK, 1]),
                 task_group=tg)
        rt.drain(f_out1.cons(), out,
                 tap=TensorAccessPattern(tensor_dims=(2 * HALF_O,), offset=HALF_O,
                                         sizes=[NVH, 4, OUT_CHUNK],
                                         strides=[OUT_PKT, OUT_CHUNK, 1]),
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
    _emit(gdn_2tile_op, "fst_gdn_chunkwise_2tile")
    print("OK: 2-tile column-split chunkwise M=K GDN (single-call stack-S, K=8, 48 v-heads)")