#!/usr/bin/env python3
"""gen_gdn_8kslab.py — chunkwise M=K GDN scan for XDNA2 (NPU2), 4-TILE COLUMN-SPLIT
+ SINGLE-CALL 8 KB-STACK-S design (the engine kernel authorized after the
full-recurrence 8 KB-stack probe measured 202.77 ns/vec — FAST).

See kernels/fst_gdn_8kslab_kernel.cc + docs/QWOPUS_GDN_8KSLAB_FULL_PROBE_POSTMORTEM.md.

4 INDEPENDENT tiles (no chain), each holding one 32-col bf16-S slab as an 8 KB
STACK-LOCAL array (the probe's proven-fast access pattern).  Per-tile shim =
exactly 2 MM2S + 2 S2MM:
  MM2S-1  s0   : S0_slab  [HV*COLS]   bf16 = 8 KB   (initial state)
  MM2S-2  par  : K × [kn|qn|v_slab|gdec|beta|c] bf16              (params)
  S2MM-1  snew : S_new_slab [HV*COLS] bf16 = 8 KB   (final state)
  S2MM-2  yd   : K × [y_slab|delta_slab] bf16       (outputs)
Each fifo is 48 elements (one per v-head); the worker acquires s0+par, calls
gdn_8kslab_vhead ONCE, releases snew+yd.  c = kn·qn is host-computed + packed.

All per-element sizes are <= 4096 words (s0/snew=4096, par=2328, yd=512), so every
TAP lowers to a single BD — no 4-way chunk split (unlike the unified 2-tile packet).

Emits fst_gdn_8kslab.xclbin + fst_gdn_8kslab_insts.bin.
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

HV      = 128
K       = 8
NVH     = 48
COLS    = 32
NTILE   = int(os.environ.get("FST_GDN_NTILE", "4"))   # 4=engine, 1=isolation
SUFFIX  = os.environ.get("FST_GDN_SUFFIX", "" if NTILE == 4 else "_1tile")
PAR_SZ  = 2 * HV + COLS + 8      # 296 bf16 per token (padded to mult of 8 for alignment)
YD_SZ   = 2 * COLS              # 64  bf16 per token

S0_ELEM  = HV * COLS             # 4096 bf16 (8 KB)  — s0 / snew element
PAR_ELEM = K * PAR_SZ            # 2328 bf16         — par element
YD_ELEM  = K * YD_SZ             # 512  bf16         — yd  element

SRC = os.environ.get("FST_GDN_SRC", str(PROJ_ROOT / "fst_gdn_8kslab_kernel.cc"))
INC = [aie_config.cxx_header_path()]
OBJ = f"fst_gdn_8kslab{SUFFIX}.o"

S0_T  = np.ndarray[(S0_ELEM,),  np.dtype[bfloat16]]
PAR_T = np.ndarray[(PAR_ELEM,), np.dtype[bfloat16]]
SN_T  = np.ndarray[(S0_ELEM,),  np.dtype[bfloat16]]
YD_T  = np.ndarray[(YD_ELEM,),  np.dtype[bfloat16]]


@iron.jit
def gdn_8kslab_op(s0_bo: iron.In, par_bo: iron.In, snew_bo: iron.Out, yd_bo: iron.Out):
    kv = ExternalFunction("gdn_8kslab_vhead", source_file=SRC,
                          arg_types=[S0_T, PAR_T, SN_T, YD_T],
                          include_dirs=INC, object_file_name=OBJ)

    # 4 tiles × (2 MM2S + 2 S2MM).  depth=2 for DMA/compute overlap; per-element
    # sizes (8 KB / 4.6 KB / 8 KB / 1 KB) × 2 ≈ 43 KB + 12 KB stack ≈ 55 KB fits
    # the tile data memory.  Drop to depth=1 if placement overflows.
    f_s0 = [ObjectFifo(S0_T,  name=f"s0{t}",   depth=2) for t in range(NTILE)]
    f_par= [ObjectFifo(PAR_T, name=f"par{t}",  depth=2) for t in range(NTILE)]
    f_sn = [ObjectFifo(SN_T,   name=f"snew{t}", depth=2) for t in range(NTILE)]
    f_yd = [ObjectFifo(YD_T,   name=f"yd{t}",  depth=2) for t in range(NTILE)]

    def core(fs0, fpar, fsn, fyd, kv):
        for _v in range_(NVH):                  # 48 v-heads, 1 dispatch per tile
            s0 = fs0.acquire(1); par = fpar.acquire(1)
            sn = fsn.acquire(1); yd = fyd.acquire(1)
            kv(s0, par, sn, yd)                 # one call: stack-S load -> K recur -> store
            fs0.release(1); fpar.release(1); fsn.release(1); fyd.release(1)

    # stack_size=0x3000 (12 KB): 8 KB S + ~1.6 KB recur_core frame (parf 1184B +
    # a/b/delta 384B) fits; 0x5000 overlaps the ObjectFifo buffer at 0x4000.
    workers = [Worker(core, [f_s0[t].cons(), f_par[t].cons(),
                             f_sn[t].prod(), f_yd[t].prod(), kv], stack_size=0x3000)
               for t in range(NTILE)]

    rt = Runtime()
    # per-tile contiguous blocks in each BO: [tile0: 48×ELEM][tile1: ...][tile3: ...]
    S0_BLK  = NVH * S0_ELEM       # 48 × 4096
    PAR_BLK = NVH * PAR_ELEM      # 48 × 2328
    YD_BLK  = NVH * YD_ELEM       # 48 × 512
    with rt.sequence(np.ndarray[(NTILE * S0_BLK,),  np.dtype[bfloat16]],   # s0  [t0|t1|t2|t3]
                     np.ndarray[(NTILE * PAR_BLK,), np.dtype[bfloat16]],   # par [t0|t1|t2|t3]
                     np.ndarray[(NTILE * S0_BLK,),  np.dtype[bfloat16]],   # snew[t0|t1|t2|t3]
                     np.ndarray[(NTILE * YD_BLK,),  np.dtype[bfloat16]]) as (s0_bo, par_bo, snew_bo, yd_bo):
        rt.start(*workers)
        tg = rt.task_group()
        for t in range(NTILE):
            rt.fill(f_s0[t].prod(), s0_bo,
                    tap=TensorAccessPattern(tensor_dims=(1, NTILE * S0_BLK),
                                            offset=t * S0_BLK,
                                            sizes=[NVH, S0_ELEM],
                                            strides=[S0_ELEM, 1]),
                    task_group=tg)
            rt.fill(f_par[t].prod(), par_bo,
                    tap=TensorAccessPattern(tensor_dims=(1, NTILE * PAR_BLK),
                                            offset=t * PAR_BLK,
                                            sizes=[NVH, PAR_ELEM],
                                            strides=[PAR_ELEM, 1]),
                    task_group=tg)
            rt.drain(f_sn[t].cons(), snew_bo,
                     tap=TensorAccessPattern(tensor_dims=(NTILE * S0_BLK,),
                                             offset=t * S0_BLK,
                                             sizes=[NVH, S0_ELEM],
                                             strides=[S0_ELEM, 1]),
                     task_group=tg)
            rt.drain(f_yd[t].cons(), yd_bo,
                     tap=TensorAccessPattern(tensor_dims=(NTILE * YD_BLK,),
                                             offset=t * YD_BLK,
                                             sizes=[NVH, YD_ELEM],
                                             strides=[YD_ELEM, 1]),
                     task_group=tg, wait=(t == NTILE - 1))
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    xclbin, insts = gdn_8kslab_op.compile()
    out_xclbin = str(PROJ_ROOT / f"fst_gdn_8kslab{SUFFIX}.xclbin")
    out_insts  = str(PROJ_ROOT / f"fst_gdn_8kslab{SUFFIX}_insts.bin")
    shutil.copy(xclbin, out_xclbin); shutil.copy(insts, out_insts)
    print(f"fst_gdn_8kslab{SUFFIX}: {out_xclbin} ({os.path.getsize(out_xclbin)}B)")
    print(f"  insts: {out_insts}")
    print(f"OK: {NTILE}-tile column-split chunkwise M=K GDN (single-call 8KB stack-S, K=8, 48 v-heads, 2+2 shim)")