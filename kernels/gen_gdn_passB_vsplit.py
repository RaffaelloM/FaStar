#!/usr/bin/env python3
"""gen_gdn_passB_vsplit.py — V-HEAD-SPLIT passB for XDNA2 (NPU2).

The SHIPPED passB (gen_gdn_scan.py::gdn_passB_op) runs all 48 v-heads on ONE
tile / ONE worker / ONE shim pair (1 MM2S + 1 S2MM).  It is DMA-bound on the
single shim's S0-read + S2-write (6 MB bidirectional @ ~280 MB/s/dir ≈ 33.6 ms).
The earlier delta-broadcast fork (gen_gdn_passB_delta.py) was ALSO 1-tile/1-shim
— it eliminated the 3.1 MB replicated delta but the S0/S2 floor stayed, so it
gave 0% gain.  MULTI-TILE PARALLELISM of the S0/S2 stream itself was NEVER
tested.

This kernel keeps gdn_passB_block byte-identical (same math, same 264-stride
row packet [S0(128)|delta(128)|kn_i(1)|gdec(1)|pad(5)], same 8-rows/block
streaming so S0 never sits on-tile as a 64 KB array) and splits the 48 v-heads
across NTILE workers — each on its own shim columns → NTILE× the aggregate DMA
bandwidth.  v-heads are fully independent (each owns S[128,128]), so v-head-split
needs NO cross-tile reduction (unlike column-split).

Per tile: 1 MM2S (f_in, row-block packets) + 1 S2MM (f_out, S2 row-blocks).
NTILE=8 → 8 MM2S + 8 S2MM = 16 streams on 8 shim columns = 2 streams/col =
the 4-tile slab's PROVEN-feasible geometry.  NTILE=4 → 8 streams (lighter).
Each tile handles NVHPT = 48//NTILE v-heads (NTILE=8 ⇒ 6, NTILE=4 ⇒ 12).

Env: FST_GDN_NTILE (default 8).  Emits fst_gdn_passB_vsplit[_Nt].xclbin + insts.
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

HV      = 128
NROWS   = 128
NV      = 48
PKT_B   = 264                        # [S0(128)|delta(128)|kn_i(1)|gdec(1)|pad(5)]
ROWS_PER_PKT = 8
NPKT_V  = NROWS // ROWS_PER_PKT      # 16 blocks per v-head
PKT_B_BLK = PKT_B * ROWS_PER_PKT     # 2112 fp32 (8 rows)
OUT_BLK = HV * ROWS_PER_PKT          # 1024 fp32 (8 rows × 128)

NTILE   = int(os.environ.get("FST_GDN_NTILE", "8"))
assert NV % NTILE == 0, f"NV={NV} must be divisible by NTILE={NTILE}"
NVHPT   = NV // NTILE                # v-heads per tile
NPKT_TILE = NVHPT * NPKT_V           # packets per tile (≤192 for NTILE≥4)

SUFFIX  = os.environ.get("FST_GDN_SUFFIX", f"_{NTILE}t")

SRC = str(PROJ_ROOT / "fst_gdn_scan_kernel.cc")   # reuse gdn_passB_block verbatim
INC = [aie_config.cxx_header_path()]
OBJ = f"fst_gdn_passB_vsplit{SUFFIX}.o"           # unique obj (avoid @iron.jit cache clash)

BLK_IN_T  = np.ndarray[(PKT_B_BLK,), np.dtype[np.float32]]
BLK_OUT_T = np.ndarray[(OUT_BLK,),  np.dtype[np.float32]]


@iron.jit
def gdn_passB_vsplit_op(in_bo: iron.In, out_bo: iron.Out):
    kB = ExternalFunction("gdn_passB_block", source_file=SRC,
        arg_types=[BLK_IN_T, BLK_OUT_T], include_dirs=INC, object_file_name=OBJ)

    f_in  = [ObjectFifo(BLK_IN_T,  name=f"in{t}",  depth=2) for t in range(NTILE)]
    f_out = [ObjectFifo(BLK_OUT_T, name=f"out{t}", depth=2) for t in range(NTILE)]

    def core_passB(fin, fo, kB):
        for _ in range_(NPKT_TILE):              # this tile's NVHPT v-heads × 16 blocks
            p = fin.acquire(1); s2 = fo.acquire(1)
            kB(p, s2)
            fin.release(1); fo.release(1)

    workers = [Worker(core_passB, [f_in[t].cons(), f_out[t].prod(), kB])
               for t in range(NTILE)]

    # per-tile contiguous block in each BO: [tile0: NPKT_TILE×PKT_B_BLK][tile1: ...]
    IN_BLK_TILE  = NPKT_TILE * PKT_B_BLK
    OUT_BLK_TILE = NPKT_TILE * OUT_BLK
    IN_TOTAL  = NTILE * IN_BLK_TILE      # = NV * NROWS * PKT_B (same as shipped)
    OUT_TOTAL = NTILE * OUT_BLK_TILE     # = NV * NROWS * HV

    # single-level TAP (NPKT_TILE ≤ 192 < 255, no BD-repeat split needed)
    def in_tap(t):
        return TensorAccessPattern(tensor_dims=(1, IN_TOTAL),
                                   offset=t * IN_BLK_TILE,
                                   sizes=[NPKT_TILE, PKT_B_BLK],
                                   strides=[PKT_B_BLK, 1])
    def out_tap(t):
        return TensorAccessPattern(tensor_dims=(OUT_TOTAL,),
                                   offset=t * OUT_BLK_TILE,
                                   sizes=[NPKT_TILE, OUT_BLK],
                                   strides=[OUT_BLK, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(IN_TOTAL,),  np.dtype[np.float32]],
                     np.ndarray[(OUT_TOTAL,), np.dtype[np.float32]]) as (in_bo, out_bo):
        rt.start(*workers)
        tg = rt.task_group()
        for t in range(NTILE):
            rt.fill(f_in[t].prod(), in_bo, tap=in_tap(t), task_group=tg)
            rt.drain(f_out[t].cons(), out_bo, tap=out_tap(t),
                     task_group=tg, wait=(t == NTILE - 1))
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    xclbin, insts = gdn_passB_vsplit_op.compile()
    out_xclbin = str(PROJ_ROOT / f"fst_gdn_passB_vsplit{SUFFIX}.xclbin")
    out_insts  = str(PROJ_ROOT / f"fst_gdn_passB_vsplit{SUFFIX}_insts.bin")
    shutil.copy(xclbin, out_xclbin)
    shutil.copy(insts, out_insts)
    print(f"fst_gdn_passB_vsplit{SUFFIX}: {out_xclbin} ({os.path.getsize(out_xclbin)}B)")
    print(f"  insts: {out_insts}")
    print(f"OK: v-head-split passB — NTILE={NTILE}, {NVHPT} v-heads/tile, "
          f"{NPKT_TILE} packets/tile, 1 dispatch (byte-identical gdn_passB_block math)")