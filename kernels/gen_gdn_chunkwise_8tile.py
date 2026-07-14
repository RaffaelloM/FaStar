#!/usr/bin/env python3
"""gen_gdn_chunkwise_8tile.py — chunkwise M=K GDN scan for XDNA2 (NPU2),
8-TILE COLUMN-SPLIT design (Stage 1.2-v2).

Collapses the shipped 3-pass-per-token GDN (3 dispatches/SSM-layer × 48 layers =
144 dispatches/token) into ONE dispatch/SSM-layer (K=8 tokens, 48 v-heads).

DESIGN — 8 tiles + column-split (fix for the 1-tile persistent-Buffer dead-end
where the 32KB 4-bank Buffer did not pipeline — 1.36µs/load_v ≈ 340× tile-local):
  - S[128,128] bf16 = 32KB split by COLUMN across 8 tiles → each tile 16 cols ×
    128 rows = 4KB (single bank — hoped to pipeline).
  - COLUMN-SPLIT → recurrence INDEPENDENT per tile (a[c]/b[c] for the tile's 16
    c's need only the tile's 16 cols + full kn,qn; c=kn·qn redundant; delta/y/
    passB elementwise) → NO per-token cross-tile reduction, NO handshake deadlock.
  - 8 tiles = 8× compute parallelism → ~5.6ms/tile, hidden under ~30ms syncobj.

TOPOLOGY — 2 CHAINS of 4 tiles (each chain = 4-tile cascade, 1 shim MM2S + 1
shim S2MM per chain → 2+2 total):
  Chain A (cols  0..63): shim MM2S1 → T0 → T1 → T2 → T3 → shim S2MM1
  Chain B (cols 64..127): shim MM2S2 → T4 → T5 → T6 → T7 → shim S2MM2
  Within-chain CO = 0,16,32,48.  All fifos flow FORWARD → no deadlock.

STREAM BUDGET — an AIE2P tile has ~2 in + 2 out core↔core streams.  The 4 logical
streams (S0, params, y, Sf) are COMBINED into TWO per tile pair:
  f_in  (shim→T0→..→T3): S0 rows (128/v-head) THEN param packets (8/v-head), all
          PKT=328-float (S0 row in the first 64, rest pad; params use full 328).
  f_out (T0→..→T3→shim): y rows (8/v-head) THEN Sf rows (128/v-head), all PKT=328
          (y/Sf in the first 64, rest pad).
  Middle tile = 2 in + 2 out (fits the budget).  The 4-separate-stream design
  (4 in + 4 out) failed placement ("compute-peer DMA budget unsatisfiable").

PKT=328 = [kn(128)|qn(128)|v(64)|gdec(1)|beta(1)|pad(6)] for a param packet; an
S0/y/Sf row occupies the first 64 (CHAIN cols), rest pad.  The host probe
interleaves 128 S0 + 8 par packets per v-head into the f_in BO (and reads 8 y +
128 Sf per v-head from the f_out BO, first 64 of each).

PER TILE: persistent S Buffer (bfloat16[128*16]=4KB) + ctr Buffer.  IRON worker
loops 48 v-heads; per v-head: ctr=0 → 128 load (acq f_in, fwd) → 8 recur (acq
f_in par + f_out y, fwd) → ctr=0 → 128 store (acq f_out sf, fwd).

Emits fst_gdn_chunkwise_8tile.xclbin + fst_gdn_chunkwise_8tile_insts.bin.
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
from aie.iron import ObjectFifo, Program, Runtime, Worker, Buffer
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
COLS   = 16
CHAIN  = 64            # columns per chain (4 tiles × 16)
PKT    = 328           # unified packet size (params = full; S0/y/Sf = first 64)
NTILES = 8
PER_VH = HV + K        # packets per v-head per stream = 128 S0 + 8 par  (== 8 y + 128 Sf)

SRC = str(PROJ_ROOT / "fst_gdn_chunkwise_8tile_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_chunkwise_8tile.o"
# Diagnostic toggles (env): GDN_8T_RECUR_NOOP (recur no-op), GDN_8T_LOADONLY
# (stream S, no MAC/RMW), GDN_8T_FWD_NOOP (skip 328-float forwarding).  When any
# is set the output is GARBAGE — latency diagnostic only, not bit-correct.
_CFLAGS = []
for _v in ("GDN_8T_RECUR_NOOP", "GDN_8T_LOADONLY", "GDN_8T_FWD_NOOP"):
    if os.environ.get(_v):
        _CFLAGS.append(f"-D{_v}")
CFLAGS = _CFLAGS or None

S_T    = np.ndarray[(HV * COLS,), np.dtype[bfloat16]]   # 4KB persistent S
CTR_T  = np.ndarray[(2,),         np.dtype[np.float32]]
PKT_T  = np.ndarray[(PKT,),       np.dtype[np.float32]] # unified 328-float packet
CO_T   = np.int32

# ExternalFunctions MUST be constructed inside the @jit body: IRON clears the
# _instances registry at the start of each @jit call, so module-level externs
# are dropped and never compiled (link then fails with "cannot open .o").
def _ext(name, types):
    return ExternalFunction(name, source_file=SRC, arg_types=types,
                            include_dirs=INC, object_file_name=OBJ, compile_flags=CFLAGS)


@iron.jit
def gdn_8tile_op(inp: iron.In, out: iron.Out):
    k_ctr   = _ext("gdn_ctr_zero",      [CTR_T])
    k_lfwd  = _ext("gdn_load_s0_fwd",   [S_T, CTR_T, PKT_T, PKT_T, CO_T])
    k_lend  = _ext("gdn_load_s0_end",   [S_T, CTR_T, PKT_T, CO_T])
    k_rhead = _ext("gdn_recur_head",    [S_T, PKT_T, PKT_T, PKT_T, CO_T])
    k_rmid  = _ext("gdn_recur_mid",     [S_T, PKT_T, PKT_T, PKT_T, PKT_T, CO_T])
    k_rtail = _ext("gdn_recur_tail",    [S_T, PKT_T, PKT_T, PKT_T, CO_T])
    k_shead = _ext("gdn_store_sf_head", [S_T, CTR_T, PKT_T, CO_T])
    k_smid  = _ext("gdn_store_sf_mid",  [S_T, CTR_T, PKT_T, PKT_T, CO_T])

    S   = [Buffer(S_T,   name=f"S{t}")   for t in range(NTILES)]
    ctr = [Buffer(CTR_T, name=f"ctr{t}") for t in range(NTILES)]

    def of(name, depth=2):
        return ObjectFifo(PKT_T, name=name, depth=depth)
    # chain A f_in (shim→T0→T1→T2→T3) and f_out (T0→T1→T2→T3→shim).
    f_inA   = [of("inA0"),  of("inA1"),  of("inA2"),  of("inA3")]   # inA0=shim→T0
    f_outA  = [of("outA0"), of("outA1"), of("outA2"), of("outA3")]  # outA3=T3→shim
    f_inB   = [of("inB0"),  of("inB1"),  of("inB2"),  of("inB3")]
    f_outB  = [of("outB0"), of("outB1"), of("outB2"), of("outB3")]

    # core fns (head/mid/tail).  Bound ExternalFunctions are trailing args.
    def core_head(ini, ino, outo, S, ctr, kctr, klfwd, krhead, kshead):
        for _v in range_(NVH):
            kctr(ctr)
            for _i in range_(HV):
                ri = ini.acquire(1); ro = ino.acquire(1)
                klfwd(S, ctr, ri, ro, 0)
                ini.release(1); ino.release(1)
            for _t in range_(K):
                pi = ini.acquire(1); po = ino.acquire(1); ye = outo.acquire(1)
                krhead(S, pi, po, ye, 0)
                ini.release(1); ino.release(1); outo.release(1)
            kctr(ctr)
            for _i in range_(HV):
                ye = outo.acquire(1)
                kshead(S, ctr, ye, 0)
                outo.release(1)

    def core_mid(ini, ino, outi, outo, S, ctr, CO, kctr, klfwd, krmid, ksmid):
        for _v in range_(NVH):
            kctr(ctr)
            for _i in range_(HV):
                ri = ini.acquire(1); ro = ino.acquire(1)
                klfwd(S, ctr, ri, ro, CO)
                ini.release(1); ino.release(1)
            for _t in range_(K):
                pi = ini.acquire(1); po = ino.acquire(1); yii = outi.acquire(1); ye = outo.acquire(1)
                krmid(S, pi, po, yii, ye, CO)
                ini.release(1); ino.release(1); outi.release(1); outo.release(1)
            kctr(ctr)
            for _i in range_(HV):
                yii = outi.acquire(1); ye = outo.acquire(1)
                ksmid(S, ctr, yii, ye, CO)
                outi.release(1); outo.release(1)

    def core_tail(ini, outi, outo, S, ctr, CO, kctr, klend, krtail, ksmid):
        for _v in range_(NVH):
            kctr(ctr)
            for _i in range_(HV):
                ri = ini.acquire(1)
                klend(S, ctr, ri, CO)
                ini.release(1)
            for _t in range_(K):
                pi = ini.acquire(1); yii = outi.acquire(1); ye = outo.acquire(1)
                krtail(S, pi, yii, ye, CO)
                ini.release(1); outi.release(1); outo.release(1)
            kctr(ctr)
            for _i in range_(HV):
                yii = outi.acquire(1); ye = outo.acquire(1)
                ksmid(S, ctr, yii, ye, CO)
                outi.release(1); outo.release(1)

    # chain A: T0..T3 (CO 0,16,32,48).  f_inA[0]=shim→T0, f_inA[i]=T(i-1)→Ti.
    w0 = Worker(core_head,
                [f_inA[0].cons(),  f_inA[1].prod(),  f_outA[0].prod(),
                 S[0], ctr[0], k_ctr, k_lfwd, k_rhead, k_shead], stack_size=0x2000)
    w1 = Worker(core_mid,
                [f_inA[1].cons(),  f_inA[2].prod(),  f_outA[0].cons(), f_outA[1].prod(),
                 S[1], ctr[1], 16, k_ctr, k_lfwd, k_rmid, k_smid], stack_size=0x2000)
    w2 = Worker(core_mid,
                [f_inA[2].cons(),  f_inA[3].prod(),  f_outA[1].cons(), f_outA[2].prod(),
                 S[2], ctr[2], 32, k_ctr, k_lfwd, k_rmid, k_smid], stack_size=0x2000)
    w3 = Worker(core_tail,
                [f_inA[3].cons(),  f_outA[2].cons(), f_outA[3].prod(),
                 S[3], ctr[3], 48, k_ctr, k_lend, k_rtail, k_smid], stack_size=0x2000)
    # chain B: T4..T7 (CO 0,16,32,48 within chain B; v sliced by the host).
    w4 = Worker(core_head,
                [f_inB[0].cons(),  f_inB[1].prod(),  f_outB[0].prod(),
                 S[4], ctr[4], k_ctr, k_lfwd, k_rhead, k_shead], stack_size=0x2000)
    w5 = Worker(core_mid,
                [f_inB[1].cons(),  f_inB[2].prod(),  f_outB[0].cons(), f_outB[1].prod(),
                 S[5], ctr[5], 16, k_ctr, k_lfwd, k_rmid, k_smid], stack_size=0x2000)
    w6 = Worker(core_mid,
                [f_inB[2].cons(),  f_inB[3].prod(),  f_outB[1].cons(), f_outB[2].prod(),
                 S[6], ctr[6], 32, k_ctr, k_lfwd, k_rmid, k_smid], stack_size=0x2000)
    w7 = Worker(core_tail,
                [f_inB[3].cons(),  f_outB[2].cons(), f_outB[3].prod(),
                 S[7], ctr[7], 48, k_ctr, k_lend, k_rtail, k_smid], stack_size=0x2000)

    rt = Runtime()
    N_CHAIN = NVH * PER_VH        # packets per chain (48 × 136)
    OFF_B   = N_CHAIN * PKT       # chain B offset within each BO
    with rt.sequence(np.ndarray[(2 * N_CHAIN * PKT,), np.dtype[np.float32]],   # inp [A|B]
                     np.ndarray[(2 * N_CHAIN * PKT,), np.dtype[np.float32]]) as (inp, out):
        rt.start(w0, w1, w2, w3, w4, w5, w6, w7)
        tg = rt.task_group()
        rt.fill(f_inA[0].prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(1, 2 * N_CHAIN * PKT), offset=0,
                                        sizes=[NVH, PER_VH, PKT], strides=[PER_VH * PKT, PKT, 1]),
                task_group=tg)
        rt.fill(f_inB[0].prod(), inp,
                tap=TensorAccessPattern(tensor_dims=(1, 2 * N_CHAIN * PKT), offset=OFF_B,
                                        sizes=[NVH, PER_VH, PKT], strides=[PER_VH * PKT, PKT, 1]),
                task_group=tg)
        rt.drain(f_outA[3].cons(), out,
                 tap=TensorAccessPattern(tensor_dims=(2 * N_CHAIN * PKT,), offset=0,
                                         sizes=[NVH, PER_VH, PKT], strides=[PER_VH * PKT, PKT, 1]),
                 task_group=tg)
        rt.drain(f_outB[3].cons(), out,
                 tap=TensorAccessPattern(tensor_dims=(2 * N_CHAIN * PKT,), offset=OFF_B,
                                         sizes=[NVH, PER_VH, PKT], strides=[PER_VH * PKT, PKT, 1]),
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
    _emit(gdn_8tile_op, "fst_gdn_chunkwise_8tile")
    print("OK: 8-tile column-split chunkwise M=K GDN (2 chains × 4 tiles, K=8, 48 v-heads)")