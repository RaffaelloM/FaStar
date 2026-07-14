#!/usr/bin/env python3
"""gen_gdn_chunkwise.py — chunkwise M=K GDN scan for XDNA2 (NPU2), ONE-TILE
persistent-Buffer design (Stage 1.2).

Collapses the shipped 3-pass-per-token GDN (3 dispatches/SSM-layer × 48 layers
= 144 dispatches/token) into ONE dispatch/SSM-layer that processes K=8 tokens
for all 48 v-heads, holding the recurrence state S[128,128] ON-TILE as a
persistent aie.iron.Buffer (32 KB bf16) so it never round-trips to DDR between
tokens.  This is the prerequisite for lossless K× speculative decoding (Stage 2).

WHY ONE TILE + persistent Buffer (the Stage-1.1 breakthrough, see
docs/QWOPUS_CHUNKWISE_GDN_MICROTEST_POSTMORTEM.md): a 32 KB bf16 S[128,128] fits
one tile as a register-addressed Buffer (escapes the stack immediate limit
[-32768,-64]), persists across C-fn re-calls (unlike a stack array), and is FAST
under C-kernel load_v/store_v (0.10 ms/call — NOT the 600× slow Python-level
Buffer RMW that killed M=1 fusion).  No 4-tile split, no cross-tile reduction.

STRUCTURE — 1 worker/1 tile, 4 row-sized depth-2 ObjectFifos + 2 persistent
Buffers:
  S   = Buffer(bfloat16[128*128])   — recurrence state (32 KB)
  ctr = Buffer(float32[2])          — row index for load/store (persistent counter)
  f_s0  (shim MM2S) : 128 fp32 × 48 v-heads       — S0 in (fp32 -> bf16 on-tile)
  f_par (shim MM2S) : K×48 packets [qn|kn|v|gdec|beta|pad] (392 fp32)
  f_y   (shim S2MM) : K×48 packets (128 fp32)      — per-token y out
  f_sf  (shim S2MM) : 128 fp32 × 48 v-heads       — S_final out (bf16 -> fp32)

  Worker core (48 v-heads):
    ctr=0; for i in 128: acq f_s0; gdn_load_s0_row(S,ctr,row); rel       # S0 -> S
    for t in 8:    acq f_par; acq f_y; gdn_chunkwise_recur(S,par,y); rel  # RMW S (1 token)
    ctr=0; for i in 128: acq f_sf; gdn_store_sf_row(S,ctr,row); rel       # S -> S_final

  The recurrence C fn has a SMALL frame (S is a passed Buffer pointer; only
  a/b/delta = 1.5 KB are stack) → the 8 KB-frame re-call breakage does NOT
  apply → 8 small-frame re-calls/v-head are clean.  S persists in the Buffer
  across the 8 per-token calls.  2 MM2S + 2 S2MM (within the ≤2+2 shim budget;
  the 4-MM2S overload that broke the 4-tile per-tile shim is avoided).

MATH is reused verbatim from the proven 3-pass fst_gdn_scan_kernel.cc; only S
storage precision changes (fp32 DDR -> bf16 on-tile, the A/B-validated lossless
mitigation FST_Q35_GDN_BF16S, 39/39 argmax-identical).  fp32 compute on
bf16-rounded S; the bf16 round happens only at the passB store.

Emits fst_gdn_chunkwise.xclbin + fst_gdn_chunkwise_insts.bin.
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

HV   = 128
K    = 8                # chunk size (tokens per dispatch)
NVH  = 48               # GDN v-heads per layer (all in 1 dispatch)
PKT_PAR = 392           # [qn(128)|kn(128)|v(128)|gdec(1)|beta(1)|pad(6)] — 8-float aligned

S_T    = np.ndarray[(HV * HV,), np.dtype[bfloat16]]      # 16384 bf16 = 32 KB persistent state
CTR_T  = np.ndarray[(2,),         np.dtype[np.float32]]   # row index counter
S0_T   = np.ndarray[(HV,),        np.dtype[np.float32]]   # one S0 row in (fp32)
PAR_T  = np.ndarray[(PKT_PAR,),   np.dtype[np.float32]]   # one token's params
Y_T    = np.ndarray[(HV,),        np.dtype[np.float32]]   # one token's y out
SF_T   = np.ndarray[(HV,),        np.dtype[np.float32]]   # one S_final row out (fp32)

SRC = str(PROJ_ROOT / "fst_gdn_chunkwise_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_chunkwise.o"
# Diagnostic toggles (env vars): GDN_RECUR_NOOP makes the recurrence a no-op
# (isolates load/store latency); GDN_LOADSTORE_NOOP makes load/store no-ops
# (isolates recurrence latency).  When set, appends -D to the kernel compile.
_CFLAGS = []
if os.environ.get("GDN_RECUR_NOOP"):      _CFLAGS.append("-DGDN_RECUR_NOOP")
if os.environ.get("GDN_RECUR_LOADONLY"):  _CFLAGS.append("-DGDN_RECUR_LOADONLY")
if os.environ.get("GDN_LOADSTORE_NOOP"):  _CFLAGS.append("-DGDN_LOADSTORE_NOOP")
CFLAGS = _CFLAGS or None


@iron.jit
def gdn_chunkwise_op(inp_s0: iron.In, inp_par: iron.In,
                     out_y: iron.Out, out_sf: iron.Out):
    kctr   = ExternalFunction("gdn_ctr_zero",        source_file=SRC,
        arg_types=[CTR_T],                          include_dirs=INC, object_file_name=OBJ, compile_flags=CFLAGS)
    kload  = ExternalFunction("gdn_load_s0_row",     source_file=SRC,
        arg_types=[S_T, CTR_T, S0_T],               include_dirs=INC, object_file_name=OBJ, compile_flags=CFLAGS)
    krecur = ExternalFunction("gdn_chunkwise_recur", source_file=SRC,
        arg_types=[S_T, PAR_T, Y_T],                include_dirs=INC, object_file_name=OBJ, compile_flags=CFLAGS)
    kstore = ExternalFunction("gdn_store_sf_row",    source_file=SRC,
        arg_types=[S_T, CTR_T, SF_T],               include_dirs=INC, object_file_name=OBJ, compile_flags=CFLAGS)

    S   = Buffer(S_T,   name="ScratchS")
    ctr = Buffer(CTR_T, name="RowCtr")

    f_s0  = ObjectFifo(S0_T,  name="s0",  depth=2)   # shim MM2S -> core
    f_par = ObjectFifo(PAR_T, name="par", depth=2)   # shim MM2S -> core
    f_y   = ObjectFifo(Y_T,   name="y",   depth=2)   # core -> shim S2MM
    f_sf  = ObjectFifo(SF_T,  name="sf",  depth=2)   # core -> shim S2MM

    def core(S, ctr, fs0, fpar, fy, fsf, kctr, kload, krecur, kstore):
        for _v in range_(NVH):                          # 48 v-heads, 1 dispatch
            kctr(ctr)                                   # ctr = 0
            for _i in range_(HV):                       # 128 rows: S0 (fp32) -> S (bf16)
                r = fs0.acquire(1)
                kload(S, ctr, r)
                fs0.release(1)
            for _t in range_(K):                        # 8 tokens: recurrence (RMW S)
                p = fpar.acquire(1)
                y = fy.acquire(1)
                krecur(S, p, y)
                fpar.release(1); fy.release(1)
            kctr(ctr)                                   # ctr = 0
            for _i in range_(HV):                       # 128 rows: S (bf16) -> S_final (fp32)
                o = fsf.acquire(1)
                kstore(S, ctr, o)
                fsf.release(1)

    w = Worker(core, [S, ctr, f_s0.cons(), f_par.cons(), f_y.prod(), f_sf.prod(),
                      kctr, kload, krecur, kstore], stack_size=0x1000)

    rt = Runtime()
    N_S0 = NVH * HV              # 6144 S0 row packets
    N_PAR = NVH * K              # 384 param packets
    N_Y   = NVH * K              # 384 y packets
    N_SF  = NVH * HV             # 6144 S_final row packets
    # BD-repeat ≤ 255: split the 6144-packet streams as 48 (BDs) × 128 (repeat).
    with rt.sequence(np.ndarray[(N_S0  * HV,),       np.dtype[np.float32]],   # inp_s0
                     np.ndarray[(N_PAR * PKT_PAR,),  np.dtype[np.float32]],   # inp_par
                     np.ndarray[(N_Y   * HV,),       np.dtype[np.float32]],   # out_y
                     np.ndarray[(N_SF  * HV,),       np.dtype[np.float32]]) as (inp_s0, inp_par, out_y, out_sf):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_s0.prod(), inp_s0,
                tap=TensorAccessPattern(tensor_dims=(1, N_S0 * HV), offset=0,
                                        sizes=[NVH, HV, HV], strides=[HV * HV, HV, 1]),
                task_group=tg)
        rt.fill(f_par.prod(), inp_par,
                tap=TensorAccessPattern(tensor_dims=(1, N_PAR * PKT_PAR), offset=0,
                                        sizes=[NVH, K, PKT_PAR], strides=[K * PKT_PAR, PKT_PAR, 1]),
                task_group=tg)
        rt.drain(f_y.cons(), out_y,
                 tap=TensorAccessPattern(tensor_dims=(N_Y * HV,), offset=0,
                                         sizes=[NVH, K, HV], strides=[K * HV, HV, 1]),
                 task_group=tg)
        rt.drain(f_sf.cons(), out_sf,
                 tap=TensorAccessPattern(tensor_dims=(N_SF * HV,), offset=0,
                                         sizes=[NVH, HV, HV], strides=[HV * HV, HV, 1]),
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
    _emit(gdn_chunkwise_op, "fst_gdn_chunkwise")
    print("OK: 1-tile chunkwise M=K GDN (persistent Buffer S, K=8, 48 v-heads, 1 dispatch/SSM-layer)")