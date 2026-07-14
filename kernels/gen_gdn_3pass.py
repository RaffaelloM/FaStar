#!/usr/bin/env python3
"""gen_gdn_3pass.py — ONE-XCLBIN 3-PASS-SEQUENTIAL GDN scan for XDNA2 (NPU2).

THE "no-state / dispatch-amortization without on-tile RMU" path.  Collapses the
shipped 3-xclbin 3-dispatch/SSM-layer GDN (passA/delta/passB each its own
xrt::run, 144 dispatches/token) to ONE xclbin / ONE xrt::run/SSM-layer (48
dispatches/token) ≈ ~3x over shipped 0.21 tok/s.

Sidesteps BOTH proven dead-ends:
  - fused (fst_gdn_fused): holds S on-chip across passes → Buffer RMW 600x slow;
  - chunkwise M=K on-tile-state: holds S on-tile across K tokens → ~340x slow at
    >=16KB in every storage class.
HERE S is NEVER held on-tile: streamed ONE 8-row block at a time from DDR (read
twice — passA then passB — as the shipped 3-pass already does), discarded.  Only
the SMALL on-tile state persists (self-loop ObjectFifos, the proven "held
accumulator" pattern w/o the shim drain): ab[a|b]=1KB, delta=512B ≈ 1.5KB — well
under the ~8KB fast threshold.  No aie.iron.Buffer; no >=16KB on-tile array.

MATH byte-identical to the shipped 3-pass (fst_gdn_scan_kernel.cc); the C fns in
fst_gdn_3pass_kernel.cc reuse that math verbatim (a/b and delta sourced from the
held fifos instead of packed in the DMA packet).

FIFOS (one worker; 2 MM2S + 1 S2MM shim = within the 2+2 budget; ab/delta are
self-loop CORE-LOCAL held ObjectFifos, not shim streams):
  f_in_S  (MM2S, 1040/pkt = 8 rows × [S_row|kn_i|qn_i], 32/v-head): 16 passA + 16 passB blocks
  f_in_par(MM2S, 386/pkt = [v|kn|qn|gdec|beta], 1/v-head)
  f_out   (S2MM, 1024/pkt, 17/v-head): 1 y packet [y@0:128] + 16 S2 packets (8 rows each)
  f_ab    (self-loop, 256 = [a|b], held): acquired once/v-head, RMW'd across passA, read by delta
  f_delta (self-loop, 128, held): written by delta, read by passB

Emits fst_gdn_3pass.xclbin + fst_gdn_3pass_insts.bin.
"""
import os, sys, shutil
from pathlib import Path

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
IRON_PATH = str(PROJ_ROOT.parent / "Source" / "IRON-devel")
# fall back to the installed mlir_aie if the Source tree is absent
if not Path(IRON_PATH).exists() and IRON_PATH not in sys.path:
    pass
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

HV      = 128
NV      = 48
ROWS    = 128
RPB     = 8                       # rows per block
NBLK    = ROWS // RPB             # 16 blocks/v-head per pass
SIN     = HV + 2                  # 130 = [S_row(128)|kn_i|qn_i]
SBLK    = RPB * SIN               # 1040 = 8-row block (passA AND passB S input)
PAR     = 3 * HV + 2              # 386 = [v(128)|kn(128)|qn(128)|gdec|beta]
AB      = 2 * HV                  # 256 = [a|b]
DELTA   = HV                      # 128
OUTPKT  = 1024                    # proven silu drain geometry
NP_S    = NV * (2 * NBLK)         # 48 × 32 S-blocks/v-head (16 passA + 16 passB)
NOUT    = NV * (1 + NBLK)         # 48 × 17 out-packets/v-head (1 y + 16 S2)

SRC = str(PROJ_ROOT / "fst_gdn_3pass_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_3pass.o"

SBLK_T  = np.ndarray[(SBLK,),   np.dtype[np.float32]]
PAR_T   = np.ndarray[(PAR,),    np.dtype[np.float32]]
AB_T    = np.ndarray[(AB,),     np.dtype[np.float32]]
DELTA_T = np.ndarray[(DELTA,),  np.dtype[np.float32]]
OUT_T   = np.ndarray[(OUTPKT,), np.dtype[np.float32]]


@iron.jit
def gdn_3pass_op(inp_S: In, inp_par: In, out: Out):
    kA = ExternalFunction("gdn_3p_passA_block", source_file=SRC,
        arg_types=[SBLK_T, AB_T], include_dirs=INC, object_file_name=OBJ)
    kD = ExternalFunction("gdn_3p_delta", source_file=SRC,
        arg_types=[AB_T, PAR_T, DELTA_T, OUT_T], include_dirs=INC, object_file_name=OBJ)
    kB = ExternalFunction("gdn_3p_passB_block", source_file=SRC,
        arg_types=[SBLK_T, DELTA_T, PAR_T, OUT_T], include_dirs=INC, object_file_name=OBJ)
    kZ = ExternalFunction("gdn_3p_zero", source_file=SRC,
        arg_types=[AB_T], include_dirs=INC, object_file_name=OBJ)

    # shim streams (2 MM2S + 1 S2MM)
    f_in_S   = ObjectFifo(SBLK_T,  name="inS",  depth=2)
    f_in_par = ObjectFifo(PAR_T,   name="inP",  depth=2)
    f_out    = ObjectFifo(OUT_T,   name="out",  depth=2)
    # self-loop CORE-LOCAL held state (prod == cons == this worker; no shim drain).
    # depth=1 ⇒ cons.acquire returns the SAME buffer the worker just produced, so
    # the C fns read the data they wrote.  The worker holds the prod-acquired
    # pointer across the produce+consume phases, then cycles the slot back via a
    # cons acquire/release to free it for the next v-head.
    f_ab     = ObjectFifo(AB_T,    name="ab",   depth=1)
    f_delta  = ObjectFifo(DELTA_T, name="dlt",  depth=1)

    def core(fS, fP, fO, fab_p, fab_c, fdlt_p, fdlt_c, kA, kD, kB, kZ):
        for _v in range_(NV):                       # 48 v-heads, 1 dispatch
            ab_p = fab_p.acquire(1);  kZ(ab_p)      # held [a|b], zeroed
            for _b in range_(NBLK):                 # passA: 16 × 8-row blocks → produce ab
                s = fS.acquire(1);  kA(s, ab_p);  fS.release(1)
            dlt_p = fdlt_p.acquire(1)               # held delta (produced by kD)
            par   = fP.acquire(1)                   # held across delta + passB (gdec)
            ypkt  = fO.acquire(1)                   # 1024, y @ [0:128]
            kD(ab_p, par, dlt_p, ypkt)              # reads ab_p, writes dlt_p + ypkt
            fO.release(1)                           # drain y
            fab_p.release(1);  fab_c.acquire(1);  fab_c.release(1)   # free ab slot
            for _b in range_(NBLK):                 # passB: 16 × 8-row blocks → consume dlt
                s  = fS.acquire(1)
                s2 = fO.acquire(1)                  # 1024, 8 S2 rows
                kB(s, dlt_p, par, s2)
                fS.release(1);  fO.release(1)
            fP.release(1)
            fdlt_p.release(1);  fdlt_c.acquire(1);  fdlt_c.release(1)   # free dlt slot

    w = Worker(core, [f_in_S.cons(), f_in_par.cons(), f_out.prod(),
                      f_ab.prod(), f_ab.cons(), f_delta.prod(), f_delta.cons(),
                      kA, kD, kB, kZ])

    rt = Runtime()
    with rt.sequence(np.ndarray[(NP_S * SBLK,), np.dtype[np.float32]],     # inp_S
                     np.ndarray[(NV * PAR,),   np.dtype[np.float32]],      # inp_par
                     np.ndarray[(NOUT * OUTPKT,), np.dtype[np.float32]]) as (inp_S, inp_par, out):
        rt.start(w)
        tg = rt.task_group()
        # f_in_S: 48 v-heads × 32 blocks × 1040 (16 passA + 16 passB, contiguous)
        rt.fill(f_in_S.prod(), inp_S,
                tap=TensorAccessPattern(tensor_dims=(1, NP_S * SBLK), offset=0,
                                        sizes=[NV, 2 * NBLK, SBLK],
                                        strides=[(2 * NBLK) * SBLK, SBLK, 1]),
                task_group=tg)
        # f_in_par: 48 × 386
        rt.fill(f_in_par.prod(), inp_par,
                tap=TensorAccessPattern(tensor_dims=(1, NV * PAR), offset=0,
                                        sizes=[NV, PAR], strides=[PAR, 1]),
                task_group=tg)
        # f_out drain: 48 × 17 × 1024 (1 y + 16 S2)
        rt.drain(f_out.cons(), out,
                 tap=TensorAccessPattern(tensor_dims=(NOUT * OUTPKT,), offset=0,
                                         sizes=[NV, 1 + NBLK, OUTPKT],
                                         strides=[(1 + NBLK) * OUTPKT, OUTPKT, 1]),
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
    _emit(gdn_3pass_op, "fst_gdn_3pass")
    print("OK: one-xclbin 3-pass-sequential GDN (DDR-streamed S, 1 xrt::run/layer, 48 v-heads)")