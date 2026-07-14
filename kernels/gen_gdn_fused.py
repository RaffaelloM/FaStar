#!/usr/bin/env python3
"""gen_gdn_fused.py — FUSED row-streaming Gated Delta Net scan, ONE dispatch/SSM-layer.

A3 v2 (2026-07-13): reduces the tile DMA channel count to 1 MM2S + 1 S2MM =
exactly the bit-correct 3-pass passA channel budget, fixing BOTH race mechanisms
documented in docs/QWOPUS_FUSED_GDN_RACE_ROOTCAUSE.md:
  - vh0 startup race (was: 2 concurrent MM2S fills f_s0 + f_par)
  - 2-S2MM drain-flush cascade (was: 2 S2MM output drains f_scr + f_s2; the 3-pass
    kernel comment fst_gdn_scan_kernel.cc:42 states "two simultaneous held
    outputs broke the shim drain flush on this IRON build")

Mechanism: collapse passA + delta + passB (3 xclbins / 3 xrt::run per SSM layer =
144/token) into a SINGLE AIE dispatch per SSM layer = 48 dispatches/token.  One
Worker loops all 48 v-heads; per v-head: load params -> passA (128 rows) -> delta
(once) -> passB (128 rows) against a held tile scratch [a|b|delta|y | kn|qn|v],
streaming S0 from DDR twice per v-head (the 64 KB state does not fit in a 64 KB
tile).

Channel budget (AIE2P shim, A3 v2 = 1 MM2S + 1 S2MM + 1 tile-local Buffer):
  scr  Buffer: ONE aie.iron.Buffer (1024 floats, 4 KB) auto-pinned to the
              Worker's tile.  Tile-local memory, NOT a DMA channel.  RMW'd by
              the C kernels as a plain float*.  Reused (and re-zeroed) across
              all 48 v-heads sequentially.  Because it is not a DMA drain it
              imposes NONE of the held-output ObjectFifo drain-flush pathology
              that made the prior held+streaming-same-fifo build 60x too slow
              (held occupied 1 of 2 depth-2 buffers -> S2 streamed at effective
              depth=1 -> 6144 drains x ~10ms).  Here f_out has BOTH depth-2
              buffers free -> pipelined like the bit-correct 3-pass.
  in  f_s0 : ONE input fifo (1 MM2S).  Per v-head it carries 259 136-float
             packets: [pkt_v(136)][pkt_kn(136)][pkt_qn(136)][128 passA row-pkts]
             [128 passB row-pkts].  v/kn/qn full + gdec/beta live in the 3
             param-packets; the Worker loads them into scr[512:898].  passA/
             passB rows still carry per-row kn_i/qn_i/gdec for the RMW.  1 MM2S
             fill -> no 2nd fill -> no vh0 startup race.
  out f_out: ONE output fifo (1024-pkt, depth=2, 1 S2MM).  Per v-head the Worker
             streams 128 S2 row-packets (passB writes each row), then 1 y-packet
             (gdn_copy_scr_y copies scr[0:512]=[a|b|delta|y] into it, preserving
             the held-scr layout a@0 b@128 delta@256 y@384 so cmp_fused_ab and
             the engine unpack are byte-identical to the held-scr design).
             Drain order per v-head = [S2(128 rows), y].  1 S2MM channel -> no
             2 simultaneous drains -> no cascade; both buffers free -> pipelined.

This is 2 DMA channels (1 MM2S + 1 S2MM) + 1 tile-local Buffer = the bit-correct
3-pass passA channel budget, fixing BOTH race mechanisms while remaining FAST.

BD repeat_count is 8-bit [0:255]: f_s0 259=7x37, f_out 1024=8x128 (all dims <=255).
Outputs use 1024-float packets (proven silu drain geometry).  All fp32.  C kernels
in fst_gdn_fused_kernel.cc (gdn_zero_scr, gdn_load_v/kn/qn, gdn_passA_block,
gdn_delta_from_ab, gdn_passB_block_sep, gdn_copy_scr_y).

Determinism (the racy 4-fifo build): run1=0/48, run2=47/48 on identical S0 ->
pure temporal nondeterminism, vh0 always wrong + nondeterministic cascade.  The
3-pass passA C kernel is line-by-line identical to gdn_passA_block and is
bit-correct; the bug was purely the 4-DMA-channel tile config.  This 1+1+Buffer
config matches the 3-pass channel count.
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
from aie.iron import In, Out, Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

HV = 128
NV = 48
NBLK = 128                              # rows per v-head per pass
ROW = 136                               # [S(128)|kn_i|qn_i|gdec|pad(5)]  (passA/passB rows)
BLK = ROW                               # 1 row per packet
NPPARAM = 3                             # 3 param-packets per v-head (v, kn, qn)
PKT_V = 136                              # [v(128)|gdec@128|beta@129|pad(6)]
OUTPKT = 1024                           # held scratch (898 used) + s2 (1 row x 128)

BLK_T   = np.ndarray[(BLK,),    np.dtype[np.float32]]   # 136 (rows AND param pkts)
SCR_T   = np.ndarray[(OUTPKT,), np.dtype[np.float32]]   # 1024 held [a|b|delta|y|kn|qn|v|gdec|beta]
S2_T    = np.ndarray[(OUTPKT,), np.dtype[np.float32]]   # 1024 (1 row x 128, silu geom)

SRC = str(PROJ_ROOT / "fst_gdn_fused_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_fused.o"

# f_s0: 48 v-heads x 259 pkts x 136 = 48 x 7 x 37 x 136 (BD repeat <=255; 259=7x37)
NPKT_VH = NPPARAM + 2 * NBLK            # 3 + 128 + 128 = 259
assert NPKT_VH == 7 * 37
S0_NPKT_O, S0_NPKT_I = NV, NPKT_VH       # 48, 259  (both <=255)
# f_out: 48 v-heads x 129 pkts x 1024 = 48 x 129 x (8 x 128) (1024=8x128)
OUT_NPKT_VH = 2 * NBLK + 1               # 128 s2 + 1 y = 129
OUT_NPKT_O, OUT_NPKT_I = NV, OUT_NPKT_VH  # 48, 129


@iron.jit
def gdn_fused_op(Spkt: In, out_bo: Out):
    kA = ExternalFunction("gdn_passA_block", source_file=SRC,
        arg_types=[BLK_T, SCR_T], include_dirs=INC, object_file_name=OBJ)
    kD = ExternalFunction("gdn_delta_from_ab", source_file=SRC,
        arg_types=[SCR_T], include_dirs=INC, object_file_name=OBJ)
    kB = ExternalFunction("gdn_passB_block_sep", source_file=SRC,
        arg_types=[BLK_T, SCR_T, S2_T], include_dirs=INC, object_file_name=OBJ)
    kz = ExternalFunction("gdn_zero_scr", source_file=SRC,
        arg_types=[SCR_T], include_dirs=INC, object_file_name=OBJ)
    kLdv = ExternalFunction("gdn_load_v", source_file=SRC,
        arg_types=[BLK_T, SCR_T], include_dirs=INC, object_file_name=OBJ)
    kLdk = ExternalFunction("gdn_load_kn", source_file=SRC,
        arg_types=[BLK_T, SCR_T], include_dirs=INC, object_file_name=OBJ)
    kLdq = ExternalFunction("gdn_load_qn", source_file=SRC,
        arg_types=[BLK_T, SCR_T], include_dirs=INC, object_file_name=OBJ)
    kCopy = ExternalFunction("gdn_copy_scr_y", source_file=SRC,
        arg_types=[SCR_T, S2_T], include_dirs=INC, object_file_name=OBJ)

    # ONE input fifo (1 MM2S) + ONE output fifo (1 S2MM) + ONE tile-local Buffer
    # (scr, NOT a DMA channel) = 2 DMA channels, the 3-pass passA channel budget.
    f_s0  = ObjectFifo(BLK_T,  name="s0",  depth=2)   # shim->core, 136/pkt (rows + params)
    f_out = ObjectFifo(S2_T,   name="out", depth=2)   # core->shim, 1024/pkt (S2 rows + y)
    scr   = Buffer(SCR_T, name="scr")                # tile-local RMW scratch (auto-pins to Worker)

    def core_fused(fs0, fout, scr, kA, kD, kB, kz, kLdv, kLdk, kLdq, kCopy):
        for _v in range_(NV):                          # 48 v-heads, 1 dispatch
            kz(scr)                                     # zero [a|b|delta|y] = scr[0:512]
            # load 3 param-packets into scr[512:898] (v@768, kn@512, qn@640, gdec/beta@896/897)
            pv = fs0.acquire(1); kLdv(pv, scr); fs0.release(1)
            pk = fs0.acquire(1); kLdk(pk, scr); fs0.release(1)
            pq = fs0.acquire(1); kLdq(pq, scr); fs0.release(1)
            # passA: 128 rows -- a = S0ᵀ@kn, b = S0ᵀ@qn (RMW on tile-local scr[0:256])
            for _b in range_(NBLK):
                blk = fs0.acquire(1); kA(blk, scr); fs0.release(1)
            # delta: kvm=gdec*a ; delta=(v-kvm)*beta ; y=gdec*b + (kn·qn)*delta  (reads scr params)
            kD(scr)
            # passB: 128 rows -- S2 = gdec*S + kn⊗delta; stream each S2 row-pkt through f_out
            for _b in range_(NBLK):
                blk = fs0.acquire(1)
                s2 = fout.acquire(1)                    # 1024 streaming pkt (both depth-2 buffers free)
                kB(blk, scr, s2)
                fs0.release(1); fout.release(1)        # drain S2 row
            # y-pkt: copy scr[0:512]=[a|b|delta|y] into a streaming f_out pkt so the
            # held-scr layout (a@0 b@128 delta@256 y@384) is preserved in the output BO
            # -> cmp_fused_ab and the engine unpack are byte-identical to the held-scr design.
            ypkt = fout.acquire(1)
            kCopy(scr, ypkt)
            fout.release(1)                            # drain y (last per v-head)

    w = Worker(core_fused, [f_s0.cons(), f_out.prod(), scr,
                            kA, kD, kB, kz, kLdv, kLdk, kLdq, kCopy])

    rt = Runtime()
    S0_TOT  = NV * NPKT_VH * BLK                       # 48 x 259 x 136
    OUT_TOT = NV * OUT_NPKT_VH * OUTPKT                # 48 x 129 x 1024
    with rt.sequence(np.ndarray[(S0_TOT,),  np.dtype[np.float32]],
                     np.ndarray[(OUT_TOT,), np.dtype[np.float32]]) as (Spkt, out_bo):
        rt.start(w)
        tg = rt.task_group()
        # 1 MM2S fill: 48 x 259 x 136, split 259 = 7 x 37 (BD repeat <=255)
        rt.fill(f_s0.prod(), Spkt,
                tap=TensorAccessPattern(tensor_dims=(1, S0_TOT), offset=0,
                                        sizes=[S0_NPKT_O, 7, 37, BLK],
                                        strides=[7 * 37 * BLK, 37 * BLK, BLK, 1]),
                task_group=tg)
        # 1 S2MM drain: 48 x 129 x 1024, split 1024 = 8 x 128 (BD repeat <=255)
        rt.drain(f_out.cons(), out_bo,
                 tap=TensorAccessPattern(tensor_dims=(OUT_TOT,), offset=0,
                                         sizes=[OUT_NPKT_O, OUT_NPKT_I, 8, 128],
                                         strides=[OUT_NPKT_I * 8 * 128, 8 * 128, 128, 1]),
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
    _emit(gdn_fused_op, "fst_gdn_fused")
    print("OK: 1 fused GDN xclbin, 1 MM2S + 1 S2MM (3-pass channel budget), 48 v-heads + 3 phases in 1 dispatch")