#!/usr/bin/env python3
"""gen_gdn_scan.py — row-streaming Gated Delta Net scan for XDNA2 (NPU2).

ALL 48 v-heads are processed in ONE dispatch per pass (3 passes total per SSM
layer = 3 xrt::run/layer, NOT 144).  Each pass's single Worker loops all 48
v-heads internally; the state for every v-head lives in one DDR BO and is
streamed row-by-row through small depth-2 objectfifos.

The GDN state S[128,128] = 64 KB fp32 is too big for a 64 KB AIE2P tile as a
single ObjectFifo packet, so the STATE LIVES IN A DDR BO and is streamed ONE
ROW (128 fp32 = 512 B) at a time.  Outputs (a|b, delta|y, S2) are streamed as
1024-float packets — the proven silu/mul drain geometry that avoids the
multi-packet +2 shift bug seen at 128-element packets.  For passA/delta each
v-head gets its OWN 1024-float output packet (only the first 256 floats are
used, the rest unused) so the per-v-head C kernels can be called with the
acquired packet directly — no pointer arithmetic / memref slicing (the IRON
MemRefValue does not support `+` offset).

The scan is split across THREE xclbins, chained sequentially (passA → delta →
passB) by the engine — 3 dispatches per SSM layer:

  passA (48 v-heads × 128 rows): a = Sᵀ@kn ; b = Sᵀ@qn          # over raw S0
  delta (48 v-heads, once each):  kvm=gdec*a ; delta=(v-kvm)*beta ; c=kn·qn ;
                                 y=gdec*b + delta*c
  passB (48 v-heads × 128 rows): S2[i] = gdec*S[i] + kn[i]*delta

qn=l2norm(q)/sqrt(128), kn=l2norm(k), gdec=exp(g_logit), beta=sigmoid(b) are
done UPSTREAM on ew_unified.  All fp32.  The per-row/per-v-head C kernels in
fst_gdn_scan_kernel.cc are UNCHANGED — only the IRON worker loops and runtime
TAPs loop over the 48 v-heads here.
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
PKT_A = 130                      # [S_row(128) | kn_i(1) | qn_i(1)]
PKT_B = 264                      # [S_row(128) | delta(128) | kn_i(1) | gdec(1) | pad(5)]
DELTA_IN = 642                  # [a|b|v|kn|qn|gdec(1)|beta(1)]
DELTA_OUT = 256                 # [delta|y]
OUTPKT = 1024                   # proven silu drain packet size (>=256; first 256 used for passA/delta)

VEC    = np.ndarray[(HV,), np.dtype[np.float32]]
AB_T   = np.ndarray[(2*HV,), np.dtype[np.float32]]   # one v-head [a(128)|b(128)]
PKT_A_T = np.ndarray[(PKT_A,), np.dtype[np.float32]]
PKT_B_T = np.ndarray[(PKT_B,), np.dtype[np.float32]]
DIN_T  = np.ndarray[(DELTA_IN,), np.dtype[np.float32]]
DOUT_T = np.ndarray[(DELTA_OUT,), np.dtype[np.float32]]
OUTBLK_T = np.ndarray[(OUTPKT,), np.dtype[np.float32]]   # 1024

SRC = str(PROJ_ROOT / "fst_gdn_scan_kernel.cc")
INC = [aie_config.cxx_header_path()]
OBJ = "fst_gdn_scan.o"   # one TU, single .o (avoid duplicate symbols)


@iron.jit
def gdn_passA_op(Spkt: In, ab_out: Out):
    kA = ExternalFunction("gdn_passA_row", source_file=SRC,
        arg_types=[PKT_A_T, OUTBLK_T], include_dirs=INC, object_file_name=OBJ)
    kz = ExternalFunction("gdn_zero", source_file=SRC,
        arg_types=[OUTBLK_T], include_dirs=INC, object_file_name=OBJ)

    f_in  = ObjectFifo(PKT_A_T,   name="in", depth=2)   # shim->core, [S_row|kn_i|qn_i]
    f_ab  = ObjectFifo(OUTBLK_T,  name="ab", depth=2)   # core->shim, 1024/pkt (1 v-head, 256 used)

    def core_passA(fin, fab, kA, kz):
        for _v in range_(NV):                          # 48 v-heads, 1 dispatch
            ab = fab.acquire(1)                        # 1024 (first 256 = [a|b])
            kz(ab)                                     # zero [a|b]
            for _i in range_(NROWS):                   # 128 rows of this v-head
                p = fin.acquire(1)
                kA(p, ab)
                fin.release(1)
            fab.release(1)

    w = Worker(core_passA, [f_in.cons(), f_ab.prod(), kA, kz])

    rt = Runtime()
    with rt.sequence(np.ndarray[(NV * NROWS * PKT_A,), np.dtype[np.float32]],
                     np.ndarray[(NV * OUTPKT,),       np.dtype[np.float32]]) as (Spkt, ab_out):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), Spkt,
                tap=TensorAccessPattern(tensor_dims=(1, NV * NROWS * PKT_A), offset=0,
                                        sizes=[NV, NROWS, PKT_A],
                                        strides=[NROWS * PKT_A, PKT_A, 1]),
                task_group=tg)
        # 48 packets of 1024 — silu-proven drain geometry (no +2 shift).
        rt.drain(f_ab.cons(), ab_out,
                 tap=TensorAccessPattern(tensor_dims=(NV * OUTPKT,), offset=0,
                                         sizes=[1, NV, OUTPKT], strides=[0, OUTPKT, 1]),
                 task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def gdn_delta_op(din: In, dout: Out):
    kD = ExternalFunction("gdn_delta_full", source_file=SRC,
        arg_types=[DIN_T, OUTBLK_T], include_dirs=INC, object_file_name=OBJ)

    f_in  = ObjectFifo(DIN_T,    name="din",  depth=2)  # 642/pkt, 48 packets
    f_out = ObjectFifo(OUTBLK_T, name="dout", depth=2)  # 1024/pkt (256 used)

    def core_delta(fin, fout, kD):
        for _v in range_(NV):                          # 48 v-heads, 1 dispatch
            o = fout.acquire(1)                        # 1024 (first 256 = [delta|y])
            d = fin.acquire(1)                          # 642 for this v-head
            kD(d, o)
            fin.release(1); fout.release(1)

    w = Worker(core_delta, [f_in.cons(), f_out.prod(), kD])

    rt = Runtime()
    with rt.sequence(np.ndarray[(NV * DELTA_IN,),  np.dtype[np.float32]],
                     np.ndarray[(NV * OUTPKT,),     np.dtype[np.float32]]) as (din, dout):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), din,
                tap=TensorAccessPattern(tensor_dims=(1, NV * DELTA_IN), offset=0,
                                        sizes=[NV, DELTA_IN], strides=[DELTA_IN, 1]),
                task_group=tg)
        rt.drain(f_out.cons(), dout,
                 tap=TensorAccessPattern(tensor_dims=(NV * OUTPKT,), offset=0,
                                         sizes=[1, NV, OUTPKT], strides=[0, OUTPKT, 1]),
                 task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def gdn_passB_op(Spkt: In, s2_out: Out):
    ROWS_PER_PKT = 8
    NPKT_V = NROWS // ROWS_PER_PKT                 # 16 input packets per v-head
    NPKT = NV * NPKT_V                             # 768 packets total (48 v-heads)
    NPKT_O = 6                                     # split 768 = 6 × 128 (BD repeat ≤255)
    NPKT_I = NPKT // NPKT_O                        # 128
    PKT_B_BLK = PKT_B * ROWS_PER_PKT               # 264*8 = 2112 (8 rows)
    OUT_BLK = HV * ROWS_PER_PKT                    # 128*8 = 1024 (proven silu drain)
    BLK_IN_T  = np.ndarray[(PKT_B_BLK,), np.dtype[np.float32]]
    BLK_OUT_T = np.ndarray[(OUT_BLK,),  np.dtype[np.float32]]
    kB = ExternalFunction("gdn_passB_block", source_file=SRC,
        arg_types=[BLK_IN_T, BLK_OUT_T], include_dirs=INC, object_file_name=OBJ)

    f_in  = ObjectFifo(BLK_IN_T,  name="in",  depth=2)  # 2112/pkt (8 rows)
    f_out = ObjectFifo(BLK_OUT_T, name="out", depth=2)  # 1024/pkt (silu geometry)

    def core_passB(fin, fo, kB):
        for _ in range_(NPKT):                          # 768 blocks (all 48 v-heads)
            p = fin.acquire(1); s2 = fo.acquire(1)
            kB(p, s2)
            fin.release(1); fo.release(1)

    w = Worker(core_passB, [f_in.cons(), f_out.prod(), kB])

    rt = Runtime()
    with rt.sequence(np.ndarray[(NV * NROWS * PKT_B,), np.dtype[np.float32]],
                     np.ndarray[(NV * NROWS * HV,),    np.dtype[np.float32]]) as (Spkt, s2_out):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(f_in.prod(), Spkt,
                tap=TensorAccessPattern(tensor_dims=(1, NV * NROWS * PKT_B), offset=0,
                                        sizes=[NPKT_O, NPKT_I, PKT_B_BLK],
                                        strides=[NPKT_I * PKT_B_BLK, PKT_B_BLK, 1]),
                task_group=tg)
        # 768 packets of 1024 — silu/mul proven drain geometry (6×128 split for BD).
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
    _emit(gdn_passA_op,  "fst_gdn_passA")
    _emit(gdn_delta_op,  "fst_gdn_delta")
    _emit(gdn_passB_op,  "fst_gdn_passB")
    print(f"OK: 3 GDN xclbins, ALL 48 v-heads per dispatch -> 3 dispatches/SSM-layer")