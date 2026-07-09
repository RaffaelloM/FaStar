#!/usr/bin/env python3
"""TRIAL: recompile ONLY the qc (MLA wq_a) kernel with k_tile=64 so K_div_k=64
(within the NPU2 BD-step <=64 limit), instead of k_tile=32 -> K_div_k=128.

Hypothesis: the b_col_maj B TAP group_tiler((N,K),(n,k),(N_div_n,K_div_k)) with
K_div_k=128 bakes a BD with 128 K-steps into fst_mla_qc.xclbin -> overflows the
NPU2 per-BD step limit -> ~5% magnitude inflation + cos 0.976 on random B.
K_div_k=64 should restore cos ~1.0 (XDNA2 bf16x bf16 -> float accumulator).

Tile (m=16,k=64,n=32): unroll (16/16)*(32/16)*(64/8)*16 = 1*2*8*16 = 256 (fits).
M_c=32 -> M_div_m=2 (kernel M-tiles internally, engine still passes M=32).
Writes fst_mla_qc.xclbin + fst_mla_qc_insts.bin (overwrites). Backups in /tmp.
"""
import os, shutil
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.utils import set_current_device
from aie.iron.device import NPU2

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))
set_current_device(NPU2())

# Import the vectorized MLA GEMM op from compile_mla_vec.py
import importlib.util
spec = importlib.util.spec_from_file_location("compile_mla_vec", os.path.join(os.path.dirname(__file__), "compile_mla_vec.py"))
cmv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmv)

# Host K-split fix: compile qc AND ob with K_c=1024 (K_div_k = 1024/64 = 16) so
# the baked BD has only 16 K-steps. Engine host-loops K chunks (qc: 4, ob: 8)
# accumulating partial C's in fp32 -> eliminates the K_div_k^2 BD-stride drift.
# ob keeps tile (16,64,32), N_c=2048 (caller N-tiles hd=4096 in 2048 steps).
JOBS = [
    ("qc", 16, 1024, 1024, 16, 64, 32, 1),   # M=16 standardization; K=4096 -> 4 chunks; wq_a/wkv/wgate
    ("ob", 16, 1024, 2048, 16, 64, 32, 1),   # M=16 standardization; K=8192 -> 8 chunks; wo_b (N-tiled by caller)
]
for (NAME, M, K, N, TM, TK, TN, BCM) in JOBS:
    print(f"compile {NAME}: M={M} K={K} N={N} tile {TM}x{TK}x{TN} bcm={BCM} "
          f"-> M_div_m={M//TM} K_div_k={K//TK} N_div_n={N//TN}", flush=True)
    assert K % TK == 0 and M % TM == 0 and N % TN == 0
    assert K // TK <= 32, f"K_div_k={K//TK} too large!"
    xclbin, insts = cmv.fst_mla_gemm_vec.specialize(
        M=M, K=K, N=N, m=TM, k=TK, n=TN, b_col_maj=BCM).compile()
    shutil.copy(xclbin, f"fst_mla_{NAME}.xclbin")
    shutil.copy(insts, f"fst_mla_{NAME}_insts.bin")
    print(f"  -> fst_mla_{NAME}.xclbin ({os.path.getsize(f'fst_mla_{NAME}.xclbin')}B) + "
          f"fst_mla_{NAME}_insts.bin ({os.path.getsize(f'fst_mla_{NAME}_insts.bin')}B)", flush=True)