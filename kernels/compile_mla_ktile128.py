#!/usr/bin/env python3
"""Device-side K fix, SIMPLE variant: k_tile=128 -> K_div_k=32 (within the NPU2
per-BD <=64 safe zone) using the ORIGINAL proven group_tiler structure (one BD,
no step_tiler K-chunking, no deadlock risk).  Reuses compile_mla_vec.py's
fst_mla_gemm_vec unchanged -- only the tile dims differ.

For qc (K=4096, N=1024): m=16, k=128, n=16 -> K_div_k=32, N_div_n=64, unroll 256.
No N-split needed (N_div_n=64 <= 64).  Streams full K=4096 on-device in ONE kernel
call (no host K-loop).  Probe must show cos>0.99 AND |npu|/|ref|~=1.000.
"""
import os, shutil, importlib
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.utils import set_current_device
from aie.iron.device import NPU2

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
set_current_device(NPU2())

import compile_mla_vec as cmv
importlib.reload(cmv)
fst_mla_gemm_vec = cmv.fst_mla_gemm_vec

# (name, M, K, N, tile_m, tile_k, tile_n, b_col_maj)
JOBS = [
    # M=16 standardization: qck problem-M=16, tile m=16 -> 1 M-tile (was M=32).
    ("qck",  16, 4096, 1024, 16, 128, 16, 1),   # K_div_k=32, N_div_n=64, unroll 256
]
for (NAME, M, K, N, TM, TK, TN, BCM) in JOBS:
    print(f"compile {NAME}: M={M} K={K} N={N} tile {TM}x{TK}x{TN} bcm={BCM} "
          f"-> K_div_k={K//TK} N_div_n={N//TN}", flush=True)
    assert K % TK == 0 and M % TM == 0 and N % TN == 0
    xclbin, insts = fst_mla_gemm_vec.specialize(
        M=M, K=K, N=N, m=TM, k=TK, n=TN, b_col_maj=BCM).compile()
    shutil.copy(xclbin, f"fst_mla_{NAME}.xclbin")
    shutil.copy(insts, f"fst_mla_{NAME}_insts.bin")
    print(f"  -> fst_mla_{NAME}.xclbin ({os.path.getsize(f'fst_mla_{NAME}.xclbin')}B)", flush=True)