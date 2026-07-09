#!/usr/bin/env python3
"""FaStar MoE expert DOWN GEMM (CANONICAL IRON single_core, b_col_maj, N=2048).

DeepSeek V4 Flash down projection is M=32, K=2048, N=4096.  N=4096 exceeds the
NPU2 BD step limit for a single fill (N_div_n>64 with n=32; n=64 produces only
half the N-tiles at runtime).  The dequant kernel emits B in [N,K] layout
(native), which is contiguous per N-half.  So the down kernel is compiled
b_col_maj at N=2048 (B stored [N=2048, K=2048]) and the engine calls it TWICE
for the two contiguous N-halves (rows 0-2047 and 2048-4095 of the [N=4096,K=2048]
dequant output).  NO transpose needed for the down projection.

    half:  A [32, 2048], B [N=2048, K=2048] (b_col_maj), C [32, 2048] = A @ B
    tile m=k=n=32 → N_div_n=64, K_div_k=64, mmul=256 (fits AIE program memory).
"""

import os, shutil
from fst_expert_gemm_vectorized import fst_expert_gemm_vectorized

M_FIX = 32
K_FIX = 2048
N_FIX = 2048
TILE_M = 16
TILE_K = 128
TILE_N = 64


def aot_compile() -> None:
    xclbin, insts = fst_expert_gemm_vectorized.specialize(
        M=M_FIX, K=K_FIX, N=N_FIX, m=TILE_M, k=TILE_K, n=TILE_N,
        b_col_maj=1).compile()
    out_xcl = "fst_expert_gemm_down.xclbin"
    out_ins = "fst_expert_gemm_down_insts.bin"
    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)
    print(f"AOT compiled CANONICAL single_core DOWN-half GEMM "
          f"{M_FIX}x{K_FIX}x{N_FIX} bf16 (tile {TILE_M}x{TILE_K}x{TILE_N}, b_col_maj)")
    print(f"  xclbin: {os.path.abspath(out_xcl)} ({os.path.getsize(out_xcl)}B)")
    print(f"  insts:  {os.path.abspath(out_ins)} ({os.path.getsize(out_ins)}B)")
    print("  NOTE: engine calls this TWICE for N=4096 (two contiguous [N=2048,K=2048] halves)")


if __name__ == "__main__":
    aot_compile()