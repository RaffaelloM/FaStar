#!/usr/bin/env python3
"""Compile the HY3 8-expert gate/up-projection GEMM (multi-core, shared xclbin).

Collapses the 8 routed experts' gate AND up GEMMs into a SINGLE multicore xclbin
(two dispatches at runtime: gate with B sub-buffer @0, up with B sub-buffer
@+N*K).  8 AIE cores each GEMM one expert's gate/up slice.  Reuses the proven
`expert_gemm_mc_op` from `compile_ffn_mc_batch.py` (DS4 MC batched GEMM,
HW-verified cos 0.9988) — only the HY3 gate/up shape + E=8 differ.

Layout (matches the 8-expert dequant output = 3x-stride [E,3,N,K]):
  A (h, a_batched=False): [M, K] = [16, 4096]  — shared activation, replicated to
                            all 8 workers (replayed across N-tiles).
  B (gate/up weights):    [E*3*N, K] view of the 302 MB dequant BO.  For GATE the
                            engine passes B @0 (worker i reads expert i's gate at
                            i*3*N*K); for UP it passes B sub-buffer @+N*K (worker
                            i reads expert i's up at N*K + i*3*N*K).
  C (gate or up output):  [E*M, N] = [128, 1536]  — expert i at rows [i*16,(i+1)*16].

Shape: M=16 K=4096 N=1536 tile 16x128x64 (K_div_k=32, N_div_n=24, both <= 64
NPU2 BD-step).  b_col_maj (B is [N,K] native per expert).

Output (in kernels/):
  fst_hy3_gemm_vec_mc_8exp.xclbin + fst_hy3_gemm_vec_mc_8exp_insts.bin
"""
import os, sys, shutil, time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compile_ffn_mc_batch import expert_gemm_mc_op, PROJ_ROOT  # reuse proven op

EL = bfloat16
N_EXPERTS = 8


def main():
    name = "fst_hy3_gemm_vec_mc_8exp"
    M, K, N = 16, 4096, 1536         # HY3 gate/up: A[16,4096] @ B[1536,4096] -> C[16,1536]
    tile_m, tile_k, tile_n = 16, 128, 64
    K_div_k, N_div_n = K // tile_k, N // tile_n   # 32, 24
    assert K_div_k <= 64 and N_div_n <= 64, f"BD-step violation: K_div_k={K_div_k} N_div_n={N_div_n}"

    print(f"=== Compile {name}: M={M} K={K} N={N} tile={tile_m}x{tile_k}x{tile_n} "
          f"E={N_EXPERTS} a_batched=False (K_div_k={K_div_k} N_div_n={N_div_n}) ===")
    t0 = time.perf_counter()
    out_xcl = PROJ_ROOT / f"{name}.xclbin"
    out_ins = PROJ_ROOT / f"{name}_insts.bin"
    xclbin, insts = expert_gemm_mc_op.specialize(
        M=M, K=K, N=N, E=N_EXPERTS,
        m=tile_m, k=tile_k, n=tile_n, el=EL, a_batched=False).compile()
    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)
    dt = time.perf_counter() - t0
    a_elems = M * K                    # 65536 (replicated)
    b_elems = N_EXPERTS * 3 * K * N    # 150,994,944 (matches dequant output)
    c_elems = N_EXPERTS * M * N        # 196608
    print(f"✓ {name} COMPILED in {dt:.1f}s  xclbin={out_xcl.stat().st_size}B "
          f"insts={out_ins.stat().st_size}B")
    print(f"  A(h) BO: {a_elems*2} B (replicated), B(gate/up) BO: {b_elems*2} B "
          f"({b_elems*2/1e6:.1f} MB), C(gate/up) BO: {c_elems*2} B ({c_elems*2/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())