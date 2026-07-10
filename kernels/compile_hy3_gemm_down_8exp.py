#!/usr/bin/env python3
"""Compile the HY3 8-expert down-projection GEMM (multi-core, ONE dispatch).

Collapses the 8 routed experts' down GEMMs (8 dispatches in the current per-op
path) into a SINGLE multicore dispatch where 8 AIE cores each GEMM one expert's
down slice.  Reuses the proven `expert_gemm_mc_op` from `compile_ffn_mc_batch.py`
(DS4 MC batched GEMM, HW-verified cos 0.9988) — only the HY3 down shape + E=8
differ.

Layout (matches the 8-expert dequant output, which is expert-major gate|up|down
contiguous = the 3x-stride [E,3,N,K] layout the MC kernel reads):
  A (hidden, a_batched=True): [E*M, K] = [128, 1536]  — expert i's silu*up hidden
                                at rows [i*16, (i+1)*16]  (differs per expert).
  B (down weights):            [E*3*N, K] view of the 302 MB dequant BO; the engine
                                passes B as a sub-buffer at base +2*N*K (elems) so
                                worker i reads expert i's DOWN slice at
                                offset 2*N*K + i*3*N*K.  (gate/up live at +0/+N*K
                                and are skipped by this down xclbin.)
  C (output):                  [E*M, N] = [128, 4096]  — expert i's down[16,4096]
                                at rows [i*16, (i+1)*16].

Shape: M=16 K=1536 N=4096 tile 16x128x64 (K_div_k=12, N_div_n=64, both <= 64
NPU2 BD-step).  b_col_maj (B is [N,K] native per expert).

Output (in kernels/):
  fst_hy3_gemm_down_8exp.xclbin + fst_hy3_gemm_down_8exp_insts.bin
"""
import os, sys, shutil, time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find sibling module

from compile_ffn_mc_batch import expert_gemm_mc_op, PROJ_ROOT  # reuse proven op

EL = bfloat16
N_EXPERTS = 8


def main():
    name = "fst_hy3_gemm_down_8exp"
    M, K, N = 16, 1536, 4096          # HY3 down: A[16,1536] @ B[4096,1536] -> C[16,4096]
    tile_m, tile_k, tile_n = 16, 128, 64
    K_div_k, N_div_n = K // tile_k, N // tile_n   # 12, 64
    assert K_div_k <= 64 and N_div_n <= 64, f"BD-step violation: K_div_k={K_div_k} N_div_n={N_div_n}"

    print(f"=== Compile {name}: M={M} K={K} N={N} tile={tile_m}x{tile_k}x{tile_n} "
          f"E={N_EXPERTS} a_batched=True (K_div_k={K_div_k} N_div_n={N_div_n}) ===")
    t0 = time.perf_counter()
    out_xcl = PROJ_ROOT / f"{name}.xclbin"
    out_ins = PROJ_ROOT / f"{name}_insts.bin"
    xclbin, insts = expert_gemm_mc_op.specialize(
        M=M, K=K, N=N, E=N_EXPERTS,
        m=tile_m, k=tile_k, n=tile_n, el=EL, a_batched=True).compile()
    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)
    dt = time.perf_counter() - t0
    a_elems = N_EXPERTS * M * K          # 128 * 1536
    b_elems = N_EXPERTS * 3 * K * N      # 150,994,944 (matches dequant output)
    c_elems = N_EXPERTS * M * N          # 128 * 4096
    print(f"✓ {name} COMPILED in {dt:.1f}s  xclbin={out_xcl.stat().st_size}B "
          f"insts={out_ins.stat().st_size}B")
    print(f"  A(hidden) BO: {a_elems*2} B, B(down) BO: {b_elems*2} B ({b_elems*2/1e6:.1f} MB), "
          f"C(out) BO: {c_elems*2} B ({c_elems*2/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())