#!/usr/bin/env python3
"""Correctness gate for the transposed-B expert GEMM kernels.

The MXFP4 dequant emits B stored [N, K] row-major (out-contiguous).  The
recompiled GEMM kernels must compute C = A @ Bᵀ with that layout — NOT
A @ B (which would be the old B[K, N] interpretation of the same bytes).

This harness runs each compiled kernel on the NPU with deterministic
small-int A and B, then compares the NPU C against:
  * C_correct = A @ B.reshape(N, K).T   (the intended transposed-B result)
  * C_wrong   = A @ B.reshape(K, N)      (the old B[K, N] interpretation)

PASS iff the NPU result matches C_correct (within BF16 accumulation
tolerance) AND is far closer to C_correct than to C_wrong.  This catches an
off-by-transpose bug (which would match C_wrong instead).
"""
import os
import sys

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault(
    "PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"),
)

import numpy as np
import torch
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import NPU2
from aie.utils import set_current_device
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

set_current_device(NPU2())

# Import the two compiled kernels (gate/up and down).
from fst_expert_gemm_vectorized import fst_expert_gemm_vectorized as kern_gate
from fst_expert_gemm_vectorized_down import fst_expert_gemm_vectorized as kern_down


def run_case(name, kern, M, K, N, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # A [M, K], B [N, K] (transposed layout: out-major).  Small ints so the
    # BF16 accumulation is well-conditioned and a wrong transpose is obvious.
    A_np = rng.integers(-2, 3, size=(M, K)).astype(np.float32)
    B_nk = rng.integers(-2, 3, size=(N, K)).astype(np.float32)   # B stored [N, K]

    # NPU tensors.  B_ty is flat (N*K,) — pass [N, K] 2D, IRON flattens.
    A_t = XRTTensor.from_torch(torch.from_numpy(A_np).to(torch.bfloat16), device="npu")
    B_t = XRTTensor.from_torch(torch.from_numpy(B_nk).to(torch.bfloat16), device="npu")
    C_t = iron.zeros((M * N,), dtype=bfloat16, device="npu")

    kern(A_t, B_t, C_t, M=M, K=K, N=N, element_type=bfloat16)

    C_npu = C_t.numpy().astype(np.float32).reshape(M, N)

    # References (float32).
    C_correct = A_np @ B_nk.T                  # A @ Bᵀ  (B is [N, K])  -> [M, N]
    B_kn = B_nk.reshape(K, N)                   # old B[K, N] interpretation of same bytes
    C_wrong = A_np @ B_kn                       # A @ B (old layout)   -> [M, N]

    err_correct = np.max(np.abs(C_npu - C_correct))
    err_wrong = np.max(np.abs(C_npu - C_wrong))
    max_c = float(np.max(np.abs(C_correct))) + 1e-6

    # Tolerance: BF16 accumulation over K gives ~5-8% relative error (3 sig
    # bits); the real correctness signal is the 10x separation from the
    # wrong-transpose reference.  10% absorbs BF16 noise while still rejecting
    # any genuinely-wrong layout (which lands at ~100% error).
    ok = (err_correct < 0.10 * max_c) and (err_correct * 10 < err_wrong)
    print(f"[{name}] M={M} K={K} N={N}")
    print(f"    max|C|          = {max_c:.1f}")
    print(f"    err vs A@Bᵀ     = {err_correct:.2f}  (correct, want < {0.10*max_c:.1f})")
    print(f"    err vs A@B(K,N) = {err_wrong:.2f}  (wrong-transpose, want >> correct)")
    print(f"    => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    # gate/up: A[M=32, K=4096] @ Bᵀ[N=2048, K=4096] -> C[32, 2048]
    results.append(run_case("gate/up", kern_gate, 32, 4096, 2048, seed=1))
    # down: A[M=32, K=2048] @ Bᵀ[N=4096, K=2048] -> C[32, 4096]
    results.append(run_case("down", kern_down, 32, 2048, 4096, seed=2))
    print("\n==== SUMMARY ====")
    print("transposed-B GEMM correctness:", "ALL PASS" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()