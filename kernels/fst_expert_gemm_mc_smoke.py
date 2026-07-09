#!/usr/bin/env python3
"""Smoke test for the multi-core batched expert GEMM xclbin.

Loads fst_expert_gemm_vec_mc.xclbin via IRON's AOT compiled callable and runs it
on random A/B with a known CPU reference.  Each of the 6 expert cores should
produce C_i = A @ B_i (b_col_maj), and all 6 must match the CPU ref within bf16
tolerance.  Confirms the multi-core dispatch executes end-to-end on the NPU
and that the per-core TAP offsets address the correct expert slices.

Pass: prints "SMOKE PASS cos=<min cos over 6 experts>" and exits 0.
Fail: prints "SMOKE FAIL" with per-expert cosines and exits 1.
"""
import os, sys
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
os.environ.setdefault("XILINX_XRT", "/usr")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))

import aie.iron as iron
from aie.iron.device import NPU2
from aie.utils import set_current_device
set_current_device(NPU2())

from compile_ffn_multicore import expert_gemm_mc_op, M_FIX, K_FIX, N_FIX, TILE_M, TILE_K, TILE_N, NUM_EXPERTS

M, K, N, E = M_FIX, K_FIX, N_FIX, NUM_EXPERTS

rng = np.random.default_rng(1234)
# Small-magnitude bf16 inputs to keep bf16 accumulator error modest.
A_np = (rng.standard_normal((M, K)) * 0.25).astype(bfloat16)
# Per-expert B, b_col_maj: stored [N, K] (dequant native layout, the layout the
# b_col_maj=True kernel reads directly with NO transpose).  The batched BO is
# [E*N, K] flat — expert i's [N,K] block at rows [i*N : (i+1)*N].
B_ref = (rng.standard_normal((E, N, K)) * 0.25).astype(bfloat16)  # [E, N, K] b_col_maj
# Flat B BO: concatenate the [N,K] blocks as [E*N, K].
B_flat = np.concatenate([np.ascontiguousarray(B_ref[i]) for i in range(E)], axis=0).reshape(-1)

C_np = np.zeros((E * M * N,), dtype=bfloat16)

A_bo = iron.tensor(A_np.reshape(-1), dtype=bfloat16, device="npu")
B_bo = iron.tensor(B_flat, dtype=bfloat16, device="npu")
C_bo = iron.zeros(E * M * N, dtype=bfloat16, device="npu")

print(f"Running multi-core GEMM: M={M} K={K} N={N} E={E} tile={TILE_M}x{TILE_K}x{TILE_N}")
expert_gemm_mc_op(A_bo, B_bo, C_bo, M=M, K=K, N=N, E=E,
                  m=TILE_M, k=TILE_K, n=TILE_N, el=bfloat16)
C_out = C_bo.numpy().reshape(E * M, N)

# CPU reference: per expert, C_i = A @ B_ref[i].T  (b_col_maj: B stored [N,K], so
# the GEMM is A[M,K] @ B[N,K].T = A @ B_ref[i] -> wait, B_ref[i] is [N,K], its
# transpose is [K,N], so A @ B_ref[i].T = [M,N]).  Yes.
cos_min = 1.0
ok = True
for i in range(E):
    ref = (A_np.astype(np.float32) @ B_ref[i].T.astype(np.float32)).astype(bfloat16)
    got = C_out[i * M:(i + 1) * M]
    # cosine
    dr = ref.reshape(-1).astype(np.float64); dg = got.reshape(-1).astype(np.float64)
    dot = float((dr * dg).sum()); nr = float((dr * dr).sum()); ng = float((dg * dg).sum())
    cos = float(dot / (np.sqrt(nr) * np.sqrt(ng) + 1e-12))
    rel = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max() /
                (np.abs(ref.astype(np.float64)).max() + 1e-12))
    print(f"  expert {i}: cos={cos:.5f} max_rel_err={rel:.4f} "
          f"|got|mx={np.abs(got.astype(np.float64)).max():.3f} "
          f"|ref|mx={np.abs(ref.astype(np.float64)).max():.3f}")
    cos_min = min(cos_min, cos)
    if cos < 0.99:
        ok = False

if ok:
    print(f"SMOKE PASS cos_min={cos_min:.5f}")
    sys.exit(0)
else:
    print(f"SMOKE FAIL cos_min={cos_min:.5f}")
    sys.exit(1)