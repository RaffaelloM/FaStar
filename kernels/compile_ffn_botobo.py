#!/usr/bin/env python3
"""BO-to-BO FFN kernel compile probe (Step 2).

Tests whether the proposed tile sizes compile for NPU2 bf16 (single_core
kernels.mm, b_col_maj, emulate_bf16_mmul_with_bfp16=False -> mac_dims (4,8,8)).

  mmul/tile = (m/4)*(k/8)*(n/8)   [empirical program-memory limit, no hard cap]

NPU2 BD step limit <=64 per dim (N, K) -> K_div_k<=64 AND N_div_n<=64.

Candidate configs:
  gate/up (M=32,K=4096,N=2048):
    GU1: m=16,k=128,n=32 -> K_div_k=32, N_div_n=64, mmul=256   (user proposal)
  down (M=32,K=2048,N=2048) -- N=2048 half, engine calls TWICE for N=4096
    (matches the working draft-path 2-half architecture; k=128 fixes the
    b_col_maj K_div_k layout bug):
    DN1: m=16,k=128,n=64 -> K_div_k=16, N_div_n=32, mmul=512   (user n=64)
    DN2: m=16,k=128,n=32 -> K_div_k=16, N_div_n=64, mmul=256   (safest, =gate/up)
"""
import os, shutil, traceback
os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

import importlib
import fst_expert_gemm_vectorized as mod
importlib.reload(mod)
fn = mod.fst_expert_gemm_vectorized

def mmul(m, k, n):  # mac_dims (4,8,8), emulate=False
    return (m // 4) * (k // 8) * (n // 8)

CANDIDATES = [
    # name, M, K, N, m, k, n, b_col_maj   (M=16 standardization; fallbacks removed)
    ("GU1_gateup", 16, 4096, 2048, 16, 128, 32, 1),
    # DOWN single-call N=4096: writes [Mx,4096] directly -> NO host interleave,
    # one readback. n=64 -> N_div_n=64 (<=64), k=128 -> K_div_k=16,
    # mmul=(16/4)*(128/8)*(64/8)=4*16*8=512.
    ("DN3_down",   16, 2048, 4096, 16, 128, 64, 1),
]

results = {}
for (name, M, K, N, m, k, n, bcm) in CANDIDATES:
    Kk, Nn = K // k, N // n
    info = f"M={M} K={K} N={N} m={m} k={k} n={n} bcm={bcm} -> K_div_k={Kk} N_div_n={Nn} mmul={mmul(m,k,n)}"
    print(f"\n=== {name}: {info} ===", flush=True)
    try:
        xclbin, insts = fn.specialize(M=M, K=K, N=N, m=m, k=k, n=n, b_col_maj=bcm).compile()
        sz = os.path.getsize(xclbin)
        print(f"  OK -> xclbin {sz}B", flush=True)
        results[name] = (True, info, xclbin, insts, sz)
    except Exception as e:
        msg = str(e)[:600]
        print(f"  FAIL: {type(e).__name__}: {msg}", flush=True)
        results[name] = (False, info, None, None, 0)

print("\n\n=== SUMMARY ===")
for name, (ok, info, _, _, sz) in results.items():
    print(f"  {name:14s} {'OK' if ok else 'FAIL':5s}  {info}  ({sz}B)" )

# Persist the winning gate/up and down as the deployed xclbins.
gu = results.get("GU1_gateup")
if gu and gu[0]:
    shutil.copy(gu[2], "fst_expert_gemm_vec.xclbin")
    shutil.copy(gu[3], "fst_expert_gemm_vec_insts.bin")
    print("\nDEPLOYED gate/up -> fst_expert_gemm_vec.xclbin")
else:
    print("\nWARNING: gate/up did not compile; fst_expert_gemm_vec.xclbin NOT updated")

# Pick the first down config that compiled (prefer DN3 single-call N=4096, then
# the 2-half N=2048 fallbacks).
for name in ("DN3_down", "DN1_down", "DN2_down"):
    d = results.get(name)
    if d and d[0]:
        shutil.copy(d[2], "fst_expert_gemm_down.xclbin")
        shutil.copy(d[3], "fst_expert_gemm_down_insts.bin")
        print(f"DEPLOYED down ({name}) -> fst_expert_gemm_down.xclbin")
        break
else:
    print("WARNING: no down config compiled; fst_expert_gemm_down.xclbin NOT updated")