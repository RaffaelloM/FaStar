#!/usr/bin/env python3
"""Compile Hunyuan-3.0 (HY3) NPU expert-GEMM kernels.

HY3 expert inter_dim = 1536 (vs DeepSeek's 2048), so the DS4-baked
fst_expert_gemm_vec/down.xclbin (N/K=2048) CANNOT be reused.  This adapts the
proven BO-to-BO single_core `kernels.mm` path (compile_ffn_botobo.py +
fst_expert_gemm_vectorized.py) to HY3 shapes:

  gate/up : A[M,4096] @ B[4096,1536] -> C[M,1536]   (M=16, m16k128n32 b_col_maj)
  down    : A[M,1536] @ B[1536,4096] -> C[M,4096]   (M=16, m16k128n64 b_col_maj)

NPU2 BD-step <=64 per (N,K) dim:
  gate/up : K_div_k=4096/128=32  N_div_n=1536/32=48   (both <=64)  mmul=256
  down    : K_div_k=1536/128=12  N_div_n=4096/64=64   (both <=64)  mmul=512

Outputs (in kernels/):
  fst_hy3_gemm_vec.xclbin  + fst_hy3_gemm_vec_insts.bin   (gate/up)
  fst_hy3_gemm_down.xclbin + fst_hy3_gemm_down_insts.bin  (down)
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
    # name, M, K, N, m, k, n, b_col_maj   (HY3 inter=1536; same tiles as DS4)
    ("HY3_gateup", 16, 4096, 1536, 16, 128, 32, 1),
    # down: K=inter=1536, N=hidden=4096. n=64 -> N_div_n=64, k=128 -> K_div_k=12.
    ("HY3_down",   16, 1536, 4096, 16, 128, 64, 1),
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
        msg = str(e)[:1200]
        print(f"  FAIL: {type(e).__name__}: {msg}", flush=True)
        traceback.print_exc()
        results[name] = (False, info, None, None, 0)

print("\n\n=== SUMMARY ===")
for name, (ok, info, _, _, sz) in results.items():
    print(f"  {name:14s} {'OK' if ok else 'FAIL':5s}  {info}  ({sz}B)")

gu = results.get("HY3_gateup")
if gu and gu[0]:
    shutil.copy(gu[2], "fst_hy3_gemm_vec.xclbin")
    shutil.copy(gu[3], "fst_hy3_gemm_vec_insts.bin")
    print("\nDEPLOYED gate/up -> fst_hy3_gemm_vec.xclbin")
else:
    print("\nWARNING: gate/up did not compile; fst_hy3_gemm_vec.xclbin NOT written")

d = results.get("HY3_down")
if d and d[0]:
    shutil.copy(d[2], "fst_hy3_gemm_down.xclbin")
    shutil.copy(d[3], "fst_hy3_gemm_down_insts.bin")
    print(f"DEPLOYED down -> fst_hy3_gemm_down.xclbin")
else:
    print("WARNING: down did not compile; fst_hy3_gemm_down.xclbin NOT written")