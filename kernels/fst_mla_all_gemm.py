#!/usr/bin/env python3
# fst_mla_all_gemm.py — Vectorized MMUL xclbins for all MLA GEMM operations
#
# Generates xclbins for DeepSeek V4 Flash MLA.  Operations with N > 4096
# are split into N_chunks of N=4096 each (the S2MM BD limit of 64 with
# tile_n=64 gives max N = 64*64 = 4096).
#
# Tile sizes:   m=min(32, M%8-adjusted)  k=64  n=64
# L1 budget:    A[8or32,64] 2x = 2-8KB, B[64,64] 2x = 16KB, C[8or32,64] 2x = 2-8KB
#               Total ≤ 32KB + 3.5KB stack = 35.5KB, fits in 64KB.

import sys, time, os, shutil, numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import (CompileTime, In, Out, ObjectFifo, Program, Runtime,
                      Worker, kernels)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
set_current_device(NPU2())


def auto_tile_m(m_val):
    for c in [32, 16, 8]:
        if m_val >= c and m_val % c == 0:
            return c
    return m_val


MLA_OPS = {
    "q_compress":   (8,    4096,  1024),
    "q_expand":     (8,    1024, 32768),
    "kv_compress":  (8,    4096,   512),
    "qk":           (512,   512,   128),
    "sv":           (512,   128,   512),
    "o_proj_a":     (8,    4096,  8192),
    "o_proj_b":     (8,    8192,  4096),
}

TILE_K = 64
TILE_N = 64
CHUNK_N = 4096  # max N per GEMM call (N_dn = CHUNK_N/TILE_N = 64 ≤ 64)


def make_gemm(op_name, M_val, K_val, N_chunk):
    """Create a specialized GEMM for one chunk of N."""
    m = auto_tile_m(M_val)
    k, n = TILE_K, TILE_N
    assert M_val % m == 0 and K_val % k == 0 and N_chunk % n == 0, \
        f"Dim mismatch: M={M_val}%{m} K={K_val}%{k} N={N_chunk}%{n}"

    M_dm, K_dk, N_dn = M_val // m, K_val // k, N_chunk // n
    assert N_dn <= 64, f"NDN={N_dn} > 64"

    @iron.jit
    def ml_kernel(A: In, B: In, C: Out, *,
                  M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
                  el: CompileTime[type]):
        mm_k = kernels.mm(dim_m=m, dim_k=k, dim_n=n,
                          input_dtype=el, output_dtype=el, vectorized=True)
        zero = mm_k.zero
        r, s, t = mm_k.mac_dims
        assert m % (2*r) == 0 and k % s == 0 and n % (2*t) == 0

        Mdm, Kdk, Ndn = M // m, K // k, N // n
        tiles = Mdm * Ndn

        A_ty = np.ndarray[(M*K,), np.dtype[el]]
        B_ty = np.ndarray[(K*N,), np.dtype[el]]
        C_ty = np.ndarray[(M*N,), np.dtype[el]]
        a_ty = np.ndarray[(m, k), np.dtype[el]]
        b_ty = np.ndarray[(k, n), np.dtype[el]]
        c_ty = np.ndarray[(m, n), np.dtype[el]]

        inA = ObjectFifo(a_ty, name="inA")
        a_dims = [(m//r, r*k), (k//s, s), (r, k), (s, 1)]
        memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

        inB = ObjectFifo(b_ty, name="inB")
        b_dims = [(k//s, s*n), (n//t, t), (s, n), (t, 1)]
        memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

        memC = ObjectFifo(c_ty, name="memC")
        c_dims = [(m//r, r*n), (r, t), (n//t, r*t), (t, 1)]
        outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

        def core_fn(of_a, of_b, of_c, z, mm):
            for _ in range_(tiles) if tiles > 1 else range(1):
                co = of_c.acquire(1); z(co)
                for _ in range_(Kdk) if Kdk > 1 else range(1):
                    ca = of_a.acquire(1); cb = of_b.acquire(1)
                    mm(ca, cb, co)
                    of_a.release(1); of_b.release(1)
                of_c.release(1)

        worker = Worker(
            core_fn,
            [memA.cons(), memB.cons(), memC.prod(), zero, mm_k],
            stack_size=0xD00,
        )

        a_tap = TensorAccessPattern(
            tensor_dims=(M, K), offset=0,
            sizes=[Ndn, Kdk, m, k], strides=[0, k, K, 1],
        )
        b_tap = TensorAccessPattern(
            tensor_dims=(K, N), offset=0,
            sizes=[Ndn, Kdk, k, n], strides=[n, k*N, N, 1],
        )
        c_tap = TensorAccessPattern(
            tensor_dims=(M, N), offset=0,
            sizes=[1, Ndn, m, n], strides=[m*N, n, N, 1],
        )

        rt = Runtime()
        with rt.sequence(A_ty, B_ty, C_ty) as (a, b, c):
            rt.start(worker)
            tg = rt.task_group()
            rt.fill(inA.prod(), a, tap=a_tap, task_group=tg)
            rt.fill(inB.prod(), b, tap=b_tap, task_group=tg)
            rt.drain(outC.cons(), c, tap=c_tap, task_group=tg, wait=True)
            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()

    return ml_kernel.specialize(M=M_val, K=K_val, N=N_chunk, el=bfloat16)


def compile_all():
    print("=" * 70)
    print("Building MLA vectorized GEMM xclbins  (k=64 n=64)")
    print("=" * 70)
    for name, (M, K, N) in MLA_OPS.items():
        m = auto_tile_m(M)
        n_chunks = (N + CHUNK_N - 1) // CHUNK_N
        print(f"\n  [{name}] M={M} K={K} N={N}  tile_m={m}  chunks={n_chunks}")
        t0 = time.perf_counter()
        try:
            compiled = make_gemm(name, M, K, min(N, CHUNK_N))
            x, ii = compiled.compile()
            dt = (time.perf_counter() - t0) * 1000
            xn = f"fst_mla_{name}.xclbin"
            shutil.copy(x, xn); shutil.copy(ii, f"fst_mla_{name}_insts.bin")
            print(f"    OK  {dt:.0f}ms  {xn}")
        except Exception as e:
            print(f"    FAIL {(time.perf_counter()-t0)*1000:.0f}ms  {e}")


def benchmark_all():
    print("\n" + "=" * 70)
    print("BENCHMARK  (vectorized MMUL, BF16)")
    print("=" * 70)
    for name, (M, K, N) in MLA_OPS.items():
        m = auto_tile_m(M)
        xp = f"fst_mla_{name}.xclbin"
        if not os.path.exists(xp):
            print(f"  {name:15s} SKIP")
            continue

        rng = np.random.RandomState(42)
        Ad = (rng.randn(M, K).astype('f4') * 0.3).astype(bfloat16)
        Bd_full = (rng.randn(K, N).astype('f4') * 0.3).astype(bfloat16)
        ref = Ad.astype('f4') @ Bd_full.astype('f4')

        Ab = iron.zeros(M*K, dtype=bfloat16, device="npu"); Ab.numpy()[:] = Ad.ravel()
        Cb = iron.zeros(M*N, dtype=bfloat16, device="npu")

        compiled = make_gemm(name, M, K, min(N, CHUNK_N))
        Cb.numpy()[:] = 0

        n_chunks = (N + CHUNK_N - 1) // CHUNK_N
        n_runs = max(5 // n_chunks, 1) if n_chunks <= 5 else 1
        times_all = []

        for _ in range(3):  # warmup
            for ci in range(n_chunks):
                n0 = ci * CHUNK_N; n1 = min(n0 + CHUNK_N, N)
                Bb = iron.zeros(K * (n1 - n0), dtype=bfloat16, device="npu")
                Bb.numpy()[:] = Bd_full[:, n0:n1].ravel()
                Cb_chunk = iron.zeros(M * (n1 - n0), dtype=bfloat16, device="npu")
                compiled(Ab, Bb, Cb_chunk)
                Cb.numpy().reshape(M, N)[:, n0:n1] = \
                    Cb_chunk.numpy().reshape(M, n1 - n0).astype('f4').view(bfloat16)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            for ci in range(n_chunks):
                n0 = ci * CHUNK_N; n1 = min(n0 + CHUNK_N, N)
                Bb = iron.zeros(K * (n1 - n0), dtype=bfloat16, device="npu")
                Bb.numpy()[:] = Bd_full[:, n0:n1].ravel()
                Cb_chunk = iron.zeros(M * (n1 - n0), dtype=bfloat16, device="npu")
                compiled(Ab, Bb, Cb_chunk)
                Cb.numpy().reshape(M, N)[:, n0:n1] = \
                    Cb_chunk.numpy().reshape(M, n1 - n0).astype('f4').view(bfloat16)
            times_all.append((time.perf_counter() - t0) * 1000)

        out_npu = Cb.numpy().reshape(M, N).astype('f4')
        err = np.abs(out_npu - ref).max()
        macs = M * K * N * 2
        dt_mean = np.mean(times_all) if times_all else -1
        gflops = macs / 1e9 / (dt_mean / 1000) if dt_mean > 0 else 0
        print(f"  {name:15s} m={m} ch={n_chunks}  {dt_mean:7.2f}ms  "
              f"[{np.min(times_all):.2f}..{np.max(times_all):.2f}]  "
              f"err={err:.2e}  {gflops:.1f}GF" if times_all else
              f"  {name:15s} m={m} ch={n_chunks}  NO_RUNS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        benchmark_all()
    else:
        compile_all()
        benchmark_all()
