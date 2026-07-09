#!/usr/bin/env python3
"""FaStar MULTI-CORE BATCHED expert GEMM — 6 routed experts in ONE xclbin dispatch.

Collapses the engine's per-expert gate/up GEMM loop (6 routed experts × 1 dispatch
= 6 dispatches/op) into ONE multi-core dispatch where 6 AIE cores each GEMM one
expert's slice of a batched B[E,K,N] BO.  A is shared (same hidden state for all
experts — the engine replicates h to Mx rows once per layer).

Modeled on compile_ffn_unified.py `gate_op` (single Worker, simple TensorAccessPattern)
extended to NUM_EXPERTS Workers via per-core ObjectFifos + per-core TAP offsets, the
same pattern compile_ffn_unified.py `dequant_op` uses for its 16-core dequant.

Layout (b_col_maj, matching fst_expert_gemm_vectorized.py b_col_maj=1):
    A : [M, K]              shared activation (replicated to all 6 workers via 6 fills)
    B : [E, K, N] flat as [E*K, N]  batched dequanted weights, one [K,N] slice/expert
    C : [E, M, N] flat as [E*M, N]  batched output, one [M,N] slice/expert
    M=16, K=4096, N=2048, tile 16x128x64  (matches working M=16 gemm_vec)

Each of the 6 workers runs the FULL M=16,K=4096,N=2048 GEMM (M_div_m=1, N_div_n=32,
K_div_k=32 → 32 output tiles × 32 K-steps) on its own expert slice.  Same total work
as 6 sequential single-core dispatches, parallelised across 6 AIE cores.

The engine calls this xclbin TWICE per layer for the routed FFN: once for "gate",
once for "up" (identical shape — same kernel, different B sub-buffer), exactly as
the single-core fst_expert_gemm_vec.xclbin is shared by "gemm" and "gemm2" today.
The DOWN projection (M=16,K=2048,N=4096) is a different shape and needs its own
multi-core variant — NOT in this file (see fst_expert_gemm_down_mc).

Output: fst_expert_gemm_vec_mc.xclbin + fst_expert_gemm_vec_mc_insts.bin
(NOT overwriting fst_expert_gemm_vec.xclbin — the single-core kernel is still used
by the engine's per-expert dispatch path until the engine is rewired to batch.
Adding this xclbin costs +1 hw_context: engine is at 8/8 → 9/9 = driver cap.)
"""
import os, sys, shutil, time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
set_current_device(NPU2())

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Shape ───────────────────────────────────────────────────────────────────
M_FIX, K_FIX, N_FIX = 16, 4096, 2048
TILE_M, TILE_K, TILE_N = 16, 128, 64
NUM_EXPERTS = 6          # 6 routed experts (router top-k) processed in parallel
EL = bfloat16


@iron.jit
def expert_gemm_mc_op(A: In, B: In, C: Out, *,
                      M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
                      E: CompileTime[int],
                      m: CompileTime[int], k: CompileTime[int], n: CompileTime[int],
                      el: CompileTime[type]):
    """Multi-core batched expert GEMM.  E workers, each GEMMs one expert slice.

    A : [M, K]            shared (replicated to all workers via per-worker fills)
    B : [E*K, N]          batched dequanted weights, expert i at rows [i*K : (i+1)*K]
    C : [E*M, N]          batched output, expert i at rows [i*M : (i+1)*M]
    b_col_maj (B stored [N, K] per expert, transposed-B GEMM) — matches the
    fst_expert_gemm_vectorized b_col_maj=1 layout the dequant kernel emits.
    """
    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n,
                    input_dtype=el, output_dtype=el,
                    b_col_maj=True, use_chess=False,
                    emulate_bf16_mmul_with_bfp16=False)
    z = mm.zero
    r, s, t = mm.mac_dims
    assert m % r == 0 and k % s == 0 and n % t == 0

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    a_ty = np.ndarray[(m, k), np.dtype[el]]
    b_ty = np.ndarray[(k, n), np.dtype[el]]
    c_ty = np.ndarray[(m, n), np.dtype[el]]

    # Per-core ObjectFifos + dims_to_stream forwarding (matches the proven
    # fst_expert_gemm_vectorized structure: DMA fills the raw fifo, a forwarded
    # fifo reshapes the flat tile into the mmul's native (r,s,t) layout so the
    # b_col_maj kernel interprets B correctly).  Without this forwarding the
    # raw [k,n] tile is misinterpreted by the mmul → cos≈0.
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]   # b_col_maj
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

    def make_fifos(i):
        inA = ObjectFifo(a_ty, name=f"mc_inA_{i}", depth=2)
        inB = ObjectFifo(b_ty, name=f"mc_inB_{i}", depth=2)
        memC = ObjectFifo(c_ty, name=f"mc_memC_{i}", depth=2)
        memA = inA.cons().forward(name=f"mc_memA_{i}", dims_to_stream=a_dims)
        memB = inB.cons().forward(name=f"mc_memB_{i}", dims_to_stream=b_dims)
        outC = memC.cons().forward(name=f"mc_outC_{i}", dims_to_stream=c_dims)
        return inA, inB, memC, memA, memB, outC

    fifos = [make_fifos(i) for i in range(E)]

    def make_core(idx):
        def core_fn(of_a, of_b, of_c, zero_k, mm_k):
            loop = range_(tiles) if tiles > 1 else range(1)
            for _ in loop:
                ec = of_c.acquire(1); zero_k(ec)
                inner = range_(K_div_k) if K_div_k > 1 else range(1)
                for _ in inner:
                    ea = of_a.acquire(1); eb = of_b.acquire(1)
                    mm_k(ea, eb, ec)
                    of_a.release(1); of_b.release(1)
                of_c.release(1)
        return core_fn

    workers = [Worker(make_core(i),
                     [fifos[i][3].cons(),   # memA (forwarded A consumer)
                      fifos[i][4].cons(),   # memB (forwarded B consumer)
                      fifos[i][2].prod(),   # memC (original C producer — worker writes raw)
                      z, mm],
                     stack_size=0xD00) for i in range(E)]

    # ── Per-worker TAPs ──────────────────────────────────────────────────────
    # A is the SAME [M,K] for every worker → all E fills use identical TAP
    # (replicated DMA; A is small: M*K = 16*4096 = 64K elems = 128 KB).
    # gate_op A TAP: TensorAccessPattern((M,K), 0, [N//n, K//k, m, k], [0, k, K, 1])
    # (M_div_m=1 here → outermost size is N_div_n, iterating over the N tile blocks)
    taps_A = [TensorAccessPattern(
                (M, K), 0,
                [N_div_n, K_div_k, m, k],
                [0, k, K, 1]) for _ in range(E)]

    # B is batched b_col_maj: each expert's [N,K] block (dequant native layout),
    # concatenated as [E*N, K] flat.  Worker i reads its [N,K] slice at row offset
    # i*N.  The b_col_maj GEMM (kernels.mm b_col_maj=True, matching the working
    # fst_expert_gemm_vectorized b_col_maj=1 path) reads B as [N,K] and transposes
    # on the fly.  TAP walks [N,K]: outer N_div_n tile-rows, K_div_k tile-cols,
    # inner n×k tile (row-major within the [N,K] block).
    #   strides: [n*K, k, K, 1]  (N-tile-row step = n*K elems; K-tile-col step = k;
    #           row-in-tile step = K; col-in-tile step = 1)
    taps_B = [TensorAccessPattern(
                (E * N, K), i * N * K,
                [N_div_n, K_div_k, n, k],
                [n * K, k, K, 1]) for i in range(E)]

    # C is batched [E*M, N] flat.  Worker i writes its [M,N] slice at row offset i*M.
    # gate_op C TAP: TensorAccessPattern((M,N), off, [1, N//n, m, n], [m*N, n, N, 1])
    # Per-expert offset = i * M * N.
    taps_C = [TensorAccessPattern(
                (E * M, N), i * M * N,
                [1, N_div_n, m, n],
                [m * N, n, N, 1]) for i in range(E)]

    # Runtime tensor types: flat batched BOs.
    A_ty = np.ndarray[(M * K,), np.dtype[el]]
    B_ty = np.ndarray[(E * K * N,), np.dtype[el]]
    C_ty = np.ndarray[(E * M * N,), np.dtype[el]]

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a, b, c):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(E):
            rt.fill(fifos[i][0].prod(), a, tap=taps_A[i], task_group=tg)   # inA
        for i in range(E):
            rt.fill(fifos[i][1].prod(), b, tap=taps_B[i], task_group=tg)   # inB
        for i in range(E):
            rt.drain(fifos[i][5].cons(), c, tap=taps_C[i], task_group=tg, wait=True)  # outC (forwarded)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def main():
    print("=== Compile Multi-core Batched Expert GEMM ===")
    print(f"  shape: M={M_FIX} K={K_FIX} N={N_FIX}  tile={TILE_M}x{TILE_K}x{TILE_N}")
    print(f"  experts(parallel cores): {NUM_EXPERTS}")
    print(f"  per-expert tiles: M_div_m={M_FIX//TILE_M} K_div_k={K_FIX//TILE_K} "
          f"N_div_n={N_FIX//TILE_N} = {(M_FIX//TILE_M)*(N_FIX//TILE_N)} out tiles × "
          f"{K_FIX//TILE_K} K-steps")
    t0 = time.perf_counter()

    out_xcl = PROJ_ROOT / "fst_expert_gemm_vec_mc.xclbin"
    out_ins = PROJ_ROOT / "fst_expert_gemm_vec_mc_insts.bin"

    try:
        xclbin, insts = expert_gemm_mc_op.specialize(
            M=M_FIX, K=K_FIX, N=N_FIX, E=NUM_EXPERTS,
            m=TILE_M, k=TILE_K, n=TILE_N, el=EL).compile()
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"\n✗ COMPILE FAILED after {dt:.1f}s")
        print(f"  {type(e).__name__}: {e}")
        # Dump full traceback for diagnosis
        import traceback
        traceback.print_exc()
        return 1

    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)

    dt = time.perf_counter() - t0
    print(f"\n✓ COMPILED in {dt:.1f}s")
    print(f"  xclbin: {out_xcl} ({out_xcl.stat().st_size}B)")
    print(f"  insts:  {out_ins} ({out_ins.stat().st_size}B)")
    print(f"\nNOTE: +1 hw_context vs fst_expert_gemm_vec.xclbin.  Engine is at 8/8")
    print(f"active contexts (driver cap 9) — wiring this in hits 9/9 (no headroom).")
    print(f"DOWN GEMM (M=16,K=2048,N=4096) needs its own multi-core variant")
    print(f"(fst_expert_gemm_down_mc) — not compiled here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())