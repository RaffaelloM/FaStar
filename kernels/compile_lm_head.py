#!/usr/bin/env python3
"""FaStar LM Head GEMM — specialized for DSpark SD verify.

Scalar shim-DMA BF16 matmul for M=8, K=4096, N=129280.
Uses the simple shim-DMA pattern (no mem-tile, no dims_to_stream)
to avoid the 4-dimension BD limit of the vectorized path.

A : [M, K] hidden state    (M=8,    K=4096)
B : [K, N] lm_head weights  (K=4096, N=129280)
C : [M, N] logits

Tiling: TILE_M=8, TILE_K=64, TILE_N=64.
N_div_n = 2020, K_div_k = 64, tiles = 1 * 2020 = 2020.
Each tile: 8x64 x 64x64 -> 8x64 output via scalar mmul.
"""

import os, sys, time, shutil
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
set_current_device(NPU2())

M_FIX = 8
K_FIX = 4096
# Pad N to 131072 (= 2048*64) so N_div_n=2048 and the 4D TAP dims stay <= 64.
# 2048 = 64 * 32, so outer=64, inner=32.  But 4D limit means we use:
#   [N_inner=32, N_outer=64, K_div_k=64, m*k]  -> 4 dims but m*k=512 > 64.
# Instead: use scalar shim-DMA with flat fifos (no dims_to_stream).
# The TAP has 4 dims: [N_div_n, K_div_k, m, k] but N_div_n=2048 > 64.
# Fix: split into [N_outer=64, K_div_k=64, m, k] and stride to skip
#   every 64th N-tile.  But N_outer=64 iterates 64 N-tiles, and we need
#   2048 total -> need 32 outer iterations -> 5th dim.
#
# Simplest: use TILE_N = 2048 so N_div_n = 64.  2048 is too large for
#   a single mmul tile, but the scalar kernel can handle it (the inner
#   K-loop accumulates).  Actually TILE_N must be valid for kernels.mm.
#
# Use: TILE_N = 64, N_FIX = 64 * 64 * 32 = 131072 (pad).
#   N_div_n = 2048.  Use TAP with [64, 64, 8, 64] and stride tricks.
# But 4D: [N1=64, N2=32, K=64, m*k=512] -> m*k > 64.
#
# Final approach: scalar kernel, TILE_M=8, TILE_K=64, TILE_N=64.
#   Flat ObjectFifo (1D), 4D TAP: [K_div_k, N_div_n, m, k]
#   But N_div_n=2048 > 64.  Split: N1=64, N2=32, K=64, m=8 -> 4D but
#   the m=8 dim and k=64 dim are separate -> 5D.
#
# Actually: [N1=64, N2=32, K_div_k=64, m*k] is 4D but m*k=512 > 64.
# [N1=64, N2=32, m, k] is 4D with K tiled inside the core loop.
# strides: N1 stride=0 (A), N2 stride=n, m stride=K, k stride=1.
# But K_div_k=64 iterations are in the core, not the TAP.
# So TAP: sizes=[N1=64, N2=32, m=8, k=64], 4D, each <= 64.
# A strides: [0, 0, K, 1]  (A doesn't change with N)
# B strides: [N2*n, n, N, 1]  (B changes with both N dims)
# C strides: [N2*n, n, N, 1]  (C changes with both N dims)
# This is 4D, each size <= 64.  K is tiled in the core loop.

N_FIX = 64 * 64 * 32  # 131072 (pad from 129280)

TILE_M = 8
TILE_K = 64
TILE_N = 64


@iron.jit
def fst_lm_head(
    input0: In,
    input1: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    el: CompileTime[type],
):
    m, k, n = TILE_M, TILE_K, TILE_N

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=el, output_dtype=el, vectorized=False)
    z = mm.zero

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n  # 2048
    N_outer = 64
    N_inner = N_div_n // N_outer  # 32
    tiles = M_div_m * N_div_n

    fifo_A = ObjectFifo(np.ndarray[(m * k,), np.dtype[el]], name="Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k * n,), np.dtype[el]], name="Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m * n,), np.dtype[el]], name="Cout", depth=2)

    def core(of_a, of_b, of_c, zero_k, mm_k):
        for _ in range_(tiles) if tiles > 1 else range(1):
            ec = of_c.acquire(1); zero_k(ec)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                ea = of_a.acquire(1); eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1); of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm],
               stack_size=0xD00)

    # 4D TAP: [N_inner=32, N_outer=64, m=8, k=64] — all dims <= 64.
    # K is tiled inside the core loop (K_div_k=64 iterations).
    # A: re-read for each N-tile (stride = m*k = 512, the tile size)
    at = TensorAccessPattern((M, K), 0,
        [N_inner, N_outer, m, k],
        [m * k, m * k, K, 1])
    bt = TensorAccessPattern((K, N), 0,
        [N_inner, N_outer, m, k],
        [N_outer * n, n, N, 1])
    ct = TensorAccessPattern((M, N), 0,
        [1, N_inner, N_outer, m * n],
        [m * N, N_outer * n, n, 1])

    rt = Runtime()
    with rt.sequence(
        np.ndarray[(M, K), np.dtype[el]],
        np.ndarray[(K, N), np.dtype[el]],
        np.ndarray[(M, N), np.dtype[el]],
    ) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    t0 = time.perf_counter()
    xclbin, insts = fst_lm_head.specialize(
        M=M_FIX, K=K_FIX, N=N_FIX, el=bfloat16
    ).compile()
    dt = time.perf_counter() - t0
    target_x = "/home/raffaele/Progetti/FaStar/fst_lm_head.xclbin"
    target_i = "/home/raffaele/Progetti/FaStar/fst_lm_head_insts.bin"
    shutil.copy2(xclbin, target_x)
    shutil.copy2(insts, target_i)
    print(f"LM Head GEMM {M_FIX}x{K_FIX}x{N_FIX} bf16 compiled ({dt:.1f}s)")
    print(f"  xclbin: {os.path.getsize(target_x)} bytes")
    print(f"  insts:  {os.path.getsize(target_i)} bytes")