#!/usr/bin/env python3
"""FaStar MoE expert GEMM — CANONICAL IRON single_core (kernels.mm + zero).

Diagnostic build: literally IRON's proven single_core matmul (b_row_maj) with
FaStar FFN gate/up shape M=32,K=4096,N=2048, tile m=k=n=32 (1 kernel, 256 mmul,
fits AIE program memory; M_div_m=1 simplest).  Used to validate that the
dims_to_stream + TAP layout is correct INDEPENDENT of the custom set/acc kernel.

    A : [M=32, K=4096]            (activations)
    B : [K=4096, N=2048]          (weights, b_row_maj)
    C : [M=32, N=2048] = A @ B

Probe: A=ones, B=[K,N] identity -> C must be all 1.0.  If `zero` clears the
ObjectFifo slot correctly, this PASSES.  If it shows the (1 + t//2)x residual,
SET-then-ACC is confirmed necessary (fst_aie_mm_vec.cc).
"""

import os, shutil
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker,
                      kernels, str_to_dtype)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.utils import set_current_device
from aie.iron.device import NPU2

set_current_device(NPU2())

M_FIX = 32
K_FIX = 4096
N_FIX = 2048
TILE_M = 32
TILE_K = 32
TILE_N = 32


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def fst_expert_gemm_canonical(
    A: In, B: In, C: Out, *,
    M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    m: CompileTime[int], k: CompileTime[int], n: CompileTime[int]):
    dtype_in = bfloat16
    dtype_out = bfloat16

    assert M % m == 0 and K % k == 0 and N % n == 0
    matmul_kernel = kernels.mm(dim_m=m, dim_k=k, dim_n=n,
                               input_dtype=dtype_in, output_dtype=dtype_out,
                               b_col_maj=False, use_chess=False,
                               emulate_bf16_mmul_with_bfp16=False)
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert m % r == 0 and k % s == 0 and n % t == 0

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]  # b_row_maj
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, zero, matmul):
        loop = range_(tiles) if tiles > 1 else range(1)
        for _ in loop:
            elem_out = of_c.acquire(1)
            zero(elem_out)
            inner = range_(K_div_k) if K_div_k > 1 else range(1)
            for _ in inner:
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn,
                    [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
                    stack_size=0xD00)

    rows_per_block = 2  # M_div_m=1 → need rows_per_block//2=1
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n, prune_step=False)
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K_div_k, N_div_n),
        tile_group_col_major=True, prune_step=False)[0]
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m, n), (rows_per_block // 2, N_div_n), prune_step=False)
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tgs = []
        for tile_row_block in range(iron.ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = (tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2)
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index],
                         task_group=tgs[-1], wait=True)
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    return Program(NPU2(), rt).resolve_program()


def aot_compile() -> None:
    xclbin, insts = fst_expert_gemm_canonical.specialize(
        M=M_FIX, K=K_FIX, N=N_FIX, m=TILE_M, k=TILE_K, n=TILE_N).compile()
    out_xcl = "fst_expert_gemm_vec.xclbin"
    out_ins = "fst_expert_gemm_vec_insts.bin"
    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)
    print(f"AOT compiled CANONICAL single_core GEMM {M_FIX}x{K_FIX}x{N_FIX} bf16 "
          f"(tile {TILE_M}x{TILE_K}x{TILE_N}, b_row_maj, kernels.mm+zero)")
    print(f"  xclbin: {os.path.abspath(out_xcl)} ({os.path.getsize(out_xcl)}B)")
    print(f"  insts:  {os.path.abspath(out_ins)} ({os.path.getsize(out_ins)}B)")


if __name__ == "__main__":
    aot_compile()