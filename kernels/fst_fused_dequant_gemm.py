#!/usr/bin/env python3
"""FaStar Fused Dequant+GEMM — IRON design for XDNA2 (NPU2).

Single xclbin that does FPGA dequant + MMUL on the same compute tile.
No intermediate BF16 weight buffer — FP4 data arrives via ObjectFifo,
the kernel dequantizes it into a local 8KB buffer, then runs aie::mmul.

Outputs: 7× less DMA traffic (16.8 MB vs 117.6 MB for 3 GEMM calls).

Dimensions match the MXFP4 tile format:
    A: [M=32, K=4096] BF16 → K-tiles [32, 32]
    B: [K=4096, N=2048] FP4 → MXFP4 tiles 2560 bytes each (32×128)
    C: [M=32, N=2048] BF16 → N-tiles [32, 128]
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

M_FIX, K_FIX, N_FIX = 32, 4096, 2048
TILE_M, TILE_K, TILE_N = 32, 32, 128  # k=32 matches MXFP4 tile rows

@iron.jit
def fst_fused_dequant_gemm(
    input0: In,   # A: BF16 activations [M, K]
    input1: In,   # B: FP4 weights as MXFP4 tiles [K/k * N/n, 2560] uint8
    output: Out,  # C: BF16 output [M, N]
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
):
    m, k, n = TILE_M, TILE_K, TILE_N
    assert M % m == 0 and K % k == 0 and N % n == 0

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    # ── Tensor types (L1 elements are flat — forward handles sub-tiling) ──
    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K_div_k * N_div_n * 2560,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    a_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(2560,), np.dtype[np.uint8]]
    c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    # ── Kernel ──────────────────────────────────────────────────────
    fused_kernel = ExternalFunction(
        "fst_fused_dequant_gemm_32x64x64",
        source_file="fst_fused_dequant_gemm_kernel.cc",
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=[aie_config.cxx_header_path()],
        object_file_name="fst_fused_dequant_gemm.o",
    )

    # ── ObjectFifos ─────────────────────────────────────────────────
    # A: L3→L2 (shim DMA) → L2→L1 forward (memtile DMA with sub-tiling)
    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // 4, 4 * k), (k // 8, 8), (4, k), (8, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # B: L3→L1 directly (no sub-tiling — kernel handles layout)
    inB = ObjectFifo(b_ty, name="inB", depth=2)

    # C: L1→L2 forward (un-sub-tiling) → L2→L3 (shim DMA)
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // 4, 4 * n), (4, 8), (n // 8, 4 * 8), (8, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, kernel):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            for i in range_(m * n):
                elem_out[i] = 0
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                kernel(elem_a, elem_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), inB.cons(), memC.prod(), fused_kernel],
        stack_size=0xD00,
    )

    # ── L3 DRAM tiling (4D TAPs to avoid 1023 BD limit) ────────────
    a_tap = TensorAccessPattern(
        tensor_dims=(M, K), offset=0,
        sizes=[N_div_n, K_div_k, m, k],
        strides=[0, k, K, 1],
    )
    b_tap = TensorAccessPattern(
        tensor_dims=(K_div_k * N_div_n, 2560), offset=0,
        sizes=[N_div_n, K_div_k, 40, 64],
        strides=[K_div_k * 2560, 2560, 64, 1],
    )
    c_tap = TensorAccessPattern(
        tensor_dims=(M, N), offset=0,
        sizes=[1, N_div_n, m, n],
        strides=[m * N, n, N, 1],
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(inA.prod(), A, tap=a_tap, task_group=tg)
        rt.fill(inB.prod(), B, tap=b_tap, task_group=tg)
        rt.drain(outC.cons(), C, tap=c_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def aot_compile():
    xclbin, insts = fst_fused_dequant_gemm.specialize(
        M=M_FIX, K=K_FIX, N=N_FIX
    ).compile()
    print(f"AOT compiled FUSED Dequant+GEMM {M_FIX}x{K_FIX}x{N_FIX} (NPU2)")
    print(f"  xclbin: {xclbin}")
    print(f"  insts:  {insts}")


if __name__ == "__main__":
    aot_compile()
