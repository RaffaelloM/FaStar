#!/usr/bin/env python3
"""FaStar MoE expert GEMM -- custom IRON/MLIR-AIE design for XDNA2 (NPU2).

Single-core BF16 matmul C = A @ B for one MoE expert FFN projection.

DeepSeek V4 Flash real dimensions (verified from safetensors):
    w1 (gate) / w3 (up):  [out=2048, in=4096] HF -> [in, out] = [4096, 2048]
    w2 (down):            [out=4096, in=2048] HF -> [in, out] = [2048, 4096]

This GEMM is compiled for the gate/up projection shape:
    A : [M, K] activations   (M = 32   , K = 4096 = hidden_size)
    B : [K, N] weights       (K = 4096 , N = 2048 = moe_intermediate_size)
    C : [M, N] output

The down projection (K=2048, N=4096) needs a separate xclbin compiled with
those dimensions; the same design function handles both via CompileTime args.

Dequantized BF16 weights are produced externally by the FaStar dequant
path; this operator only does the GEMM on BF16.

Why Shim DMA only (no mem-tile / 0xC400 DMA)
-------------------------------------------
The reverse-engineering of the FastFlowLM Q4NX xclbin showed the memory-tile
DMA (0xC400) path cannot apply DDR_PATCH correctly for our expert paging
scheme. The proven FaStar dequant xclbin uses Shim DMA (0x1D000) exclusively,
via rt.fill (MM2S) / rt.drain (S2MM) on direct Shim<->core ObjectFifos. This
design replicates that exact data-flow contract for GEMM.

Why the scalar matmul kernel (for now)
--------------------------------------
The vectorized aie::mmul kernel expects A/B/C pre-tiled into r x s / s x t /
r x t intrinsic sub-tiles. Producing that layout requires an L2->L1
dims_to_stream transform, which is a mem-tile (0xC400) DMA operation -- the
dead-end path. The scalar `matmul_scalar_bf16_bf16` kernel (from the IRON
kernel library, aie_kernels/aie2p/mm.cc) reads plain row-major A (m,k),
B (k,n), C (m,n), so the Shim DMA can deliver row-major tiles straight into
L1 with no layout transform and no mem-tile hop.

This trades peak MAC throughput for a correct, single-core, Shim-DMA-only
baseline. Multi-core + a custom vectorized kernel that does the row-major ->
intrinsic rearrangement inside L1 is a Phase 7 optimization, measured after
the pipeline end-to-end runs (AGENTS.md Rule 7).

Flow (mirrors programming_examples/getting_started/03_matrix_multiplication_
single_core, with the L2/L1 forward fifos collapsed away):
    @iron.jit -> ObjectFifos (direct Shim<->core) -> Worker(core_fn) ->
    Runtime: rt.fill(A), rt.fill(B), rt.drain(C) over task_groups ->
    Program(NPU2, rt).resolve_program() -> .specialize().compile() -> xclbin
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.utils import set_current_device

# DeepSeek V4 Flash gate/up projection: K=hidden=4096, N=intermediate=2048.
M_FIX = 32  # activation bucket capacity
K_FIX = 4096  # hidden_size
N_FIX = 2048  # moe_intermediate_size

# L1 tile size. m covers the whole M dimension in one row of tiles; k, n are
# chosen to fit a single core's L1 alongside fifo depth 2:
#   A tile 32x64 = 4 KB, B tile 64x64 = 8 KB, C tile 32x64 = 4 KB  (bf16).
TILE_M = 32
TILE_K = 64
TILE_N = 64

# Pin the target to NPU2 (AIE2P, Strix Point) so kernels.mm() resolves to
# aie_kernels/aie2p/mm.cc and _detect_arch() returns "aie2p". Without this,
# get_current_device() is None (no XRT on this host) and the arch falls back
# to "aie2", selecting the wrong kernel source.
set_current_device(NPU2())


@iron.jit
def fst_expert_gemm(
    input0: In,
    input1: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    element_type: CompileTime[type],
):
    m, k, n = TILE_M, TILE_K, TILE_N

    assert M % m == 0, f"M ({M}) must be a multiple of tile m ({m})"
    assert K % k == 0, f"K ({K}) must be a multiple of tile k ({k})"
    assert N % n == 0, f"N ({N}) must be a multiple of tile n ({n})"

    # Scalar matmul + its zero companion. Both symbols live in the same .o
    # (mm.cc emits matmul_scalar_* and zero_scalar_*), so matmul_kernel.zero
    # reuses the already-compiled object -- no second compile, no link clash.
    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=element_type,
        output_dtype=element_type,
        vectorized=False,
    )
    zero_kernel = matmul_kernel.zero

    A_ty = np.ndarray[(M, K), np.dtype[element_type]]
    B_ty = np.ndarray[(K, N), np.dtype[element_type]]
    C_ty = np.ndarray[(M, N), np.dtype[element_type]]
    # Flat 1-D tile types match the kernel's ExternalFunction signature
    # (kernels.mm declares a/b/c as flat m*k / k*n / m*n arrays).
    a_tile_ty = np.ndarray[(m * k,), np.dtype[element_type]]
    b_tile_ty = np.ndarray[(k * n,), np.dtype[element_type]]
    c_tile_ty = np.ndarray[(m * n,), np.dtype[element_type]]

    # Direct Shim<->core ObjectFifos: no .forward()/.split()/.join(), hence no
    # mem-tile (0xC400) DMA in the graph. depth=2 lets the Shim DMA prefetch
    # the next tile while the core computes -- the same depth the dequant uses
    # for tiles that fit comfortably in L1.
    fifo_A = ObjectFifo(a_tile_ty, name="A_L3L1", depth=2)
    fifo_B = ObjectFifo(b_tile_ty, name="B_L3L1", depth=2)
    fifo_C = ObjectFifo(c_tile_ty, name="C_L1L3", depth=2)

    # Worker core loop. range_() (not range) is required on the NPU side.
    # For each output tile: zero the accumulator, then reduce over K//k tiles.
    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                matmul(elem_a, elem_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), zero_kernel, matmul_kernel],
    )

    # Custom 4D TensorAccessPatterns that keep all BD dimension sizes under
    # the AIE DMA hardware limit of 1023. The TensorTiler2D.group_tiler merges
    # K-tiles and K-rows into a single BD dimension of size K (4096), which
    # exceeds 1023. By explicitly decomposing into (N-tiles, K-tiles, rows,
    # cols) with separate strides, every dimension stays <= 64.
    #
    # A [M, K]: repeat for each N-tile column (stride=0), iterate K-tiles,
    # then M-rows and K-cols within each tile.
    a_tap = TensorAccessPattern(
        tensor_dims=(M, K),
        offset=0,
        sizes=[N // n, K // k, m, k],
        strides=[0, k, K, 1],
    )
    # B [K, N]: col-major over N-tiles, then K-tiles, then K-rows and N-cols.
    b_tap = TensorAccessPattern(
        tensor_dims=(K, N),
        offset=0,
        sizes=[N // n, K // k, k, n],
        strides=[n, k * N, N, 1],
    )
    # C [M, N]: row-major over N-tiles, then M-rows and N-cols.
    c_tap = TensorAccessPattern(
        tensor_dims=(M, N),
        offset=0,
        sizes=[1, N // n, m, n],
        strides=[m * N, n, N, 1],
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        # Single task_group: M//m = 1, so all output tiles are in one row.
        # A is repeated N//n times (stride=0 in dim0).
        # B covers the full matrix in one BD (all sizes <= 64 < 1023).
        # C drains all output tiles in one BD.
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), A, tap=a_tap, task_group=tg)
        rt.fill(fifo_B.prod(), B, tap=b_tap, task_group=tg)
        rt.drain(
            fifo_C.cons(),
            C,
            tap=c_tap,
            task_group=tg,
            wait=True,
        )
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def aot_compile() -> None:
    """Pre-compile the design for the fixed expert shapes and print artifacts."""
    xclbin, insts = fst_expert_gemm.specialize(
        M=M_FIX, K=K_FIX, N=N_FIX, element_type=bfloat16
    ).compile()
    print(f"AOT compiled fst_expert_gemm {M_FIX}x{K_FIX}x{N_FIX} bf16 (NPU2)")
    print(f"  xclbin: {xclbin}")
    print(f"  insts:  {insts}")


if __name__ == "__main__":
    aot_compile()
