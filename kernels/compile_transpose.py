#!/usr/bin/env python3
"""FaStar NPU bf16 [N,K] -> [K,N] transpose — IRON/MLIR-AIE for XDNA2 (NPU2).

Transposes the MXFP4 dequant output (B[N,K] row-major) into B[K,N] row-major
so the vectorized FFN GEMM can use the IRON-default b_row_maj path.  Pure NPU
data movement (no mmul, no CPU).

Tile: 4 N-rows x 8 K-cols (32 bf16 = one v32).  Strided DMA TAPs gather a
[4,8] block from src[N,K] and scatter the transposed [8,4] block to dst[K,N].
16 cores, each handling an N-slab.

Two specializations (the dequant emits gate/up as [2048,4096], down as
[4096,2048]):
  gu : N=2048, K=4096  (gate + up — engine calls it twice, sub-BO per proj)
  dn : N=4096, K=2048  (down)

The kernel is offset-agnostic: the engine passes a sub-BO view starting at the
projection's offset, so the TAP offset 0 = projection start.
"""

import os, shutil
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config
from pathlib import Path

set_current_device(NPU2())
PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Tile geometry (must match fst_bf16_transpose_kernel.cc: 4x8 = 32 bf16)
TS_N = 4    # N-rows per tile
TS_K = 8    # K-cols per tile
TILE_ELEMS = TS_N * TS_K  # 32

# Multi-core: 8 shim cols x 2 AIE rows = 16 cores, N-slab distribution
NUM_COLS = 8
NUM_ROWS = 2
NUM_CORES = NUM_COLS * NUM_ROWS


@iron.jit
def fst_transpose(
    input0: In, output: Out, *,
    N: CompileTime[int], K: CompileTime[int],
):
    in_dtype = bfloat16
    assert N % TS_N == 0 and K % TS_K == 0
    N_div_n = N // TS_N
    K_div_k = K // TS_K
    assert N_div_n % NUM_CORES == 0, "N must divide evenly over 16 cores"
    n_per_core = N_div_n // NUM_CORES     # N-tiles per core
    tiles_per_core = n_per_core * K_div_k

    in_tensor_ty  = np.ndarray[(N * K,), np.dtype[in_dtype]]
    out_tensor_ty = np.ndarray[(K * N,), np.dtype[in_dtype]]
    tile_ty       = np.ndarray[(TILE_ELEMS,), np.dtype[in_dtype]]

    kernel = iron.kernel.ExternalFunction(
        "fst_transpose_4x8",
        source_file="fst_bf16_transpose_kernel.cc",
        arg_types=[tile_ty, tile_ty],
        include_dirs=[aie_config.cxx_header_path()],
        object_file_name="fst_transpose_4x8.o",
    )

    f_in  = [ObjectFifo(tile_ty, name=f"in_{i}",  depth=2) for i in range(NUM_CORES)]
    f_out = [ObjectFifo(tile_ty, name=f"out_{i}", depth=2) for i in range(NUM_CORES)]

    def make_core_fn(idx):
        def core_fn(of_in, of_out, kern):
            for _ in range_(tiles_per_core):
                ei = of_in.acquire(1)
                eo = of_out.acquire(1)
                kern(ei, eo)
                of_in.release(1)
                of_out.release(1)
        return core_fn

    workers = [
        Worker(make_core_fn(i),
               [f_in[i].cons(), f_out[i].prod(), kernel])
        for i in range(NUM_CORES)
    ]

    # Per-core N-slab: core i handles N-tiles [i*n_per_core, (i+1)*n_per_core).
    # src [N,K] row-major: N-tile stride = TS_N*K, K-tile stride = TS_K,
    #   within-tile row stride = K, col stride = 1.
    # 4D TAP (AIE DMA BD allows at most 4 dims); K_div_k fits in one dim
    # (<=1023 elements/dim).  n_per_core is the outer (repeat) dim.
    assert K_div_k <= 1023, "K_div_k exceeds single BD-dim size limit"
    sizes_in = [n_per_core, K_div_k, TS_N, TS_K]
    strides_in = [TS_N * K, TS_K, K, 1]

    # dst [K,N] row-major: kernel writes transposed tile as [TS_K, TS_N]
    # row-major (TS_K K-rows of TS_N N-cols).  Drain scatters with
    #   within-tile row stride = N, col stride = 1
    #   K-tile stride = TS_K*N, N-tile stride = TS_N (consecutive N-cols in dst).
    sizes_out = [n_per_core, K_div_k, TS_K, TS_N]
    strides_out = [TS_N, TS_K * N, N, 1]

    taps_in = [
        TensorAccessPattern(
            tensor_dims=(1, N * K),
            offset=i * n_per_core * TS_N * K,
            sizes=sizes_in,
            strides=strides_in,
        )
        for i in range(NUM_CORES)
    ]
    taps_out = [
        TensorAccessPattern(
            tensor_dims=(1, K * N),
            offset=i * n_per_core * TS_N,
            sizes=sizes_out,
            strides=strides_out,
        )
        for i in range(NUM_CORES)
    ]

    rt = Runtime()
    with rt.sequence(in_tensor_ty, out_tensor_ty) as (A, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(NUM_CORES):
            rt.fill(f_in[i].prod(), A, tap=taps_in[i], task_group=tg)
        for i in range(NUM_CORES):
            rt.drain(f_out[i].cons(), C, tap=taps_out[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def _compile_one(N, K, tag):
    xclbin, insts = fst_transpose.specialize(N=N, K=K).compile()
    out_xcl = f"fst_transpose_{tag}.xclbin"
    out_ins = f"fst_transpose_{tag}_insts.bin"
    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)
    print(f"AOT compiled transpose {tag} N={N} K={K} ({NUM_CORES} cores)")
    print(f"  xclbin: {os.path.abspath(out_xcl)} ({os.path.getsize(out_xcl)}B)")
    print(f"  insts:  {os.path.abspath(out_ins)} ({os.path.getsize(out_ins)}B)")


def aot_compile() -> None:
    # gate/up: [N=2048, K=4096] -> [4096, 2048]
    _compile_one(2048, 4096, "gu")
    # down:    [N=4096, K=2048] -> [2048, 4096]
    _compile_one(4096, 2048, "dn")


if __name__ == "__main__":
    aot_compile()