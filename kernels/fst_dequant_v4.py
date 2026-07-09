#!/usr/bin/env python3
"""FaStar V4 FP4 Dequant — IRON/MLIR-AIE for XDNA2 (NPU2).

Multi-core dequant of DeepSeek V4 Flash dense FP4 expert blocks → BF16.
Dense block: 17 bytes (1 E8M0 scale + 16 packed FP4 nibbles) → 32 BF16.
Kernel processes 4 blocks per call (68 bytes, DMA-aligned to 4 bytes).
Uses 16 cores (8 cols × 2 rows), each processing 49,152 blocks.
4D TensorAccessPattern decomposes large transfers to stay under
the AIE DMA BD limit of 1023 elements/dim and 255 repeat count.
"""

import os

# Point Peano/llvm-aie to the pip-installed location before AIE imports.
if "PEANO_INSTALL_DIR" not in os.environ:
    _site_pkgs = os.path.dirname(
        list(__import__("mlir_aie").__path__)[0]
    )
    _candidate = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_candidate):
        os.environ["PEANO_INSTALL_DIR"] = _candidate

import numpy as np
from ml_dtypes import bfloat16

# Force-override the configure module's cached value if still wrong.
import aie.compiler.aiecc.configure as _aie_cfg
if not os.path.isdir(getattr(_aie_cfg, "peano_install_dir", "")):
    _site_pkgs = os.path.dirname(
        list(__import__("mlir_aie").__path__)[0]
    )
    _llvm = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_llvm):
        _aie_cfg.peano_install_dir = _llvm

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

# ── Dense block geometry ─────────────────────────────────────────────
BLOCK_BYTES = 17      # 1 e8m0 scale + 16 packed FP4 nibbles (on disk)
BLOCK_ELEMS = 32      # 32 elements per block

# DMA processes 4 blocks per kernel call → 68 bytes (multiple of 4)
BLOCKS_PER_CALL = 4
DMA_IN_BYTES  = BLOCKS_PER_CALL * BLOCK_BYTES    # 68 (4-byte aligned)
DMA_OUT_ELEMS = BLOCKS_PER_CALL * BLOCK_ELEMS    # 128 BF16

PROJ_BLOCKS = 262144        # 4096*2048/32 blocks per projection
TOTAL_BLOCKS = 3 * PROJ_BLOCKS  # 786,432 blocks total (gate+up+down)

# Multi-core distribution: 8 shim columns × 2 AIE rows = 16 cores
NUM_COLS = 8
NUM_ROWS = 2
NUM_CORES = NUM_COLS * NUM_ROWS
BLOCKS_PER_CORE = TOTAL_BLOCKS // NUM_CORES  # 49,152
CALLS_PER_CORE = BLOCKS_PER_CORE // BLOCKS_PER_CALL  # 12,288

CORE_IN_BYTES  = BLOCKS_PER_CORE * BLOCK_BYTES   # 835,584
CORE_OUT_ELEMS = BLOCKS_PER_CORE * BLOCK_ELEMS   # 1,572,864

# 4D BD decomposition per core, all sizes under 255 repeat limit.
# Input:  49152 blocks × 17 bytes  = 12288 groups × 68 bytes
#         12288 = 8 × 48 × 32     → sizes=[8, 48, 32, 68]
# Output: 49152 blocks × 32 BF16  = 12288 groups × 128 BF16
#         sizes=[8, 48, 32, 128]


@iron.jit
def fst_dequant_v4(
    input0: In,
    output: Out,
    *,
    total_blocks: CompileTime[int],
):
    in_dtype = np.uint8
    out_dtype = bfloat16

    calls = (total_blocks // NUM_CORES) // BLOCKS_PER_CALL
    total_in_bytes  = total_blocks * BLOCK_BYTES
    total_out_elems = total_blocks * BLOCK_ELEMS

    in_tensor_ty  = np.ndarray[(total_in_bytes,),  np.dtype[in_dtype]]
    out_tensor_ty = np.ndarray[(total_out_elems,), np.dtype[out_dtype]]
    in_call_ty    = np.ndarray[(DMA_IN_BYTES,), np.dtype[in_dtype]]
    out_call_ty   = np.ndarray[(DMA_OUT_ELEMS,), np.dtype[out_dtype]]

    dequant_kernel = ExternalFunction(
        "fst_dequant_v4_4096",
        source_file="fst_dequant_v4_vectorized_kernel.cc",
        arg_types=[in_call_ty, out_call_ty],
        include_dirs=[aie_config.cxx_header_path()],
        object_file_name="fst_dequant_v4_4096.o",
    )

    # One ObjectFifo pair per core
    f_in  = [ObjectFifo(in_call_ty,  name=f"in_{i}",  depth=2) for i in range(NUM_CORES)]
    f_out = [ObjectFifo(out_call_ty, name=f"out_{i}", depth=2) for i in range(NUM_CORES)]

    def make_core_fn(idx):
        def core_fn(of_in, of_out, kernel):
            for _ in range_(calls):
                ei = of_in.acquire(1)
                eo = of_out.acquire(1)
                kernel(ei, eo)
                of_in.release(1)
                of_out.release(1)
        return core_fn

    workers = [
        Worker(
            make_core_fn(i),
            [f_in[i].cons(), f_out[i].prod(), dequant_kernel],
        )
        for i in range(NUM_CORES)
    ]

    # TAPs: each core gets a contiguous slice of the data.
    # Input:  12288 calls × 68 bytes  = 8 × 48 × 32 × 68
    # Output: 12288 calls × 128 BF16  = 8 × 48 × 32 × 128
    taps_in = [
        TensorAccessPattern(
            tensor_dims=(1, total_in_bytes),
            offset=i * calls * DMA_IN_BYTES,
            sizes=[8, 48, 32, DMA_IN_BYTES],
            strides=[48 * 32 * DMA_IN_BYTES, 32 * DMA_IN_BYTES, DMA_IN_BYTES, 1],
        )
        for i in range(NUM_CORES)
    ]
    taps_out = [
        TensorAccessPattern(
            tensor_dims=(1, total_out_elems),
            offset=i * calls * DMA_OUT_ELEMS,
            sizes=[8, 48, 32, DMA_OUT_ELEMS],
            strides=[48 * 32 * DMA_OUT_ELEMS, 32 * DMA_OUT_ELEMS, DMA_OUT_ELEMS, 1],
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


def aot_compile():
    xclbin, insts = fst_dequant_v4.specialize(total_blocks=TOTAL_BLOCKS).compile()
    import os, shutil
    out_xcl = "fst_dequant_v4.xclbin"
    out_ins = "fst_dequant_v4_insts.bin"
    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)
    print(f"AOT compiled V4 dequant ({TOTAL_BLOCKS} blocks, {NUM_CORES} cores) — NPU2")
    print(f"  xclbin: {os.path.abspath(out_xcl)}")
    print(f"  insts:  {os.path.abspath(out_ins)}")


if __name__ == "__main__":
    aot_compile()
