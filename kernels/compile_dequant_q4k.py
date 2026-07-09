#!/usr/bin/env python3
"""FaStar Q4_K Dequant — IRON/MLIR-AIE for XDNA2 (NPU2).

Dequantizes Q4_K expert blocks (144 bytes, 256 elements) → BF16.
Q4_K layout: fp16 d + fp16 dmin + 12B scales + 128B nibbles = 144 bytes.
Output: 256 BF16 values per block.

Expert projections:
  gate: [2048, 2048] = 16384 blocks → 16384 * 256 = 4M BF16 = 8 MB
  up:   [2048, 2048] = 16384 blocks → same
  down: [4096, 1024] = 16384 blocks → same
  Total: 49152 blocks, 49152 * 144 = 7,077,888 bytes input
         49152 * 256 * 2 = 25,165,824 bytes output (3 * 8 MB)
"""

import os

if "PEANO_INSTALL_DIR" not in os.environ:
    _site_pkgs = os.path.dirname(
        list(__import__("mlir_aie").__path__)[0]
    )
    _candidate = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_candidate):
        os.environ["PEANO_INSTALL_DIR"] = _candidate

import numpy as np
from ml_dtypes import bfloat16

import aie.compiler.aiecc.configure as _aie_cfg
if not os.path.isdir(getattr(_aie_cfg, "peano_install_dir", "")):
    _site_pkgs = os.path.dirname(
        list(__import__("mlir_aie").__path__)[0]
    )
    _llvm = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_llvm):
        _aie_cfg.peano_install_dir = _llvm

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config as aie_config

set_current_device(NPU2())

BLOCK_BYTES = 144      # Q4_K: 144 bytes per 256 elements
BLOCK_ELEMS = 256      # 256 BF16 output per block
BLOCKS_PER_CALL = 16   # 16 blocks per call (2304 bytes in, 8 KB BF16 out)
DMA_IN_BYTES = BLOCKS_PER_CALL * BLOCK_BYTES   # 2304
DMA_OUT_ELEMS = BLOCKS_PER_CALL * BLOCK_ELEMS  # 4096

# Expert: gate [in=4096,out=2048] + up [in=4096,out=2048] + down [in=2048,out=4096]
# gate: 4096*2048/256 = 32768 blocks, down: 2048*4096/256 = 32768 blocks
PROJ_BLOCKS = 32768
TOTAL_BLOCKS = 3 * PROJ_BLOCKS  # 98,304

NUM_COLS = 8
NUM_ROWS = 2
NUM_CORES = NUM_COLS * NUM_ROWS
BLOCKS_PER_CORE = TOTAL_BLOCKS // NUM_CORES  # 3,072
CALLS_PER_CORE = BLOCKS_PER_CORE // BLOCKS_PER_CALL  # 3,072

CORE_IN_BYTES = BLOCKS_PER_CORE * BLOCK_BYTES   # 442,368
CORE_OUT_ELEMS = BLOCKS_PER_CORE * BLOCK_ELEMS  # 786,432


@iron.jit
def fst_dequant_q4k(
    input0: In,
    output: Out,
    *,
    total_blocks: CompileTime[int],
):
    in_dtype = np.uint8
    out_dtype = bfloat16

    calls = (total_blocks // NUM_CORES) // BLOCKS_PER_CALL
    total_in_bytes = total_blocks * BLOCK_BYTES
    total_out_elems = total_blocks * BLOCK_ELEMS

    in_tensor_ty = np.ndarray[(total_in_bytes,), np.dtype[in_dtype]]
    out_tensor_ty = np.ndarray[(total_out_elems,), np.dtype[out_dtype]]
    in_call_ty = np.ndarray[(DMA_IN_BYTES,), np.dtype[in_dtype]]
    out_call_ty = np.ndarray[(DMA_OUT_ELEMS,), np.dtype[out_dtype]]

    dequant_kernel = ExternalFunction(
        "dequant_q4k_block",
        source_file="fst_dequant_q4k_kernel.cc",
        arg_types=[in_call_ty, out_call_ty],
        include_dirs=[aie_config.cxx_header_path()],
        object_file_name="fst_dequant_q4k_block.o",
    )

    f_in = [ObjectFifo(in_call_ty, name=f"in_{i}", depth=2) for i in range(NUM_CORES)]
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

    # TAPs: each core gets a contiguous slice
    # 384 calls × 2304 bytes = 884,736 bytes per core (BPC=16)
    # 384 = 12 × 16 × 2 → sizes=[12, 16, 2, DMA_IN_BYTES]
    taps_in = [
        TensorAccessPattern(
            tensor_dims=(1, total_in_bytes),
            offset=i * calls * DMA_IN_BYTES,
            sizes=[12, 16, 2, DMA_IN_BYTES],
            strides=[16 * 2 * DMA_IN_BYTES, 2 * DMA_IN_BYTES, DMA_IN_BYTES, 1],
        )
        for i in range(NUM_CORES)
    ]
    taps_out = [
        TensorAccessPattern(
            tensor_dims=(1, total_out_elems),
            offset=i * calls * DMA_OUT_ELEMS,
            sizes=[12, 16, 2, DMA_OUT_ELEMS],
            strides=[16 * 2 * DMA_OUT_ELEMS, 2 * DMA_OUT_ELEMS, DMA_OUT_ELEMS, 1],
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
    xclbin, insts = fst_dequant_q4k.specialize(total_blocks=TOTAL_BLOCKS).compile()
    print(f"AOT compiled Q4_K dequant ({TOTAL_BLOCKS} blocks, {NUM_CORES} cores) — NPU2")
    print(f"  xclbin: {xclbin}")
    print(f"  insts:  {insts}")


if __name__ == "__main__":
    aot_compile()