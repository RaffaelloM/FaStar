#!/usr/bin/env python3
"""Compile Hunyuan-3.0 (HY3) MXFP4 dequant kernel for XDNA2 (NPU2).

HY3 expert inter_dim = 1536 (vs DeepSeek's 2048).  The DS4 `fst_dequant_v4.xclbin`
is hard-baked to TOTAL_BLOCKS = 3*262144 = 786432 (reads 13,369,344 B); an HY3
expert block is only 10,027,008 B (589,824 blocks), so reusing the DS4 dequant
would OVERRUN the input BO.  This re-specializes the proven multicore dequant
(`fst_dequant_v4.py`) at HY3's block count.

Geometry (HY3):
  PROJ_BLOCKS = 4096*1536/32 = 196,608 blocks per projection (gate|up|down)
  TOTAL_BLOCKS = 3 * 196,608 = 589,824
  BLOCKS_PER_CORE = 589824 / 16 = 36,864
  CALLS_PER_CORE = 36,864 / 4 = 9,216 = 8 * 48 * 24   (TAP 4D decomposition)

The C++ block kernel `fst_dequant_v4_4096` is block-count-agnostic (4
blocks/call); only the IRON wrapper's total_blocks + TAP factorization change.

Outputs (in kernels/):
  fst_hy3_dequant.xclbin + fst_hy3_dequant_insts.bin
"""
import os, shutil

if "PEANO_INSTALL_DIR" not in os.environ:
    _site_pkgs = os.path.dirname(list(__import__("mlir_aie").__path__)[0])
    _cand = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_cand):
        os.environ["PEANO_INSTALL_DIR"] = _cand

import numpy as np
from ml_dtypes import bfloat16
import aie.compiler.aiecc.configure as _aie_cfg
if not os.path.isdir(getattr(_aie_cfg, "peano_install_dir", "")):
    _site_pkgs = os.path.dirname(list(__import__("mlir_aie").__path__)[0])
    _llvm = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_llvm):
        _aie_cfg.peano_install_dir = _llvm

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

BLOCK_BYTES = 17
BLOCK_ELEMS = 32
BLOCKS_PER_CALL = 4
DMA_IN_BYTES  = BLOCKS_PER_CALL * BLOCK_BYTES     # 68
DMA_OUT_ELEMS = BLOCKS_PER_CALL * BLOCK_ELEMS      # 128

PROJ_BLOCKS = 196608          # 4096*1536/32  (HY3 inter=1536)
TOTAL_BLOCKS = 3 * PROJ_BLOCKS   # 589,824

NUM_COLS, NUM_ROWS = 8, 2
NUM_CORES = NUM_COLS * NUM_ROWS            # 16
BLOCKS_PER_CORE = TOTAL_BLOCKS // NUM_CORES  # 36,864
CALLS_PER_CORE = BLOCKS_PER_CORE // BLOCKS_PER_CALL  # 9,216 = 8*48*24

# 4D BD decomposition: 9216 = 8 * 48 * 24  (all <= 255 repeat limit)
TAP0, TAP1, TAP2 = 8, 48, 24
assert TAP0 * TAP1 * TAP2 == CALLS_PER_CORE


@iron.jit
def fst_hy3_dequant(input0: In, output: Out, *, total_blocks: CompileTime[int]):
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
        Worker(make_core_fn(i), [f_in[i].cons(), f_out[i].prod(), dequant_kernel])
        for i in range(NUM_CORES)
    ]

    taps_in = [
        TensorAccessPattern(
            tensor_dims=(1, total_in_bytes),
            offset=i * calls * DMA_IN_BYTES,
            sizes=[TAP0, TAP1, TAP2, DMA_IN_BYTES],
            strides=[TAP1 * TAP2 * DMA_IN_BYTES, TAP2 * DMA_IN_BYTES, DMA_IN_BYTES, 1],
        )
        for i in range(NUM_CORES)
    ]
    taps_out = [
        TensorAccessPattern(
            tensor_dims=(1, total_out_elems),
            offset=i * calls * DMA_OUT_ELEMS,
            sizes=[TAP0, TAP1, TAP2, DMA_OUT_ELEMS],
            strides=[TAP1 * TAP2 * DMA_OUT_ELEMS, TAP2 * DMA_OUT_ELEMS, DMA_OUT_ELEMS, 1],
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


if __name__ == "__main__":
    xclbin, insts = fst_hy3_dequant.specialize(total_blocks=TOTAL_BLOCKS).compile()
    shutil.copy(xclbin, "fst_hy3_dequant.xclbin")
    shutil.copy(insts, "fst_hy3_dequant_insts.bin")
    print(f"HY3 dequant AOT compiled ({TOTAL_BLOCKS} blocks, {NUM_CORES} cores, "
          f"calls/core={CALLS_PER_CORE}={TAP0}x{TAP1}x{TAP2}) -> NPU2")
    print(f"  xclbin: {os.path.abspath('fst_hy3_dequant.xclbin')} ({os.path.getsize('fst_hy3_dequant.xclbin')}B)")
    print(f"  insts:  {os.path.abspath('fst_hy3_dequant_insts.bin')} ({os.path.getsize('fst_hy3_dequant_insts.bin')}B)")