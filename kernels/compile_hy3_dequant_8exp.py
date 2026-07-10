#!/usr/bin/env python3
"""Compile the HY3 8-expert MXFP4 dequant kernel for XDNA2 (NPU2).

This is the dispatch-collapse variant of `compile_hy3_dequant.py`: instead of
one dequant dispatch per routed expert (8 dispatches/layer), it dequants ALL 8
router-selected experts in a SINGLE dispatch.  The host packs the 8 experts'
MXFP4 blocks contiguously into one ~80 MB input BO (expert e at byte offset
e*10,027,008); because the C++ block kernel `fst_dequant_v4_4096` processes
blocks linearly and is block-count-agnostic, the BF16 output lands
expert-contiguously too (expert e at byte offset e*37,748,736 = 3*196608*32*2),
which is exactly the layout the fused GEMM-FFN kernels read via per-core TAP
offsets.

Geometry (8 experts):
  PROJ_BLOCKS   = 4096*1536/32 = 196,608 per projection (gate|up|down)
  EXPERT_BLOCKS = 3 * 196,608  = 589,824
  TOTAL_BLOCKS  = 8 * 589,824  = 4,718,592
  BLOCKS_PER_CORE = 4,718,592 / 16 = 294,912
  CALLS_PER_CORE  = 294,912 / 4    = 73,728 = 64 * 48 * 24   (TAP 4D decomp)

Input  BO = 4,718,592 * 17 = 80,216,064 B  (16 cores * 294,912 * 17, exact fit)
Output BO = 4,718,592 * 32 * 2 = 301,989,888 B (~302 MB, 8 expert slices)

The C++ kernel `fst_dequant_v4_4096` is unchanged (4 blocks/call); only the
IRON wrapper's total_blocks + TAP factorization change vs compile_hy3_dequant.py.

Outputs (in kernels/):
  fst_hy3_dequant_8exp.xclbin + fst_hy3_dequant_8exp_insts.bin
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
EXPERT_BLOCKS = 3 * PROJ_BLOCKS   # 589,824
N_EXPERTS = 8
TOTAL_BLOCKS = N_EXPERTS * EXPERT_BLOCKS   # 4,718,592

NUM_COLS, NUM_ROWS = 8, 2
NUM_CORES = NUM_COLS * NUM_ROWS            # 16
BLOCKS_PER_CORE = TOTAL_BLOCKS // NUM_CORES  # 294,912
CALLS_PER_CORE = BLOCKS_PER_CORE // BLOCKS_PER_CALL  # 73,728 = 64*48*24

# 4D BD decomposition: 73728 = 64 * 48 * 24  (all <= 255 repeat limit)
TAP0, TAP1, TAP2 = 64, 48, 24
assert TAP0 * TAP1 * TAP2 == CALLS_PER_CORE


@iron.jit
def fst_hy3_dequant_8exp(input0: In, output: Out, *, total_blocks: CompileTime[int]):
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
    xclbin, insts = fst_hy3_dequant_8exp.specialize(total_blocks=TOTAL_BLOCKS).compile()
    shutil.copy(xclbin, "fst_hy3_dequant_8exp.xclbin")
    shutil.copy(insts, "fst_hy3_dequant_8exp_insts.bin")
    in_b = TOTAL_BLOCKS * BLOCK_BYTES
    out_b = TOTAL_BLOCKS * BLOCK_ELEMS * 2
    assert NUM_CORES * BLOCKS_PER_CORE * BLOCK_BYTES == in_b, "input BO exact-fit check"
    print(f"HY3 8-expert dequant AOT compiled ({TOTAL_BLOCKS} blocks, {N_EXPERTS} experts, "
          f"{NUM_CORES} cores, calls/core={CALLS_PER_CORE}={TAP0}x{TAP1}x{TAP2}) -> NPU2")
    print(f"  xclbin: {os.path.abspath('fst_hy3_dequant_8exp.xclbin')} "
          f"({os.path.getsize('fst_hy3_dequant_8exp.xclbin')}B)")
    print(f"  insts:  {os.path.abspath('fst_hy3_dequant_8exp_insts.bin')} "
          f"({os.path.getsize('fst_hy3_dequant_8exp_insts.bin')}B)")
    print(f"  input BO: {in_b} B ({in_b/1e6:.1f} MB), output BO: {out_b} B ({out_b/1e6:.1f} MB)")