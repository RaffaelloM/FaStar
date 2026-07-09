#!/usr/bin/env python3
"""Compile combined MLA xclbin for DS4-XDNA (NPU2).

Combines qc, kvc, oa, ob shim-DMA kernels into a single xclbin
to reduce hw_context usage. Each kernel uses the same shim-DMA pattern.
"""
import os, sys, shutil, time
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
from pathlib import Path

set_current_device(NPU2())

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
INCLUDE_DIRS = [str(AIE_KERNEL_DIR.parent)]

# QC: h [M, 4096] @ wq_a [4096, 1024] -> qc [M, 1024]
@iron.jit
def qc_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    TILE_M, TILE_K, TILE_N = 8, 64, 64
    mm = ExternalFunction("fst_qk_gemm",
        source_file=str(PROJ_ROOT / "fst_qk_gemm.cc"),
        arg_types=[np.ndarray[(TILE_M*TILE_K,), np.dtype[el]],
                   np.ndarray[(TILE_K*TILE_N,), np.dtype[el]],
                   np.ndarray[(TILE_M*TILE_N,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])

    fifo_A = ObjectFifo(np.ndarray[(TILE_M*TILE_K,), np.dtype[el]], name="Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE_K*TILE_N,), np.dtype[el]], name="Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE_M*TILE_N,), np.dtype[el]], name="Cout", depth=2)

    N_div_n, K_div_k = N // TILE_N, K // TILE_K
    M_div_m = M // TILE_M

    def core(of_a, of_b, of_c, mm_k):
        for _ in range_(M_div_m * N_div_n) if M_div_m * N_div_n > 1 else range(1):
            ec = of_c.acquire(1)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                ea = of_a.acquire(1); eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1); of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), mm])

    at = TensorAccessPattern((M, K), 0, [M_div_m, N_div_n, TILE_M, TILE_K], [0, TILE_K, K, 1])
    bt = TensorAccessPattern((K, N), 0, [M_div_m, N_div_n, TILE_K, TILE_N], [0, TILE_N*K, N, 1])
    ct = TensorAccessPattern((M, N), 0, [1, M_div_m, N_div_n, TILE_M*TILE_N], [M*N, N, TILE_N*TILE_M, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(M, K), np.dtype[el]], np.ndarray[(K, N), np.dtype[el]], np.ndarray[(M, N), np.dtype[el]]) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()

# KVC: h [M, 4096] @ wkv [4096, 512] -> kv [M, 512]
@iron.jit
def kvc_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    TILE_M, TILE_K, TILE_N = 8, 64, 64
    mm = ExternalFunction("fst_sv_gemm",
        source_file=str(PROJ_ROOT / "fst_sv_gemm.cc"),
        arg_types=[np.ndarray[(TILE_M*TILE_K,), np.dtype[el]],
                   np.ndarray[(TILE_K*TILE_N,), np.dtype[el]],
                   np.ndarray[(TILE_M*TILE_N,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])

    fifo_A = ObjectFifo(np.ndarray[(TILE_M*TILE_K,), np.dtype[el]], name="Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE_K*TILE_N,), np.dtype[el]], name="Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE_M*TILE_N,), np.dtype[el]], name="Cout", depth=2)

    N_div_n, K_div_k = N // TILE_N, K // TILE_K
    M_div_m = M // TILE_M

    def core(of_a, of_b, of_c, mm_k):
        for _ in range_(M_div_m * N_div_n) if M_div_m * N_div_n > 1 else range(1):
            ec = of_c.acquire(1)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                ea = of_a.acquire(1); eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1); of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), mm])

    at = TensorAccessPattern((M, K), 0, [M_div_m, N_div_n, TILE_M, TILE_K], [0, TILE_K, K, 1])
    bt = TensorAccessPattern((K, N), 0, [M_div_m, N_div_n, TILE_K, TILE_N], [0, TILE_N*K, N, 1])
    ct = TensorAccessPattern((M, N), 0, [1, M_div_m, N_div_n, TILE_M*TILE_N], [M*N, N, TILE_N*TILE_M, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(M, K), np.dtype[el]], np.ndarray[(K, N), np.dtype[el]], np.ndarray[(M, N), np.dtype[el]]) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def compile_and_copy(name, jit_fn, specialize_args, xclbin_name, inst_name):
    t0 = time.perf_counter()
    result = jit_fn.specialize(**specialize_args)
    xclbin, insts = result.compile()
    tx = str(PROJ_ROOT / xclbin_name)
    ti = str(PROJ_ROOT / inst_name)
    shutil.copy2(xclbin, tx)
    shutil.copy2(insts, ti)
    dt = time.perf_counter() - t0
    print(f"  {name}: {os.path.getsize(tx)}B xclbin + {os.path.getsize(ti)}B insts ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    print("=== Compiling Combined MLA Kernels ===\n")

    # QC: M=8, K=4096, N=1024
    print("Compiling QC kernel...")
    compile_and_copy("qc", qc_op,
        {"M": 8, "K": 4096, "N": 1024, "el": bfloat16},
        "qc_shimdma.xclbin", "qc_shimdma_insts.bin")

    # KVC: M=8, K=4096, N=512
    print("Compiling KVC kernel...")
    compile_and_copy("kvc", kvc_op,
        {"M": 8, "K": 4096, "N": 512, "el": bfloat16},
        "kvc_shimdma.xclbin", "kvc_shimdma_insts.bin")

    print("\n=== Done ===")
