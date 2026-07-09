#!/usr/bin/env python3
"""Compile ALL MLA kernels into a SINGLE fst_mla_unified.xclbin.

Uses IRON's native multi-kernel compilation - define all @iron.jit functions
and compile them as a single module.
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
INCLUDE_DIRS = [str(AIE_KERNEL_DIR.parent), str(PROJ_ROOT)]

# ============================================================================
# QC: h [M, 4096] @ wq_a [4096, 1024] -> qc [M, 1024]
# ============================================================================
@iron.jit
def qc_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    TILE_M, TILE_K, TILE_N = 8, 64, 64
    mm = ExternalFunction("fst_qk_gemm",
        source_file=str(PROJ_ROOT / "fst_qk_gemm.cc"),
        arg_types=[np.ndarray[(TILE_M*TILE_K,), np.dtype[el]],
                   np.ndarray[(TILE_K*TILE_N,), np.dtype[el]],
                   np.ndarray[(TILE_M*TILE_N,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS)

    fifo_A = ObjectFifo(np.ndarray[(TILE_M*TILE_K,), np.dtype[el]], name="qc_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE_K*TILE_N,), np.dtype[el]], name="qc_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE_M*TILE_N,), np.dtype[el]], name="qc_Cout", depth=2)

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


# ============================================================================
# KVC: h [M, 4096] @ wkv [4096, 512] -> kv [M, 512]
# ============================================================================
@iron.jit
def kvc_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    TILE_M, TILE_K, TILE_N = 8, 64, 64
    mm = ExternalFunction("fst_sv_gemm",
        source_file=str(PROJ_ROOT / "fst_sv_gemm.cc"),
        arg_types=[np.ndarray[(TILE_M*TILE_K,), np.dtype[el]],
                   np.ndarray[(TILE_K*TILE_N,), np.dtype[el]],
                   np.ndarray[(TILE_M*TILE_N,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS)

    fifo_A = ObjectFifo(np.ndarray[(TILE_M*TILE_K,), np.dtype[el]], name="kvc_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE_K*TILE_N,), np.dtype[el]], name="kvc_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE_M*TILE_N,), np.dtype[el]], name="kvc_Cout", depth=2)

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


# ============================================================================
# OA: attn [M, 1024] @ wo_a [1024, 4096] -> out [M, 4096]
# ============================================================================
@iron.jit
def oa_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    TILE_M, TILE_K, TILE_N = 8, 64, 64
    mm = ExternalFunction("fst_qk_gemm",
        source_file=str(PROJ_ROOT / "fst_qk_gemm.cc"),
        arg_types=[np.ndarray[(TILE_M*TILE_K,), np.dtype[el]],
                   np.ndarray[(TILE_K*TILE_N,), np.dtype[el]],
                   np.ndarray[(TILE_M*TILE_N,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS)

    fifo_A = ObjectFifo(np.ndarray[(TILE_M*TILE_K,), np.dtype[el]], name="oa_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE_K*TILE_N,), np.dtype[el]], name="oa_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE_M*TILE_N,), np.dtype[el]], name="oa_Cout", depth=2)

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


# ============================================================================
# OB: attn [M, 1024] @ wo_b [1024, 4096] -> out [M, 4096]
# ============================================================================
@iron.jit
def ob_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[int], el: CompileTime[type]):
    TILE_M, TILE_K, TILE_N = 8, 64, 64
    mm = ExternalFunction("fst_qk_gemm",
        source_file=str(PROJ_ROOT / "fst_qk_gemm.cc"),
        arg_types=[np.ndarray[(TILE_M*TILE_K,), np.dtype[el]],
                   np.ndarray[(TILE_K*TILE_N,), np.dtype[el]],
                   np.ndarray[(TILE_M*TILE_N,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS)

    fifo_A = ObjectFifo(np.ndarray[(TILE_M*TILE_K,), np.dtype[el]], name="ob_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE_K*TILE_N,), np.dtype[el]], name="ob_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE_M*TILE_N,), np.dtype[el]], name="ob_Cout", depth=2)

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


# ============================================================================
# Main: compile all kernels together
# ============================================================================
if __name__ == "__main__":
    print("=== Compiling MLA Unified XCLBIN (Native IRON Multi-Kernel) ===\n")
    t_total = time.perf_counter()

    # Specialize all kernels
    print("Specializing kernels...", flush=True)
    qc_program = qc_op.specialize(M=8, K=4096, N=1024, el=bfloat16)
    print("  qc_op")
    kvc_program = kvc_op.specialize(M=8, K=4096, N=512, el=bfloat16)
    print("  kvc_op")
    oa_program = oa_op.specialize(M=8, K=1024, N=4096, el=bfloat16)
    print("  oa_op")
    ob_program = ob_op.specialize(M=8, K=1024, N=4096, el=bfloat16)
    print("  ob_op")

    # Compile all programs together
    print("\nCompiling to xclbin (this may take a minute)...", flush=True)
    t0 = time.perf_counter()

    try:
        # Try IRON's multi-program compile if available
        results = [qc_program, kvc_program, oa_program, ob_program]
        xclbin_paths = []
        insts_paths = []

        for i, prog in enumerate(results):
            xclbin, insts = prog.compile()
            name = ["qc", "kvc", "oa", "ob"][i]
            dst_xclbin = PROJ_ROOT / f"{name}.xclbin"
            dst_insts = PROJ_ROOT / f"{name}_insts.bin"
            shutil.copy2(xclbin, dst_xclbin)
            shutil.copy2(insts, dst_insts)
            xclbin_paths.append(dst_xclbin)
            insts_paths.append(dst_insts)
            print(f"  {name}: {dst_xclbin.stat().st_size}B + {dst_insts.stat().st_size}B")

        print(f"\nCompiled {len(results)} kernels in {time.perf_counter() - t0:.1f}s")
        print("\nNOTE: IRON compiled each kernel separately.")
        print("      For true unified xclbin, need different approach.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    total_dt = time.perf_counter() - t_total
    print(f"\n=== Total time: {total_dt:.1f}s ===")
