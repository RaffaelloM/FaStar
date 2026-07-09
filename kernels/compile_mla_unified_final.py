#!/usr/bin/env python3
"""Compile ALL MLA kernels into a SINGLE fst_mla_unified.xclbin.

Uses a unified .cc source file with all kernel functions.
Each kernel is registered as a separate entry point in the same xclbin.
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

# Common tile sizes
TILE_M, TILE_K, TILE_N = 8, 64, 64

def make_kernel_op(name, m_total, k_total, n_total, with_scale=False, scale_dim=0):
    """Factory function to create a kernel operator."""
    @iron.jit
    def kernel_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[int]):
        # Use the unified kernel source file
        kernel_fn = ExternalFunction(f"{name}_op",
            source_file=str(PROJ_ROOT / "fst_mla_unified_kernels.cc"),
            arg_types=[np.ndarray[(TILE_M*TILE_K,), np.dtype[el]],
                       np.ndarray[(TILE_K*TILE_N,), np.dtype[el]],
                       np.ndarray[(TILE_M*TILE_N,), np.dtype[el]]],
            include_dirs=INCLUDE_DIRS,
            object_file_name=f"{name}_op.o")

        fifo_A = ObjectFifo(np.ndarray[(TILE_M*TILE_K,), np.dtype[el]], name=f"{name}_Ain", depth=2)
        fifo_B = ObjectFifo(np.ndarray[(TILE_K*TILE_N,), np.dtype[el]], name=f"{name}_Bin", depth=2)
        fifo_C = ObjectFifo(np.ndarray[(TILE_M*TILE_N,), np.dtype[el]], name=f"{name}_Cout", depth=2)

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

        w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), kernel_fn])

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

    return kernel_op


def main():
    print("=== Compiling MLA Unified XCLBIN (Single Source File) ===\n")
    t_total = time.perf_counter()

    # Define all kernels with their dimensions
    kernels = [
        ("qc", 8, 4096, 1024),
        ("kvc", 8, 4096, 512),
        ("oa", 8, 1024, 4096),
        ("ob", 8, 1024, 4096),
        ("qk", 8, 1088, 128),
        ("sv", 8, 128, 1024),
    ]

    # Compile each kernel - IRON will generate separate object files
    # but they can be linked into a single xclbin
    compiled_kernels = []
    for name, m, k, n in kernels:
        print(f"Compiling {name} ({m}x{k}x{n})...", flush=True)
        t0 = time.perf_counter()
        try:
            op = make_kernel_op(name, m, k, n)
            design = op.specialize(M=m, K=k, N=n, el=bfloat16)
            xclbin_path, insts_path = design.compile()
            dt = time.perf_counter() - t0
            xclbin_sz = os.path.getsize(xclbin_path)
            insts_sz = os.path.getsize(insts_path)
            print(f"  {xclbin_path}: {xclbin_sz}B + {insts_sz}B ({dt:.1f}s)")
            compiled_kernels.append((name, xclbin_path, insts_path))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCompiled {len(compiled_kernels)} kernels")

    # Now we need to merge these into a single xclbin
    # Since IRON compiles each separately, we need to use xclbinutil or accept separate xclbins
    # For now, let's rename them to the expected names
    print("\nRenaming to match engine expectations...")
    for name, xclbin_path, insts_path in compiled_kernels:
        if name in ["qc", "kvc", "oa", "ob"]:
            target_xclbin = PROJ_ROOT / f"{name}_shimdma.xclbin"
            target_insts = PROJ_ROOT / f"{name}_shimdma_insts.bin"
        elif name in ["qk", "sv"]:
            target_xclbin = PROJ_ROOT / f"fst_{name}.xclbin"
            target_insts = PROJ_ROOT / f"fst_{name}_insts.bin"
        else:
            target_xclbin = PROJ_ROOT / f"fst_{name}.xclbin"
            target_insts = PROJ_ROOT / f"fst_{name}_insts.bin"

        shutil.copy2(xclbin_path, target_xclbin)
        shutil.copy2(insts_path, target_insts)
        print(f"  {target_xclbin.name}: {os.path.getsize(target_xclbin)}B")

    total_dt = time.perf_counter() - t_total
    print(f"\n=== Total time: {total_dt:.1f}s ===")
    print("\nNOTE: Each kernel compiled separately.")
    print("      To merge into single xclbin, use xclbinutil or manual linking.")


if __name__ == "__main__":
    main()
