#!/usr/bin/env python3
"""Compile QK and SV attention xclbins for DS4-XDNA (NPU2).

Uses vectorized aie::mmul<4,8,8,bf16> GEMM kernels.
One source file per ExternalFunction to avoid duplicate symbols.
"""
import os, sys, shutil, time
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))

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

M_FIX = 8
K_FULL_DIM = 1088
D_FIX = 1024
S_MAX = 128
TILE_M, TILE_K, TILE_N = 8, 64, 64


@iron.jit
def qk_op(Q: In, K_mat: In, scores: Out,
          *, M: CompileTime[int], K_full: CompileTime[int],
            S: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N

    mm = ExternalFunction("fst_qk_gemm",
        source_file=str(PROJ_ROOT / "fst_qk_gemm.cc"),
        arg_types=[np.ndarray[(m*k,), np.dtype[el]],
                   np.ndarray[(k*n,), np.dtype[el]],
                   np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])
    z = ExternalFunction("fst_qk_zero",
        source_file=str(PROJ_ROOT / "fst_qk_zero.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])
    scale_k = ExternalFunction("fst_qk_scale",
        source_file=str(PROJ_ROOT / "fst_qk_scale.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]], np.int32],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])

    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="Cout", depth=2)

    S_div_n, K_div_k = S // n, K_full // k

    def core(of_a, of_b, of_c, zero_k, mm_k, sc_k):
        for _ in range_(S_div_n):
            ec = of_c.acquire(1)
            zero_k(ec)
            for _ in range_(K_div_k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            sc_k(ec, K_full)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm, scale_k])

    at = TensorAccessPattern((M, K_full), 0, [S_div_n, K_div_k, m, k], [0, k, K_full, 1])
    bt = TensorAccessPattern((K_full, S), 0, [S_div_n, K_div_k, k, n], [n, k*S, S, 1])
    ct = TensorAccessPattern((M, S), 0, [1, S_div_n, m, n], [m*S, n, S, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(M, K_full), np.dtype[el]],
                     np.ndarray[(K_full, S), np.dtype[el]],
                     np.ndarray[(M, S), np.dtype[el]]) as (q, k, sc):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), q, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), k, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), sc, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def sv_op(scores_in: In, V_in: In, out: Out,
          *, M: CompileTime[int], S: CompileTime[int],
            D: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N

    mm = ExternalFunction("fst_sv_gemm",
        source_file=str(PROJ_ROOT / "fst_sv_gemm.cc"),
        arg_types=[np.ndarray[(m*k,), np.dtype[el]],
                   np.ndarray[(k*n,), np.dtype[el]],
                   np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])
    z = ExternalFunction("fst_sv_zero",
        source_file=str(PROJ_ROOT / "fst_sv_zero.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])
    scale_k = ExternalFunction("fst_sv_scale",
        source_file=str(PROJ_ROOT / "fst_sv_scale.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]], np.int32],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])

    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="Cout", depth=2)

    D_div_n, S_div_k = D // n, S // k

    def core(of_a, of_b, of_c, zero_k, mm_k, sc_k):
        for _ in range_(D_div_n):
            ec = of_c.acquire(1)
            zero_k(ec)
            for _ in range_(S_div_k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            sc_k(ec, D)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm, scale_k])

    at = TensorAccessPattern((M, S), 0, [D_div_n, S_div_k, m, k], [0, k, S, 1])
    bt = TensorAccessPattern((S, D), 0, [D_div_n, S_div_k, k, n], [n, k*D, D, 1])
    ct = TensorAccessPattern((M, D), 0, [1, D_div_n, m, n], [m*D, n, D, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(M, S), np.dtype[el]],
                     np.ndarray[(S, D), np.dtype[el]],
                     np.ndarray[(M, D), np.dtype[el]]) as (sc, v, o):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), sc, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), v, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), o, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def _compile_and_copy(name, jit_fn, specialize_args):
    t0 = time.perf_counter()
    result = jit_fn.specialize(**specialize_args)
    xclbin, insts = result.compile()
    tx, ti = str(PROJ_ROOT / f"{name}.xclbin"), str(PROJ_ROOT / f"{name}_insts.bin")
    shutil.copy2(xclbin, tx)
    shutil.copy2(insts, ti)
    dt = time.perf_counter() - t0
    print(f"  {name}: {os.path.getsize(tx)}B xclbin + {os.path.getsize(ti)}B insts ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    print("=== Compiling QK attention xclbin ===\n")
    _compile_and_copy("fst_qk", qk_op, {"M": M_FIX, "K_full": K_FULL_DIM, "S": S_MAX, "el": bfloat16})
    print("\n=== Compiling SV attention xclbin ===\n")
    _compile_and_copy("fst_sv", sv_op, {"M": M_FIX, "S": S_MAX, "D": D_FIX, "el": bfloat16})
    print("\n=== Done ===")
