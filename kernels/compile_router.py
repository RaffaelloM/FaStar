#!/usr/bin/env python3
"""Compile MoE Router xclbin for DS4-XDNA (NPU2).

Uses vectorized aie::mmul<4,8,8,bf16> GEMM kernel.
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

M_FIX, K_FIX, N_FIX, TOP_K_FIX = 8, 4096, 256, 8
TILE_M, TILE_K, TILE_N = 8, 64, 64


@iron.jit
def router_gemm_op(A: In, B: In, C: Out,
                   *, M: CompileTime[int], K: CompileTime[int],
                     N: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N

    mm = ExternalFunction("fst_router_gemm",
        source_file=str(PROJ_ROOT / "fst_router_gemm.cc"),
        arg_types=[np.ndarray[(m*k,), np.dtype[el]],
                   np.ndarray[(k*n,), np.dtype[el]],
                   np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])
    z = ExternalFunction("fst_router_zero",
        source_file=str(PROJ_ROOT / "fst_router_zero.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])

    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="Cout", depth=2)

    N_div_n, K_div_k = N // n, K // k

    def core(of_a, of_b, of_c, zero_k, mm_k):
        for _ in range_(N_div_n):
            ec = of_c.acquire(1)
            zero_k(ec)
            for _ in range_(K_div_k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm])

    at = TensorAccessPattern((M, K), 0, [N_div_n, K_div_k, m, k], [0, k, K, 1])
    bt = TensorAccessPattern((K, N), 0, [N_div_n, K_div_k, k, n], [n, k*N, N, 1])
    ct = TensorAccessPattern((M, N), 0, [1, N_div_n, m, n], [m*N, n, N, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(M, K), np.dtype[el]],
                     np.ndarray[(K, N), np.dtype[el]],
                     np.ndarray[(M, N), np.dtype[el]]) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def router_post_op(logits: In, ids: Out, wts: Out,
                   *, M: CompileTime[int], N: CompileTime[int],
                     TOP_K: CompileTime[int], el: CompileTime[type]):
    post_k = ExternalFunction("fst_router_post",
        source_file=str(PROJ_ROOT / "fst_router_post.cc"),
        arg_types=[np.ndarray[(M*N,), np.dtype[el]],
                   np.ndarray[(M*TOP_K,), np.dtype[np.int32]],
                   np.ndarray[(M*TOP_K,), np.dtype[np.float32]],
                   np.int32, np.int32, np.int32],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)])

    fifo_L = ObjectFifo(np.ndarray[(M*N,), np.dtype[el]], name="Lin", depth=2)
    fifo_I = ObjectFifo(np.ndarray[(M*TOP_K,), np.dtype[np.int32]], name="Iout", depth=2)
    fifo_O = ObjectFifo(np.ndarray[(M*TOP_K,), np.dtype[np.float32]], name="Wout", depth=2)

    def core(of_l, of_i, of_o, post_fn):
        for _ in range_(1):
            el_buf = of_l.acquire(1)
            ei = of_i.acquire(1)
            eo = of_o.acquire(1)
            post_fn(el_buf, ei, eo, M, N, TOP_K)
            of_l.release(1)
            of_i.release(1)
            of_o.release(1)

    w = Worker(core, [fifo_L.cons(), fifo_I.prod(), fifo_O.prod(), post_k])

    l_tap = TensorAccessPattern((M*N,), 0, [1, M*N], [0, 1])
    i_tap = TensorAccessPattern((M*TOP_K,), 0, [1, M*TOP_K], [0, 1])
    o_tap = TensorAccessPattern((M*TOP_K,), 0, [1, M*TOP_K], [0, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(M, N), np.dtype[el]],
                     np.ndarray[(M, TOP_K), np.dtype[np.int32]],
                     np.ndarray[(M, TOP_K), np.dtype[np.float32]]) as (lg, idx, wgt):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_L.prod(), lg, tap=l_tap, task_group=tg)
        rt.drain(fifo_I.cons(), idx, tap=i_tap, task_group=tg)
        rt.drain(fifo_O.cons(), wgt, tap=o_tap, wait=True, task_group=tg)
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
    print("=== Compiling Router GEMM xclbin ===\n")
    _compile_and_copy("fst_router", router_gemm_op,
                      {"M": M_FIX, "K": K_FIX, "N": N_FIX, "el": bfloat16})
    print("\n=== Compiling Router Post-GEMM xclbin ===\n")
    _compile_and_copy("fst_router_post", router_post_op,
                      {"M": M_FIX, "N": N_FIX, "TOP_K": TOP_K_FIX, "el": bfloat16})
    print("\n=== Done ===")
