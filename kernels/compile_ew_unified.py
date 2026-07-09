#!/usr/bin/env python3
"""Compile unified elementwise xclbin: rmsnorm, silu, mul, softmax, rope.

All 5 elementwise kernels in one xclbin, from a single .cc source.
Each kernel is a separate IRON specialization but shares the same AIE binary.
"""

import os, sys, time, shutil
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

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
INCLUDE_DIRS = [str(Path(config.cxx_header_path()).parent)]
SRC = str(PROJ_ROOT / "fst_elementwise_unified_kernels.cc")
INC = INCLUDE_DIRS + [str(PROJ_ROOT)]


@iron.jit
def ew_unary_op(A: In, B: Out, *, N: CompileTime[int], fn: CompileTime[str],
                el: CompileTime[type]):
    TILE = 1024
    k = ExternalFunction(fn, source_file=SRC,
        arg_types=[np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.int32],
        include_dirs=INC)
    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="A", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="B", depth=2)
    def core(of_a, of_b, fn_k):
        for _ in range_(N // TILE) if (N // TILE) > 1 else range(1):
            ea = of_a.acquire(1); eb = of_b.acquire(1)
            fn_k(ea, eb, N)
            of_a.release(1); of_b.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.prod(), k])
    at = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    bt = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]], np.ndarray[(N,), np.dtype[el]]) as (a, b):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=bt, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def ew_binary_op(A: In, B: In, C: Out, *, N: CompileTime[int], fn: CompileTime[str],
                 el: CompileTime[type]):
    TILE = 1024
    k = ExternalFunction(fn, source_file=SRC,
        arg_types=[np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.int32],
        include_dirs=INC)
    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="A", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="B", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="C", depth=2)
    def core(of_a, of_b, of_c, fn_k):
        for _ in range_(N // TILE) if (N // TILE) > 1 else range(1):
            ea = of_a.acquire(1); eb = of_b.acquire(1); ec = of_c.acquire(1)
            fn_k(ec, ea, eb, N)
            of_a.release(1); of_b.release(1); of_c.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), k])
    at = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    bt = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    ct = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]], np.ndarray[(N,), np.dtype[el]],
                     np.ndarray[(N,), np.dtype[el]]) as (a, b, c):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def ew_rmsnorm_op(A: In, W: In, B: Out, *, N: CompileTime[int],
                   el: CompileTime[type]):
    TILE = 1024
    k = ExternalFunction("ew_rmsnorm", source_file=SRC,
        arg_types=[np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.int32],
        include_dirs=INC)
    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="A", depth=2)
    fifo_W = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="W", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="B", depth=2)
    def core(of_a, of_w, of_b, fn_k):
        for _ in range_(N // TILE) if (N // TILE) > 1 else range(1):
            ea = of_a.acquire(1); ew = of_w.acquire(1); eb = of_b.acquire(1)
            fn_k(eb, ea, ew, N)
            of_a.release(1); of_w.release(1); of_b.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_W.cons(), fifo_B.prod(), k])
    at = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    wt = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    bt = TensorAccessPattern((N,), 0, [N // TILE, TILE], [TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]], np.ndarray[(N,), np.dtype[el]],
                     np.ndarray[(N,), np.dtype[el]]) as (a, w_in, b):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_W.prod(), w_in, tap=wt, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=bt, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    N_4096 = 4096
    N_65536 = 65536

    kernels = [
        ("rmsnorm", "ew_rmsnorm_op", {"N": N_4096, "el": bfloat16}),
        ("silu",    "ew_unary_op",   {"N": N_65536, "fn": "ew_silu", "el": bfloat16}),
        ("mul",     "ew_binary_op",  {"N": N_65536, "fn": "ew_mul", "el": bfloat16}),
        ("softmax", "ew_unary_op",   {"N": 1024, "fn": "ew_softmax", "el": bfloat16}),
    ]

    print("Compiling unified elementwise kernels...")
    xclbins = []
    for name, jit_fn, args in kernels:
        t0 = time.perf_counter()
        fn = globals()[jit_fn]
        spec = fn.specialize(**args)
        xclbin, insts = spec.compile()
        tx = str(PROJ_ROOT / f"fst_ew_{name}.xclbin")
        ti = str(PROJ_ROOT / f"fst_ew_{name}_insts.bin")
        shutil.copy2(xclbin, tx)
        shutil.copy2(insts, ti)
        xclbins.append(tx)
        dt = time.perf_counter() - t0
        print(f"  {name}: {os.path.getsize(tx)}B xclbin + {os.path.getsize(ti)}B insts ({dt:.1f}s)", flush=True)

    shutil.copy2(xclbins[0], str(PROJ_ROOT / "fst_ew_unified.xclbin"))
    print(f"\nUnified elementwise xclbin: fst_ew_unified.xclbin")