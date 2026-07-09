#!/usr/bin/env python3
"""Compile NPU element-wise operation xclbins for DS4-XDNA.

Compiles 4 kernel types using the IRON @iron.jit framework:
  1. rmsnorm  — RMSNorm (unweighted, x * inv_rms, weight applied on host)
  2. silu     — SiLU activation (x * sigmoid(x))
  3. softmax  — row-wise softmax (causal mask on host before NPU call)
  4. rope     — Rotary Position Embedding (interleaved cos/sin)

Each produces a .xclbin + _insts.bin pair loaded by FSTEngine.
"""
import os, sys, shutil, time

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import (CompileTime, In, Out, ObjectFifo,
                       Program, Runtime, Worker, kernels)
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
import aie.iron as iron
from pathlib import Path

set_current_device(NPU2())

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
AIE2P_DIR = AIE_KERNEL_DIR / "aie2p"
INCLUDE_DIRS = [str(AIE_KERNEL_DIR.parent)]


def _compile_and_copy(name, jit_fn, specialize_args):
    t0 = time.perf_counter()
    result = jit_fn.specialize(**specialize_args)
    xclbin, insts = result.compile()
    tx = os.path.join(PROJ_ROOT, f"fst_{name}.xclbin")
    ti = os.path.join(PROJ_ROOT, f"fst_{name}_insts.bin")
    shutil.copy2(xclbin, tx)
    shutil.copy2(insts, ti)
    dt = time.perf_counter() - t0
    print(f"  {name}: {os.path.getsize(tx)}B xclbin + "
          f"{os.path.getsize(ti)}B insts ({dt:.1f}s)", flush=True)


# ── 1. RMSNorm (unweighted, installed kernel multiplies by gamma=1.0) ───────
@iron.jit
def rmsnorm_op(A: In, B: Out, *, N: CompileTime[int], el: CompileTime[type]):
    TILE = 1024
    rms_k = ExternalFunction(
        "rms_norm",
        source_file=str(AIE2P_DIR / "rms_norm.cc"),
        arg_types=[np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.int32],
        include_dirs=INCLUDE_DIRS,
    )

    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Ain",  depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Bout", depth=2)

    def core(of_a, of_b, rms_fn):
        for _ in range_(N // TILE):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            rms_fn(ea, eb, TILE)
            of_a.release(1)
            of_b.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.prod(), rms_k])
    tap = TensorAccessPattern((N,), 0, [1, N // TILE, TILE], [0, TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]],
                     np.ndarray[(N,), np.dtype[el]]) as (a, b):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=tap, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# ── 2. SiLU (built-in aie.iron kernel) ──────────────────────────────────────
@iron.jit
def silu_op(A: In, B: Out, *, N: CompileTime[int], el: CompileTime[type]):
    TILE = 1024
    silu_k = kernels.silu(tile_size=TILE)

    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Ain",  depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Bout", depth=2)

    def core(of_a, of_b, silu_fn):
        for _ in range_(N // TILE):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            silu_fn(ea, eb)
            of_a.release(1)
            of_b.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.prod(), silu_k])
    tap = TensorAccessPattern((N,), 0, [1, N // TILE, TILE], [0, TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]],
                     np.ndarray[(N,), np.dtype[el]]) as (a, b):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=tap, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# ── 3. Softmax (built-in aie.iron kernel) ───────────────────────────────────
@iron.jit
def softmax_op(A: In, B: Out, *, N: CompileTime[int], el: CompileTime[type]):
    TILE = 1024
    sm_k = kernels.softmax(tile_size=TILE)

    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Ain",  depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Bout", depth=2)

    def core(of_a, of_b, sm_fn):
        for _ in range_(N // TILE):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            sm_fn(ea, eb, TILE)
            of_a.release(1)
            of_b.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.prod(), sm_k])
    tap = TensorAccessPattern((N,), 0, [1, N // TILE, TILE], [0, TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]],
                     np.ndarray[(N,), np.dtype[el]]) as (a, b):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=tap, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# ── 4. RoPE (installed kernel: interleaved method) ──────────────────────────
# LUT contains interleaved [cos0, sin0, cos1, sin1, ...] per row.
# For rope_dim=64: rows=M, cols=64, angle_rows=M.
@iron.jit
def rope_op(A: In, LUT: In, B: Out, *,
            ROWS: CompileTime[int], COLS: CompileTime[int],
            AROWS: CompileTime[int], el: CompileTime[type]):
    TILE = COLS  # each tile = one row of COLS elements
    rope_k = ExternalFunction(
        "rope",
        source_file=str(AIE2P_DIR / "rope.cc"),
        arg_types=[np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.ndarray[(TILE,), np.dtype[el]],
                   np.int32],
        include_dirs=INCLUDE_DIRS,
    )

    fifo_A   = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Ain",  depth=2)
    fifo_LUT = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Lin",  depth=2)
    fifo_B   = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Bout", depth=2)

    def core(of_a, of_lut, of_b, rope_fn):
        for _ in range_(ROWS):
            ea    = of_a.acquire(1)
            e_lut = of_lut.acquire(1)
            eb    = of_b.acquire(1)
            rope_fn(ea, e_lut, eb, TILE)
            of_a.release(1)
            of_lut.release(1)
            of_b.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_LUT.cons(), fifo_B.prod(), rope_k])
    a_tap   = TensorAccessPattern((ROWS, COLS), 0,
                                   [1, ROWS, 1, COLS], [0, COLS, 1, 1])
    lut_tap = TensorAccessPattern((AROWS, COLS), 0,
                                   [1, 1, COLS], [0, COLS, 1])
    b_tap   = TensorAccessPattern((ROWS, COLS), 0,
                                   [1, ROWS, 1, COLS], [0, COLS, 1, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(ROWS, COLS), np.dtype[el]],
                     np.ndarray[(AROWS, COLS), np.dtype[el]],
                     np.ndarray[(ROWS, COLS), np.dtype[el]]) as (a, lut, b):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(),   a,   tap=a_tap,   task_group=tg)
        rt.fill(fifo_LUT.prod(), lut, tap=lut_tap, task_group=tg)
        rt.drain(fifo_B.cons(),  b,   tap=b_tap,   wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# ── 5. Element-wise multiply (for RMSNorm weight application) ───────────────
@iron.jit
def mul_op(A: In, B: In, C: Out, *, N: CompileTime[int], el: CompileTime[type]):
    TILE = 1024
    mul_k = kernels.mul(tile_size=TILE, dtype=el, vectorized=True)

    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Ain",  depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Bin",  depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="Cout", depth=2)

    def core(of_a, of_b, of_c, mul_fn):
        for _ in range_(N // TILE):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            ec = of_c.acquire(1)
            mul_fn(ea, eb, ec)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), mul_k])
    tap = TensorAccessPattern((N,), 0, [1, N // TILE, TILE], [0, TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]],
                     np.ndarray[(N,), np.dtype[el]],
                     np.ndarray[(N,), np.dtype[el]]) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=tap, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=tap, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Compiling NPU element-wise kernels ===\n")

    # 1) RMSNorm — hidden_dim=4096
    _compile_and_copy("rmsnorm", rmsnorm_op, {"N": 4096, "el": bfloat16})

    # 2) SiLU — M=32 expert FFN: 32*2048=65536; M=1 shared expert: 2048
    _compile_and_copy("silu", silu_op, {"N": 65536, "el": bfloat16})
    _compile_and_copy("silu_2048", silu_op, {"N": 2048, "el": bfloat16})

    # 3) Softmax — S=128 (padded to 1024); also 2048 for longer contexts
    _compile_and_copy("softmax", softmax_op, {"N": 1024, "el": bfloat16})
    _compile_and_copy("softmax_2048", softmax_op, {"N": 2048, "el": bfloat16})

    # 4) RoPE — rope_dim=64 (MLA_N_HEADS=64, rope_dim=1 per head,
    #    but we process the full MLA_KV_DECOMP=1024 with 64-dim chunks)
    #    For standard RoPE: rows=M, cols=rope_dim=64
    _compile_and_copy("rope", rope_op,
                      {"ROWS": 1, "COLS": 64, "AROWS": 1, "el": bfloat16})

    # 5) Element-wise multiply — hidden_dim=4096 (for RMSNorm weight application)
    _compile_and_copy("mul", mul_op, {"N": 4096, "el": bfloat16})

    print("\n=== Done ===")
