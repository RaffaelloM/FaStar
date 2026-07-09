#!/usr/bin/env python3
"""Compile Token Sampling xclbin for DS4-XDNA (NPU2).

Tiled sampling: Worker loops over V/TILE tiles per dispatch.
Host dispatches 3 separate passes (scale, softmax, argmax).
params buffer: NT copies of [inv_temp, global_max, inv_sum, pass_num].
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
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
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

VOCAB_SIZE = 129280
TILE = 1024
NTILES = VOCAB_SIZE // TILE  # 126


@iron.jit
def sample_op(data_in: In, data_out: Out, aux_out: Out, params_in: In,
              *, NT: CompileTime[int], TILE_CT: CompileTime[int],
                 el: CompileTime[type], fl: CompileTime[type]):
    sample_k = ExternalFunction(
        "fst_sample_kernel",
        source_file=str(PROJ_ROOT / "fst_sample_kernel.cc"),
        arg_types=[np.ndarray[(TILE_CT,), np.dtype[el]],
                   np.ndarray[(TILE_CT,), np.dtype[el]],
                   np.ndarray[(2,), np.dtype[fl]],
                   np.ndarray[(4,), np.dtype[fl]],
                   np.int32],
        include_dirs=INCLUDE_DIRS + [str(PROJ_ROOT)],
    )

    fifo_in   = ObjectFifo(np.ndarray[(TILE_CT,), np.dtype[el]], name="Fin",   depth=2)
    fifo_out  = ObjectFifo(np.ndarray[(TILE_CT,), np.dtype[el]], name="Fout",  depth=2)
    fifo_aux  = ObjectFifo(np.ndarray[(2,), np.dtype[fl]],       name="Faux",  depth=2)
    fifo_params = ObjectFifo(np.ndarray[(4,), np.dtype[fl]],     name="Fpar",  depth=NT)

    def core(of_i, of_o, of_a, of_p, fn):
        for _ in range_(NT):
            ti = of_i.acquire(1)
            to = of_o.acquire(1)
            ao = of_a.acquire(1)
            pa = of_p.acquire(1)
            fn(ti, to, ao, pa, TILE_CT)
            of_i.release(1)
            of_o.release(1)
            of_a.release(1)
            of_p.release(1)

    w = Worker(core, [fifo_in.cons(), fifo_out.prod(), fifo_aux.prod(),
                       fifo_params.cons(), sample_k])

    ti_tap  = TensorAccessPattern((VOCAB_SIZE,), 0, [1, NT, TILE_CT], [0, TILE_CT, 1])
    to_tap  = TensorAccessPattern((VOCAB_SIZE,), 0, [1, NT, TILE_CT], [0, TILE_CT, 1])
    ao_tap  = TensorAccessPattern((NT * 2,), 0, [1, NT, 2], [0, 2, 1])
    par_tap = TensorAccessPattern((NT * 4,), 0, [1, NT, 4], [0, 4, 1])

    rt = Runtime()
    with rt.sequence(np.ndarray[(VOCAB_SIZE,), np.dtype[el]],
                     np.ndarray[(VOCAB_SIZE,), np.dtype[el]],
                     np.ndarray[(NT * 2,), np.dtype[fl]],
                     np.ndarray[(NT * 4,), np.dtype[fl]]) as (din, dout, aux, par):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_in.prod(), din, tap=ti_tap, task_group=tg)
        rt.fill(fifo_params.prod(), par, tap=par_tap, task_group=tg)
        rt.drain(fifo_aux.cons(), aux, tap=ao_tap, task_group=tg)
        rt.drain(fifo_out.cons(), dout, tap=to_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def _compile_and_copy(name, jit_fn, specialize_args):
    t0 = time.perf_counter()
    result = jit_fn.specialize(**specialize_args)
    xclbin, insts = result.compile()
    tx = str(PROJ_ROOT / f"{name}.xclbin")
    ti = str(PROJ_ROOT / f"{name}_insts.bin")
    shutil.copy2(xclbin, tx)
    shutil.copy2(insts, ti)
    dt = time.perf_counter() - t0
    print(f"  {name}: {os.path.getsize(tx)}B xclbin + "
          f"{os.path.getsize(ti)}B insts ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    print("=== Compiling Token Sampling xclbin ===\n")
    _compile_and_copy("fst_sample", sample_op,
                      {"NT": NTILES, "TILE_CT": TILE,
                       "el": bfloat16, "fl": np.float32})
    print("\n=== Done ===")
