#!/usr/bin/env python3
"""Test minimal version: single objectfifo pair (no in_b)."""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
IRON_PATH = str(PROJ_ROOT / "Source" / "IRON-devel")
if IRON_PATH not in sys.path:
    sys.path.insert(0, IRON_PATH)

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config

set_current_device(NPU2())

SRC = str(PROJ_ROOT / "fst_ew_dispatch_kernel.cc")
INC = [str(Path(config.cxx_header_path()).parent), str(PROJ_ROOT)]

CHUNK = 1024
MAX_ELEM = 65536

@iron.jit
def simple_dispatch(A: In, C: Out):
    el = bfloat16
    chunk_fn = ExternalFunction(
        "ew_dispatch_chunk_simpler",
        source_file=SRC,
        arg_types=[np.ndarray[(CHUNK,), np.dtype[el]], np.ndarray[(CHUNK,), np.dtype[el]]],
        include_dirs=INC,
    )
    fifo_A = ObjectFifo(np.ndarray[(CHUNK,), np.dtype[el]], name="simp_a", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(CHUNK,), np.dtype[el]], name="simp_c", depth=2)

    def core(of_a, of_c, fn_k):
        for _ in range_(MAX_ELEM // CHUNK):
            ea = of_a.acquire(1)
            ec = of_c.acquire(1)
            fn_k(ea, ec)
            of_a.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_C.prod(), chunk_fn])
    tap = TensorAccessPattern((MAX_ELEM,), 0, [MAX_ELEM // CHUNK, CHUNK], [CHUNK, 1])

    rt = Runtime()
    with rt.sequence(
        np.ndarray[(MAX_ELEM,), np.dtype[el]],
        np.ndarray[(MAX_ELEM,), np.dtype[el]],
    ) as (a, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=tap, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def main():
    from aie.utils.compile import compile_cxx_core_function
    out_o = PROJ_ROOT / "fst_simple_dispatch.o"
    compile_cxx_core_function(
        source_path=SRC, target_arch="aie2p", output_path=str(out_o),
        include_dirs=INC, compile_args=["-O2", "-DNDEBUG"], cwd=str(PROJ_ROOT),
    )

    mlir_text = simple_dispatch.specialize().as_mlir()
    mlir_text = re.sub(
        r'attributes \{link_with = "[^"]+\.o"\}',
        'attributes {link_with = "fst_simple_dispatch.o"}',
        mlir_text,
    )
    mlir_path = PROJ_ROOT / "fst_simple_dispatch.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)
    print(f"MLIR: {len(mlir_text)} chars")

    peano_dir = os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie")
    cmd = [
        "aiecc", "--aie-generate-xclbin", "--aie-generate-npu-insts",
        "--dump-intermediates", "--no-xchesscc", "--no-xbridge",
        f"--peano={peano_dir}", str(mlir_path),
        "-o", str(PROJ_ROOT / "fst_simple_dispatch"),
    ]
    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
    print(f"aiecc exit={result.returncode}")
    if result.stderr:
        print(f"aiecc stderr: {result.stderr[-500:]}")

    main_x = PROJ_ROOT / "main.xclbin"
    if main_x.exists():
        target = PROJ_ROOT / "fst_simple_dispatch.xclbin"
        shutil.copy2(str(main_x), str(target))
        main_x.unlink()
        print(f"xclbin: {target}")

    prj = PROJ_ROOT / "fst_simple_dispatch.mlir.prj"
    lowered = prj / "main_npu_lowered.mlir"
    out_bin = PROJ_ROOT / "fst_simple_dispatch_insts.bin"
    subprocess.run([
        "aie-translate", str(lowered), "--aie-npu-to-binary",
        "--aie-output-binary", "-o", str(out_bin),
    ], capture_output=True, text=True)
    print(f"insts: {out_bin.stat().st_size if out_bin.exists() else 0}B")


if __name__ == "__main__":
    main()
