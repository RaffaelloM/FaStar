#!/usr/bin/env python3
"""gen_ew_dispatch.py -- 2-fifo single-sequence opcode-dispatcher EW kernel.

ONE aie.device, ONE CoreTile, ONE aie.runtime_sequence (@ew_dispatch).
Two ObjectFifos: ew_in, ew_out — each 65536 bf16, depth 2.
The host packs the opcode in ew_in[0], then data follows.
The C function processes element-by-element based on the opcode.
"""

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
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
INC = [str(AIE_KERNEL_DIR.parent), str(PROJ_ROOT)]

CHUNK = 1024
MAX_ELEM = 65536
MAX_CHUNKS = MAX_ELEM // CHUNK  # 64


@iron.jit
def ew_dispatch_op(A: In, C: Out):
    el = bfloat16
    chunk_fn = ExternalFunction(
        "ew_dispatch_chunk",
        source_file=SRC,
        arg_types=[
            np.ndarray[(CHUNK,), np.dtype[el]],
            np.ndarray[(CHUNK,), np.dtype[el]],
        ],
        include_dirs=INC,
    )

    fifo_A = ObjectFifo(np.ndarray[(CHUNK,), np.dtype[el]], name="ew_in", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(CHUNK,), np.dtype[el]], name="ew_out", depth=2)

    def core(of_a, of_c, fn_k):
        for _ in range_(MAX_CHUNKS):
            ea = of_a.acquire(1)
            ec = of_c.acquire(1)
            fn_k(ea, ec)
            of_a.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_C.prod(), chunk_fn])

    tap = TensorAccessPattern((MAX_ELEM,), 0, [MAX_CHUNKS, CHUNK], [CHUNK, 1])

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


def compile_dispatch():
    from aie.utils.compile import compile_cxx_core_function

    out_o = PROJ_ROOT / "fst_ew_dispatch_kernel.o"
    print(f"\nCompiling {out_o.name}...", flush=True)
    compile_cxx_core_function(
        source_path=SRC, target_arch="aie2p", output_path=str(out_o),
        include_dirs=INC, compile_args=["-O2", "-DNDEBUG"], cwd=str(PROJ_ROOT),
    )
    print(f"  {out_o.name}: {out_o.stat().st_size}B")

    print("Generating ew_dispatch MLIR...", flush=True)
    mlir_text = ew_dispatch_op.specialize().as_mlir()
    mlir_text = re.sub(
        r'attributes \{link_with = "[^"]+\.o"\}',
        'attributes {link_with = "fst_ew_dispatch_kernel.o"}',
        mlir_text,
    )
    mlir_path = PROJ_ROOT / "fst_ew_dispatch.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)
    print(f"  Wrote {mlir_path} ({len(mlir_text)} chars)")

    peano_dir = os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie")
    cmd = [
        "aiecc", "--aie-generate-xclbin", "--aie-generate-npu-insts",
        "--dump-intermediates", "--no-xchesscc", "--no-xbridge",
        f"--peano={peano_dir}", str(mlir_path),
        "-o", str(PROJ_ROOT / "fst_ew_dispatch"),
    ]
    print(f"Running aiecc...", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            print(f"  aiecc: {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-10:]:
            print(f"  aiecc ERR: {line}")
    if result.returncode != 0:
        raise RuntimeError(f"aiecc failed (exit {result.returncode})")

    main_xclbin = PROJ_ROOT / "main.xclbin"
    xclbin_path = PROJ_ROOT / "fst_ew_dispatch.xclbin"
    if main_xclbin.exists():
        shutil.copy2(str(main_xclbin), str(xclbin_path))
        main_xclbin.unlink()
    else:
        raise RuntimeError("main.xclbin not produced")
    print(f"SUCCESS: {xclbin_path}: {xclbin_path.stat().st_size}B ({time.perf_counter() - t0:.1f}s)")

    prj = PROJ_ROOT / "fst_ew_dispatch.mlir.prj"
    lowered_mlir = prj / "main_npu_lowered.mlir"
    if not lowered_mlir.exists():
        raise RuntimeError(f"{lowered_mlir} not found")

    out_bin = PROJ_ROOT / "fst_ew_dispatch_insts.bin"
    cmd = [
        "aie-translate", str(lowered_mlir), "--aie-npu-to-binary",
        f"--aie-sequence-name=ew_dispatch_op",
        "--aie-output-binary", "-o", str(out_bin),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd2 = [
            "aie-translate", str(lowered_mlir), "--aie-npu-to-binary",
            "--aie-output-binary", "-o", str(out_bin),
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"aie-translate failed: {result.stderr[-300:]}")
    sz = out_bin.stat().st_size if out_bin.exists() else 0
    print(f"  {out_bin.name}: {sz}B")


def main():
    print("=== EW 2-fifo Opcode Dispatcher ===\n")
    t_total = time.perf_counter()
    compile_dispatch()
    print(f"\n=== Total: {time.perf_counter() - t_total:.1f}s ===")


if __name__ == "__main__":
    main()
