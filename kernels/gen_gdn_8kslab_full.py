#!/usr/bin/env python3
"""gen_gdn_8kslab_full.py — FULL-RECURRENCE 8 KB column-slab stack probe builder.

Builds fst_gdn_8kslab_full.xclbin (real GDN 3-pass recurrence, K=8 × 48 v-heads,
8 KB bf16 stack S) and fst_gdn_8kslab_noop.xclbin (same 8 KB stack + DMA, no
recurrence — DMA/alloc floor for subtraction).  1 tile, 1 MM2S + 1 S2MM, single
call, stack_size=0x5000 (8 KB S + recur scratch frame).

See kernels/fst_gdn_8kslab_full_kernel.cc + docs/QWOPUS_GDN_8KB_SLAB_ANALYSIS.md.

Build each variant in a SEPARATE process: @iron.jit caches the resolved program
in-process and fn_name is not part of the cache key, so building both in one
process yields two identical xclbins.  `both` spawns two subprocesses.

  python3 kernels/gen_gdn_8kslab_full.py full   # build the recurrence xclbin
  python3 kernels/gen_gdn_8kslab_full.py noop   # build the noop xclbin
  python3 kernels/gen_gdn_8kslab_full.py both    # both, in separate processes
"""
import os, sys, shutil, subprocess
from pathlib import Path
PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PATH", os.environ["PATH"] + ":" + os.path.expanduser("~/.local/bin"))
os.environ.setdefault("PEANO_INSTALL_DIR",
                       os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config
set_current_device(NPU2())

NVH = 48; TOTAL = NVH            # in: 48 floats (unused, deterministic); out: 48 checksums
SRC = str(PROJ_ROOT / "fst_gdn_8kslab_full_kernel.cc")
INC = [aie_config.cxx_header_path()]


def _build(fn_name, xclbin_name, obj_name):
    T = np.ndarray[(TOTAL,), np.dtype[np.float32]]

    @iron.jit
    def op(inp: iron.In, outp: iron.Out):
        k = ExternalFunction(fn_name, source_file=SRC, arg_types=[T, T],
                             include_dirs=INC, object_file_name=obj_name)
        f_in  = ObjectFifo(T, name="in",  depth=2)
        f_out = ObjectFifo(T, name="out", depth=2)
        def core(fin, fout, k):
            for _ in range_(1):
                pi = fin.acquire(1); po = fout.acquire(1)
                k(pi, po)
                fin.release(1); fout.release(1)
        # stack_size=0x3000 (12 KB): 8 KB S + ~1.5 KB recur scratch (kn/qn/a/b/delta/
        # vv) fits; larger (0x5000) overlaps the ObjectFifo buffer at 0x4000.
        w = Worker(core, [f_in.cons(), f_out.prod(), k], stack_size=0x3000)
        rt = Runtime()
        with rt.sequence(np.ndarray[(TOTAL,), np.dtype[np.float32]],
                         np.ndarray[(TOTAL,), np.dtype[np.float32]]) as (inp, outp):
            rt.start(w)
            tg = rt.task_group()
            rt.fill(f_in.prod(), inp,
                    tap=TensorAccessPattern(tensor_dims=(1, TOTAL), offset=0,
                                            sizes=[1, TOTAL], strides=[TOTAL, 1]),
                    task_group=tg)
            rt.drain(f_out.cons(), outp,
                     tap=TensorAccessPattern(tensor_dims=(TOTAL,), offset=0,
                                             sizes=[1, TOTAL], strides=[0, 1]),
                     task_group=tg, wait=True)
            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()

    xclbin, insts = op.compile()
    out_xclbin = str(PROJ_ROOT / f"{xclbin_name}.xclbin")
    out_insts  = str(PROJ_ROOT / f"{xclbin_name}_insts.bin")
    shutil.copy(xclbin, out_xclbin); shutil.copy(insts, out_insts)
    print(f"{xclbin_name}: {out_xclbin} ({os.path.getsize(out_xclbin)}B)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which == "both":
        # separate processes — @iron.jit in-process cache would otherwise make
        # both xclbins identical (the first build).
        for sub in ("full", "noop"):
            r = subprocess.run([sys.executable, __file__, sub])
            if r.returncode != 0:
                sys.exit(r.returncode)
    elif which == "full":
        _build("gdn_8kslab_full", "fst_gdn_8kslab_full", "fst_gdn_8kslab_full.o")
    elif which == "noop":
        _build("gdn_8kslab_noop", "fst_gdn_8kslab_noop", "fst_gdn_8kslab_noop.o")
    else:
        sys.exit(f"unknown variant '{which}' (full|noop|both)")
    print(f"OK gdn_8kslab [{which}] — 1 tile, 8KB stack, 48 v-heads x K=8")