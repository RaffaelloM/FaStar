#!/usr/bin/env python3
"""gen_ew_unified.py -- Generate ONE unified EW xclbin with 7 named
runtime_sequences and extract a per-sequence _insts.bin for each.

This mirrors gen_mla_unified.py exactly:
  - ONE aie.device(npu2)
  - ONE shared C++ source (fst_ew_unified_kernel.cc)
  - 7 named aie.runtime_sequence entries (@rmsnorm, @silu, @mul, @softmax,
    @rope, @router_gemm, @lm_head_gemm)
  - ONE aiecc invocation -> fst_ew_unified.xclbin
  - aie-translate --aie-sequence-name=<kernel> extracts 7 insts.bin payloads

Rules followed:
  - ZERO subprocess hacks.  ZERO manual MLIR pasting.
  - One MLIR file, one aiecc invocation, one xclbin.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
from ml_dtypes import bfloat16

# Make IRON importable
PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
IRON_PATH = str(PROJ_ROOT / "Source" / "IRON-devel")
if IRON_PATH not in sys.path:
    sys.path.insert(0, IRON_PATH)

os.environ.setdefault(
    "PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
)
os.environ.setdefault(
    "PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"),
)

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils import set_current_device, config

set_current_device(NPU2())

SRC = str(PROJ_ROOT / "fst_ew_unified_kernel.cc")
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
INC = [str(AIE_KERNEL_DIR.parent), str(PROJ_ROOT)]

# Compile-time shapes (must match engine padding)
TILE_M = 16   # M=16 standardization (router_gemm + lm_head_gemm gemm tile)
TILE_K = 64
TILE_N = 64

# Elementwise tile sizes
RMSNORM_N = 4096        # hidden dim (one row per ObjectFifo element)
RMSNORM_M = 8           # batched rows
SILU_TOTAL = 32768      # Mx(16) * INTER_DIM(2048)   [M=16 standardization]
SILU_TILE = 1024
# Batched draft-FFN silu/mul over 6 experts x Mx(16) x INTER_DIM(2048) = 196608.
SILU_BATCH = 6 * 16 * 2048
SOFTMAX_N = 1024        # S_padded
ROPE_ROWS = 8
ROPE_COLS = 512


# ============================================================================
# GEMM tile kernel builder (router_gemm, lm_head_gemm) -- mirrors MLA gemm
# ============================================================================
@iron.jit
def gemm_kernel(
    A: In,
    B: In,
    C: Out,
    *,
    name: CompileTime[str],
    mm_fn: CompileTime[str],
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    el: CompileTime[type],
):
    m, k, n = TILE_M, TILE_K, TILE_N
    # mmul mac_dims used by ew_matmul_bf16 (aie::mmul<4,8,8>).  The kernel is a
    # copy of IRON's matmul_vectorized_2x2_mmul, so it expects A/B/C tiles in
    # the SAME interleaved (r,s,t)-sub-block layout that IRON's kernels.mm
    # produces via dims_to_stream.  Without dims_to_stream the BD delivers raw
    # row-major tiles and the kernel reads the wrong sub-blocks -> correct
    # ones@ones (4096) but garbage on real weights (step-c DIFFER).
    r, s, t = 4, 8, 8

    zero_k = ExternalFunction(
        "ew_gemm_zero",
        source_file=SRC,
        arg_types=[np.ndarray[(m, n), np.dtype[el]]],
        include_dirs=INC,
    )
    mm_k = ExternalFunction(
        mm_fn,
        source_file=SRC,
        arg_types=[
            np.ndarray[(m, k), np.dtype[el]],
            np.ndarray[(k, n), np.dtype[el]],
            np.ndarray[(m, n), np.dtype[el]],
        ],
        include_dirs=INC,
    )

    inA = ObjectFifo(np.ndarray[(m, k), np.dtype[el]], name=f"{name}_Ain", depth=2)
    inB = ObjectFifo(np.ndarray[(k, n), np.dtype[el]], name=f"{name}_Bin", depth=2)
    memC = ObjectFifo(np.ndarray[(m, n), np.dtype[el]], name=f"{name}_Cout", depth=2)

    # dims_to_stream (identical to IRON's kernels.mm for b_row_maj): reshapes
    # each raw row-major tile into the (r,s,t) sub-block layout the mmul reads.
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name=f"{name}_memA", dims_to_stream=a_dims)
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]  # b_row_maj
    memB = inB.cons().forward(name=f"{name}_memB", dims_to_stream=b_dims)
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name=f"{name}_outC", dims_to_stream=c_dims)

    def core(of_a, of_b, of_c, zero_fn, mm):
        for _ in range_(M // m * N // n):
            ec = of_c.acquire(1)
            zero_fn(ec)
            for _ in range_(K // k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mm(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    w = Worker(core, [memA.cons(), memB.cons(), memC.prod(), zero_k, mm_k],
               stack_size=0xD00)

    # TAPs via group_tiler (same pattern as the probe-verified FFN/MLA canonical
    # builders).  The previous manual TensorAccessPattern used a stride-0 N//n
    # dim to re-read A for each N-tile; the NPU2 BD engine cannot express that
    # re-read via stride-0 and truncated the K-stream to 1 tile (lm_head/router
    # gave 64 = 4096/64 for ones@ones).  group_tiler's pattern_repeat emits a
    # proper BD repeat field, so all K//k K-tiles stream per N-tile.
    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k),
        pattern_repeat=N_div_n, prune_step=False)
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K_div_k, N_div_n),
        tile_group_col_major=True, prune_step=False)[0]
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m, n), (1, N_div_n), prune_step=False)

    rt = Runtime()
    with rt.sequence(
        np.ndarray[(M, K), np.dtype[el]],
        np.ndarray[(K, N), np.dtype[el]],
        np.ndarray[(M, N), np.dtype[el]],
    ) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(inA.prod(), a, tap=A_tiles[0], task_group=tg)
        rt.fill(inB.prod(), b, tap=b_tap, task_group=tg)
        rt.drain(outC.cons(), c, tap=C_tiles[0], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


# ============================================================================
# Elementwise builders
# ============================================================================
@iron.jit
def unary_kernel(
    A: In,
    B: Out,
    *,
    name: CompileTime[str],
    fn: CompileTime[str],
    TOTAL: CompileTime[int],
    TILE: CompileTime[int],
    el: CompileTime[type],
):
    k = ExternalFunction(
        fn,
        source_file=SRC,
        arg_types=[
            np.ndarray[(TILE,), np.dtype[el]],
            np.ndarray[(TILE,), np.dtype[el]],
            np.int32,
        ],
        include_dirs=INC,
    )
    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name=f"{name}_A", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name=f"{name}_B", depth=2)

    def core(of_a, of_b, fn_k):
        for _ in range_(TOTAL // TILE):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            fn_k(eb, ea, TILE)
            of_a.release(1)
            of_b.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.prod(), k])
    at = TensorAccessPattern((TOTAL,), 0, [TOTAL // TILE, TILE], [TILE, 1])
    bt = TensorAccessPattern((TOTAL,), 0, [TOTAL // TILE, TILE], [TILE, 1])
    rt = Runtime()
    with rt.sequence(
        np.ndarray[(TOTAL,), np.dtype[el]], np.ndarray[(TOTAL,), np.dtype[el]]
    ) as (a, b):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=bt, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def binary_kernel(
    A: In,
    B: In,
    C: Out,
    *,
    name: CompileTime[str],
    fn: CompileTime[str],
    TOTAL: CompileTime[int],
    TILE: CompileTime[int],
    el: CompileTime[type],
):
    k = ExternalFunction(
        fn,
        source_file=SRC,
        arg_types=[
            np.ndarray[(TILE,), np.dtype[el]],
            np.ndarray[(TILE,), np.dtype[el]],
            np.ndarray[(TILE,), np.dtype[el]],
            np.int32,
        ],
        include_dirs=INC,
    )
    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name=f"{name}_A", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name=f"{name}_B", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name=f"{name}_C", depth=2)

    def core(of_a, of_b, of_c, fn_k):
        for _ in range_(TOTAL // TILE):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            ec = of_c.acquire(1)
            fn_k(ec, ea, eb, TILE)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), k])
    at = TensorAccessPattern((TOTAL,), 0, [TOTAL // TILE, TILE], [TILE, 1])
    bt = TensorAccessPattern((TOTAL,), 0, [TOTAL // TILE, TILE], [TILE, 1])
    ct = TensorAccessPattern((TOTAL,), 0, [TOTAL // TILE, TILE], [TILE, 1])
    rt = Runtime()
    with rt.sequence(
        np.ndarray[(TOTAL,), np.dtype[el]],
        np.ndarray[(TOTAL,), np.dtype[el]],
        np.ndarray[(TOTAL,), np.dtype[el]],
    ) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def rmsnorm_kernel(
    A: In,
    W: In,
    B: Out,
    *,
    name: CompileTime[str],
    M: CompileTime[int],
    N: CompileTime[int],
    el: CompileTime[type],
):
    k = ExternalFunction(
        "ew_rmsnorm",
        source_file=SRC,
        arg_types=[
            np.ndarray[(N,), np.dtype[el]],
            np.ndarray[(N,), np.dtype[el]],
            np.ndarray[(N,), np.dtype[el]],
            np.int32,
        ],
        include_dirs=INC,
    )
    fifo_A = ObjectFifo(np.ndarray[(N,), np.dtype[el]], name=f"{name}_A", depth=2)
    fifo_W = ObjectFifo(np.ndarray[(N,), np.dtype[el]], name=f"{name}_W", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(N,), np.dtype[el]], name=f"{name}_B", depth=2)

    def core(of_a, of_w, of_b, fn_k):
        for _ in range_(M):
            ea = of_a.acquire(1)
            ew = of_w.acquire(1)
            eb = of_b.acquire(1)
            fn_k(eb, ea, ew, N)
            of_a.release(1)
            of_w.release(1)
            of_b.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_W.cons(), fifo_B.prod(), k])
    at = TensorAccessPattern((M, N), 0, [M, N], [N, 1])
    wt = TensorAccessPattern((M, N), 0, [M, N], [N, 1])
    bt = TensorAccessPattern((M, N), 0, [M, N], [N, 1])
    rt = Runtime()
    with rt.sequence(
        np.ndarray[(M, N), np.dtype[el]],
        np.ndarray[(M, N), np.dtype[el]],
        np.ndarray[(M, N), np.dtype[el]],
    ) as (a, w_in, b):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_W.prod(), w_in, tap=wt, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=bt, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


@iron.jit
def rope_kernel(
    A: In,
    L: In,
    B: Out,
    *,
    name: CompileTime[str],
    ROWS: CompileTime[int],
    COLS: CompileTime[int],
    el: CompileTime[type],
):
    k = ExternalFunction(
        "ew_rope",
        source_file=SRC,
        arg_types=[
            np.ndarray[(COLS,), np.dtype[el]],
            np.ndarray[(COLS,), np.dtype[el]],
            np.ndarray[(COLS,), np.dtype[el]],
            np.int32,
            np.int32,
        ],
        include_dirs=INC,
    )
    fifo_A = ObjectFifo(np.ndarray[(COLS,), np.dtype[el]], name=f"{name}_A", depth=2)
    fifo_L = ObjectFifo(np.ndarray[(COLS,), np.dtype[el]], name=f"{name}_L", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(COLS,), np.dtype[el]], name=f"{name}_B", depth=2)

    def core(of_a, of_l, of_b, fn_k):
        for _ in range_(ROWS):
            ea = of_a.acquire(1)
            el_ = of_l.acquire(1)
            eb = of_b.acquire(1)
            fn_k(eb, ea, el_, 1, COLS)
            of_a.release(1)
            of_l.release(1)
            of_b.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_L.cons(), fifo_B.prod(), k])
    at = TensorAccessPattern((ROWS, COLS), 0, [ROWS, COLS], [COLS, 1])
    lt = TensorAccessPattern((ROWS, COLS), 0, [ROWS, COLS], [COLS, 1])
    bt = TensorAccessPattern((ROWS, COLS), 0, [ROWS, COLS], [COLS, 1])
    rt = Runtime()
    with rt.sequence(
        np.ndarray[(ROWS, COLS), np.dtype[el]],
        np.ndarray[(ROWS, COLS), np.dtype[el]],
        np.ndarray[(ROWS, COLS), np.dtype[el]],
    ) as (a, l, b):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_L.prod(), l, tap=lt, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=bt, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# ============================================================================
# MLIR merging helpers (copied from gen_mla_unified.py)
# ============================================================================
def extract_device_body(mlir_text: str) -> str:
    lines = mlir_text.split("\n")
    start = -1
    depth = 0
    for i, line in enumerate(lines):
        if "aie.device(npu2)" in line or "aie.device(npu1)" in line:
            start = i
            depth = line.count("{") - line.count("}")
            continue
        if start >= 0:
            depth += line.count("{") - line.count("}")
            if depth == 0:
                return "\n".join(lines[start + 1 : i])
    return ""


def extract_runtime_sequence(body: str):
    idx = body.find("aie.runtime_sequence(")
    if idx < 0:
        return body, [], [], ""
    decls = body[:idx].rstrip()
    args_start = body.index("(", idx) + 1
    args_end = body.index(")", args_start)
    args_str = body[args_start:args_end]
    rt_arg_decls = []
    rt_arg_names = []
    for arg in args_str.split(","):
        arg = arg.strip()
        if ":" in arg:
            name, typ = arg.split(":", 1)
            rt_arg_decls.append((name.strip(), typ.strip()))
            rt_arg_names.append(name.strip())
    body_open = body.index("{", args_end)
    depth = 1
    i = body_open + 1
    while depth > 0 and i < len(body):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
        i += 1
    rt_body = body[body_open + 1 : i - 1].strip()
    return decls, rt_arg_decls, rt_arg_names, rt_body


def rename_decls_and_cores(decls_and_cores: str, prefix: str) -> str:
    result = []
    for line in decls_and_cores.split("\n"):
        if "func.func private" in line:
            continue
        line = re.sub(r"(%[a-zA-Z_][a-zA-Z0-9_]*)", lambda m: f"{m.group(1)}_{prefix}", line)
        line = re.sub(r"%(\d+)\b", lambda m: f"%v{m.group(1)}_{prefix}", line)
        result.append(line)
    return "\n".join(result)


def rename_rt_body(rt_body: str, original_arg_names: list[str], prefix: str) -> str:
    result = rt_body
    for arg_name in original_arg_names:
        escaped = re.escape(arg_name)
        result = re.sub(escaped + r"\b", f"{arg_name}_{prefix}", result)
    result = re.sub(r"%(\d+)\b", lambda m: f"%v{m.group(1)}_{prefix}", result)
    return result


def deduplicate_func_decls(all_raw_bodies: list[str]) -> list[str]:
    seen = set()
    decls = []
    for body in all_raw_bodies:
        for line in body.split("\n"):
            if "func.func private" in line:
                m = re.search(r"func\.func\s+private\s+(@?\S+)\(([^)]*)\)", line)
                if m:
                    key = (m.group(1), m.group(2))
                    if key not in seen:
                        seen.add(key)
                        decls.append(line.strip())
    return decls


# ============================================================================
# Generate each kernel's MLIR and merge
# ============================================================================
def generate_unified_mlir() -> str:
    kernel_mlirs = OrderedDict()

    specs = [
        ("rmsnorm", dict(builder="rmsnorm", M=RMSNORM_M, N=RMSNORM_N)),
        ("silu", dict(builder="unary", fn="ew_silu", TOTAL=SILU_TOTAL, TILE=SILU_TILE)),
        ("mul", dict(builder="binary", fn="ew_mul", TOTAL=SILU_TOTAL, TILE=SILU_TILE)),
        # Batched draft-FFN: 6 experts x Mx(16) x INTER_DIM(2048) in one dispatch.
        ("silu_b", dict(builder="unary", fn="ew_silu", TOTAL=SILU_BATCH, TILE=SILU_TILE)),
        ("mul_b", dict(builder="binary", fn="ew_mul", TOTAL=SILU_BATCH, TILE=SILU_TILE)),
        ("softmax", dict(builder="unary", fn="ew_softmax", TOTAL=SOFTMAX_N, TILE=SOFTMAX_N)),
        ("rope", dict(builder="rope", ROWS=ROPE_ROWS, COLS=ROPE_COLS)),
        ("router_gemm", dict(builder="gemm", mm_fn="ew_router_gemm", M=16, K=4096, N=256)),
        ("lm_head_gemm", dict(builder="gemm", mm_fn="ew_lm_head_gemm", M=16, K=4096, N=2048)),
    ]

    for name, spec in specs:
        print(f"Generating {name} MLIR...", flush=True)
        t0 = time.perf_counter()
        b = spec["builder"]
        if b == "gemm":
            design = gemm_kernel.specialize(
                name=name, mm_fn=spec["mm_fn"], M=spec["M"], K=spec["K"], N=spec["N"],
                el=bfloat16,
            )
        elif b == "unary":
            design = unary_kernel.specialize(
                name=name, fn=spec["fn"], TOTAL=spec["TOTAL"], TILE=spec["TILE"], el=bfloat16,
            )
        elif b == "binary":
            design = binary_kernel.specialize(
                name=name, fn=spec["fn"], TOTAL=spec["TOTAL"], TILE=spec["TILE"], el=bfloat16,
            )
        elif b == "rmsnorm":
            design = rmsnorm_kernel.specialize(
                name=name, M=spec["M"], N=spec["N"], el=bfloat16,
            )
        elif b == "rope":
            design = rope_kernel.specialize(
                name=name, ROWS=spec["ROWS"], COLS=spec["COLS"], el=bfloat16,
            )
        mlir = design.as_mlir()
        kernel_mlirs[name] = mlir
        print(f"  {len(mlir)} chars ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("\nMerging into unified device...")
    kernel_parts = OrderedDict()
    for name, full_mlir in kernel_mlirs.items():
        body = extract_device_body(full_mlir)
        decls, rt_arg_decls, rt_arg_names, rt_body = extract_runtime_sequence(body)
        kernel_parts[name] = (decls, rt_arg_decls, rt_arg_names, rt_body)
        print(f"  {name}: decls={len(decls)} chars, rt_args={len(rt_arg_decls)}")

    all_raw_bodies = [decls for decls, _, _, _ in kernel_parts.values()]
    func_decls = deduplicate_func_decls(all_raw_bodies)

    renamed_cores = []
    sequence_blocks = []
    for name, (decls, rt_arg_decls, rt_arg_names, rt_body) in kernel_parts.items():
        renamed_cores.append(rename_decls_and_cores(decls, name))
        renamed_arg_decls = [(f"{n}_{name}", t) for n, t in rt_arg_decls]
        rt_args_str = ", ".join(f"{n}: {t}" for n, t in renamed_arg_decls)
        renamed_body = rename_rt_body(rt_body, rt_arg_names, name)
        indented_body = "\n".join("      " + line for line in renamed_body.split("\n") if line.strip())
        seq_block = f"    aie.runtime_sequence @{name}({rt_args_str}) {{\n{indented_body}\n    }}"
        sequence_blocks.append(seq_block)

    func_decls_str = "\n".join(f"    {d}" for d in func_decls)
    cores_str = "\n".join(renamed_cores)
    sequences_str = "\n\n".join(sequence_blocks)

    combined = f"""module {{
  aie.device(npu2) {{
    // === Shared kernel function declarations ===
{func_decls_str}

    // === Kernel tiles / objectfifos / cores ===
{cores_str}

    // === Named runtime sequences (one per host-callable kernel) ===
{sequences_str}
  }}
}}"""
    return combined


# ============================================================================
# Compilation and per-kernel insts.bin extraction
# ============================================================================
def compile_external_objects():
    from aie.utils.compile import compile_cxx_core_function

    out_o = PROJ_ROOT / "fst_ew_unified_kernel.o"
    print(f"Compiling {out_o.name}...", flush=True)
    compile_cxx_core_function(
        source_path=SRC,
        target_arch="aie2p",
        output_path=str(out_o),
        include_dirs=INC,
        compile_args=["-O2", "-DNDEBUG"],
        cwd=str(PROJ_ROOT),
    )
    print(f"  {out_o.name}: {out_o.stat().st_size}B")


def unify_link_with(mlir_text: str) -> str:
    return re.sub(
        r'attributes \{link_with = "[^"]+\.o"\}',
        'attributes {link_with = "fst_ew_unified_kernel.o"}',
        mlir_text,
    )


def compile_unified(mlir_text: str) -> Path:
    compile_external_objects()
    mlir_path = PROJ_ROOT / "fst_ew_unified.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)
    print(f"\nWrote unified MLIR: {mlir_path}")

    # Remove stale aiecc project dir so per-sequence extraction sees fresh MLIR
    prj = PROJ_ROOT / "fst_ew_unified.mlir.prj"
    if prj.exists():
        shutil.rmtree(prj)
        print(f"Removed stale {prj}")

    peano_dir = os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie")
    cmd = [
        "aiecc",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--dump-intermediates",
        "--no-xchesscc",
        "--no-xbridge",
        f"--peano={peano_dir}",
        str(mlir_path),
        "-o",
        str(PROJ_ROOT / "fst_ew_unified"),
    ]
    print(f"Running aiecc...\n  {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-20:]:
            print(f"  aiecc stdout: {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"  aiecc stderr: {line}")
    if result.returncode != 0:
        raise RuntimeError(f"aiecc failed (exit {result.returncode})")

    main_xclbin = PROJ_ROOT / "main.xclbin"
    xclbin_path = PROJ_ROOT / "fst_ew_unified.xclbin"
    if main_xclbin.exists():
        shutil.copy2(str(main_xclbin), str(xclbin_path))
        main_xclbin.unlink()
    else:
        raise RuntimeError("main.xclbin not produced by aiecc")
    print(f"SUCCESS: {xclbin_path}: {xclbin_path.stat().st_size}B ({dt:.1f}s)")
    return xclbin_path


def extract_per_kernel_insts():
    prj = PROJ_ROOT / "fst_ew_unified.mlir.prj"
    if not prj.exists():
        raise RuntimeError(f"aiecc project directory not found: {prj}")
    lowered_mlir = prj / "main_npu_lowered.mlir"
    print(f"\nExtracting per-kernel insts from {lowered_mlir}")
    kernel_names = ["rmsnorm", "silu", "mul", "silu_b", "mul_b", "softmax", "rope", "router_gemm", "lm_head_gemm"]
    for name in kernel_names:
        out_bin = PROJ_ROOT / f"fst_ew_{name}_insts.bin"
        cmd = [
            "aie-translate",
            str(lowered_mlir),
            "--aie-npu-to-binary",
            f"--aie-sequence-name={name}",
            "--aie-output-binary",
            "-o",
            str(out_bin),
        ]
        print(f"  {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED to extract {name}: {result.stderr[-300:]}")
            continue
        sz = out_bin.stat().st_size if out_bin.exists() else 0
        print(f"  {out_bin.name}: {sz}B")


def main():
    print("=== Unified EW Generator ===\n")
    t_total = time.perf_counter()
    mlir_text = generate_unified_mlir()
    mlir_text = unify_link_with(mlir_text)
    mlir_path = PROJ_ROOT / "fst_ew_unified.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)
    print(f"\nUnified MLIR: {len(mlir_text)} chars -> {mlir_path}")
    compile_unified(mlir_text)
    extract_per_kernel_insts()
    print(f"\n=== Total time: {time.perf_counter() - t_total:.1f}s ===")


if __name__ == "__main__":
    main()
