#!/usr/bin/env python3
"""gen_mla_unified.py -- Generate a single unified MLA MLIR and compile it.

This script uses the IRON Python API to build each of the 8 MLA kernels
(qc, kvc, qk, sv, oa, ob, wq_b, k_pe) as a separate `aie.runtime_sequence`
inside one coherent `aie.device(npu2)`.  All kernels link against a single
C++ source file: `fst_mla_unified_kernels.cc`.

The script then invokes `aiecc` once and extracts per-kernel NPU instruction
binaries with `aie-translate --aie-sequence-name=<kernel>` so that every
`insts.bin` payload matches the unified xclbin's DMA layout.

Rules followed:
  - ZERO subprocess hacks.
  - ZERO manual MLIR text pasting.
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
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config

set_current_device(NPU2())

# Tile dimensions used by every kernel
TILE_M = 8
TILE_K = 64
TILE_N = 64

# Full matrix dimensions
M_FIX = 8
K_FULL_DIM = 32768          # kept for reference (old decompressed-K qk)
D_FIX = 1024
S_MAX = 128                 # max_seq = 128

# Latent attention dimensions (DeepSeek V4 Flash: no KV decompression)
M_LATENT = 512              # M_FIX * MLA_N_HEADS = 8 * 64
K_LATENT = 512              # MLA_HEAD_DIM (kv_latent shared across heads)
D_LATENT = 512              # SV output dim = head_dim (same as kv_latent)

SRC = str(PROJ_ROOT / "fst_mla_unified_kernels.cc")
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
INC = [str(AIE_KERNEL_DIR.parent), str(PROJ_ROOT)]


# ============================================================================
# IRON kernel builders.  Each builder is a separate @iron.jit function so that
# the generated MLIR contains exactly one runtime_sequence per kernel.  We call
# them sequentially in the same process to avoid the multi-@iron.jit segfault.
# ============================================================================

@iron.jit
def gemm_kernel(
    A: In,
    B: In,
    C: Out,
    *,
    name: CompileTime[str],
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    el: CompileTime[type],
):
    """Standard GEMM tile kernel (qc, kvc, oa, ob, wq_b, k_pe)."""
    m, k, n = TILE_M, TILE_K, TILE_N

    zero_k = ExternalFunction(
        "mla_zero",
        source_file=SRC,
        arg_types=[np.ndarray[(m * n,), np.dtype[el]]],
        include_dirs=INC,
    )
    mm_k = ExternalFunction(
        "mla_gemm",
        source_file=SRC,
        arg_types=[
            np.ndarray[(m * k,), np.dtype[el]],
            np.ndarray[(k * n,), np.dtype[el]],
            np.ndarray[(m * n,), np.dtype[el]],
        ],
        include_dirs=INC,
    )

    fifo_A = ObjectFifo(
        np.ndarray[(m * k,), np.dtype[el]], name=f"{name}_Ain", depth=2
    )
    fifo_B = ObjectFifo(
        np.ndarray[(k * n,), np.dtype[el]], name=f"{name}_Bin", depth=2
    )
    fifo_C = ObjectFifo(
        np.ndarray[(m * n,), np.dtype[el]], name=f"{name}_Cout", depth=2
    )

    def core(of_a, of_b, of_c, zero_fn, mm_fn):
        for _ in range_(M // m * N // n):
            ec = of_c.acquire(1)
            zero_fn(ec)
            for _ in range_(K // k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mm_fn(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), zero_k, mm_k])

    at = TensorAccessPattern((M, K), 0, [N // n, K // k, m, k], [0, k, K, 1])
    bt = TensorAccessPattern((K, N), 0, [N // n, K // k, k, n], [n, k * N, N, 1])
    ct = TensorAccessPattern((M, N), 0, [1, N // n, m, n], [m * N, n, N, 1])

    rt = Runtime()
    with rt.sequence(
        np.ndarray[(M, K), np.dtype[el]],
        np.ndarray[(K, N), np.dtype[el]],
        np.ndarray[(M, N), np.dtype[el]],
    ) as (a, b, c):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


@iron.jit
def qk_kernel(
    Q: In,
    K_mat: In,
    scores: Out,
    *,
    M: CompileTime[int],
    K_dim: CompileTime[int],
    S: CompileTime[int],
    el: CompileTime[type],
):
    m, k, n = TILE_M, TILE_K, TILE_N

    zero_k = ExternalFunction(
        "mla_zero",
        source_file=SRC,
        arg_types=[np.ndarray[(m * n,), np.dtype[el]]],
        include_dirs=INC,
    )
    mm_k = ExternalFunction(
        "mla_gemm",
        source_file=SRC,
        arg_types=[
            np.ndarray[(m * k,), np.dtype[el]],
            np.ndarray[(k * n,), np.dtype[el]],
            np.ndarray[(m * n,), np.dtype[el]],
        ],
        include_dirs=INC,
    )
    sc_k = ExternalFunction(
        "mla_qk_scale",
        source_file=SRC,
        arg_types=[np.ndarray[(m * n,), np.dtype[el]], np.int32],
        include_dirs=INC,
    )

    fifo_A = ObjectFifo(np.ndarray[(m * k,), np.dtype[el]], name="qk_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k * n,), np.dtype[el]], name="qk_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m * n,), np.dtype[el]], name="qk_Cout", depth=2)

    S_div_n, K_div_k = S // n, K_dim // k

    def core(of_a, of_b, of_c, zero_fn, mm_fn, scale_fn):
        for _ in range_(S_div_n):
            ec = of_c.acquire(1)
            zero_fn(ec)
            for _ in range_(K_div_k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mm_fn(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            scale_fn(ec, K_dim)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), zero_k, mm_k, sc_k])

    at = TensorAccessPattern((M, K_dim), 0, [S_div_n, K_div_k, m, k], [0, k, K_dim, 1])
    bt = TensorAccessPattern(
        (K_dim, S), 0, [S_div_n, K_div_k, k, n], [n, k * S, S, 1]
    )
    ct = TensorAccessPattern((M, S), 0, [1, S_div_n, m, n], [m * S, n, S, 1])

    rt = Runtime()
    with rt.sequence(
        np.ndarray[(M, K_dim), np.dtype[el]],
        np.ndarray[(K_dim, S), np.dtype[el]],
        np.ndarray[(M, S), np.dtype[el]],
    ) as (q, k, sc):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), q, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), k, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), sc, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


@iron.jit
def sv_kernel(
    scores_in: In,
    V_in: In,
    out: Out,
    *,
    M: CompileTime[int],
    S: CompileTime[int],
    D: CompileTime[int],
    el: CompileTime[type],
):
    m, k, n = TILE_M, TILE_K, TILE_N

    zero_k = ExternalFunction(
        "mla_zero",
        source_file=SRC,
        arg_types=[np.ndarray[(m * n,), np.dtype[el]]],
        include_dirs=INC,
    )
    mm_k = ExternalFunction(
        "mla_gemm",
        source_file=SRC,
        arg_types=[
            np.ndarray[(m * k,), np.dtype[el]],
            np.ndarray[(k * n,), np.dtype[el]],
            np.ndarray[(m * n,), np.dtype[el]],
        ],
        include_dirs=INC,
    )
    sc_k = ExternalFunction(
        "mla_sv_scale",
        source_file=SRC,
        arg_types=[np.ndarray[(m * n,), np.dtype[el]], np.int32],
        include_dirs=INC,
    )

    fifo_A = ObjectFifo(np.ndarray[(m * k,), np.dtype[el]], name="sv_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k * n,), np.dtype[el]], name="sv_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m * n,), np.dtype[el]], name="sv_Cout", depth=2)

    D_div_n, S_div_k = D // n, S // k

    def core(of_a, of_b, of_c, zero_fn, mm_fn, scale_fn):
        for _ in range_(D_div_n):
            ec = of_c.acquire(1)
            zero_fn(ec)
            for _ in range_(S_div_k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mm_fn(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            scale_fn(ec, D)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), zero_k, mm_k, sc_k])

    at = TensorAccessPattern((M, S), 0, [D_div_n, S_div_k, m, k], [0, k, S, 1])
    bt = TensorAccessPattern((S, D), 0, [D_div_n, S_div_k, k, n], [n, k * D, D, 1])
    ct = TensorAccessPattern((M, D), 0, [1, D_div_n, m, n], [m * D, n, D, 1])

    rt = Runtime()
    with rt.sequence(
        np.ndarray[(M, S), np.dtype[el]],
        np.ndarray[(S, D), np.dtype[el]],
        np.ndarray[(M, D), np.dtype[el]],
    ) as (sc, v, o):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(fifo_A.prod(), sc, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), v, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), o, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


# ============================================================================
# MLIR merging helpers
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
    """Rename device-level SSA values to avoid collisions across kernels."""
    result = []
    for line in decls_and_cores.split("\n"):
        if "func.func private" in line:
            # Function declarations are deduplicated globally; skip here.
            continue
        # Rename named SSA values like %logical_core -> %logical_core_qc
        line = re.sub(r"(%[a-zA-Z_][a-zA-Z0-9_]*)", lambda m: f"{m.group(1)}_{prefix}", line)
        # Rename numeric SSA values like %0 -> %v0_qc
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


def name_runtime_sequence(body: str, seq_name: str) -> str:
    """Give the first aie.runtime_sequence a symbol name."""
    return re.sub(
        r"aie\.runtime_sequence\s*\(",
        f"aie.runtime_sequence @{seq_name}(",
        body,
        count=1,
    )


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
    gemm_specs = [
        ("qc", M_FIX, 4096, D_FIX),
        ("kvc", M_FIX, 4096, K_LATENT),   # kv_latent dim = 512
        ("oa", M_FIX, D_FIX, 4096),
        ("ob", M_FIX, D_FIX, 4096),
        ("wq_b", M_FIX, D_FIX, 2048),   # host tiles N=32768 as 2048 chunks
        ("k_pe", M_FIX, 4096, 2048),    # host tiles N=4096 as 2048 chunks
    ]

    kernel_mlirs = OrderedDict()

    for name, m, k, n in gemm_specs:
        print(f"Generating {name} MLIR ({m}x{k}x{n})...", flush=True)
        t0 = time.perf_counter()
        design = gemm_kernel.specialize(name=name, M=m, K=k, N=n, el=bfloat16)
        mlir = design.as_mlir()
        kernel_mlirs[name] = mlir
        print(f"  {len(mlir)} chars ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Generating qk MLIR...", flush=True)
    t0 = time.perf_counter()
    design = qk_kernel.specialize(M=M_LATENT, K_dim=K_LATENT, S=S_MAX, el=bfloat16)
    kernel_mlirs["qk"] = design.as_mlir()
    print(f"  {len(kernel_mlirs['qk'])} chars ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Generating sv MLIR...", flush=True)
    t0 = time.perf_counter()
    design = sv_kernel.specialize(M=M_LATENT, S=S_MAX, D=D_LATENT, el=bfloat16)
    kernel_mlirs["sv"] = design.as_mlir()
    print(f"  {len(kernel_mlirs['sv'])} chars ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("\nMerging into unified device...")

    # Collect raw bodies and split each into decls/runtime-sequence
    kernel_parts = OrderedDict()
    for name, full_mlir in kernel_mlirs.items():
        body = extract_device_body(full_mlir)
        decls, rt_arg_decls, rt_arg_names, rt_body = extract_runtime_sequence(body)
        kernel_parts[name] = (decls, rt_arg_decls, rt_arg_names, rt_body)
        print(f"  {name}: decls={len(decls)} chars, rt_args={len(rt_arg_decls)}")

    # Deduplicate func.func private declarations
    all_raw_bodies = [decls for decls, _, _, _ in kernel_parts.values()]
    func_decls = deduplicate_func_decls(all_raw_bodies)

    # Rename SSA values per kernel and build one named runtime_sequence per kernel
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
    """Compile the shared C++ tile kernels once; aiecc links one object."""
    from aie.utils.compile import compile_cxx_core_function

    out_o = PROJ_ROOT / "fst_mla_unified_kernels.o"
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
    """Point every func.func private link_with at the single shared object."""
    return re.sub(
        r'attributes \{link_with = "[^"]+\.o"\}',
        'attributes {link_with = "fst_mla_unified_kernels.o"}',
        mlir_text,
    )


def compile_unified(mlir_text: str) -> Path:
    compile_external_objects()

    mlir_path = PROJ_ROOT / "fst_mla_unified.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)
    print(f"\nWrote unified MLIR: {mlir_path}")

    peano_dir = os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie")
    cmd = [
        "aiecc",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--no-xchesscc",
        "--no-xbridge",
        f"--peano={peano_dir}",
        str(mlir_path),
        "-o",
        str(PROJ_ROOT / "fst_mla_unified"),
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

    # aiecc writes main.xclbin in cwd regardless of -o base
    main_xclbin = PROJ_ROOT / "main.xclbin"
    xclbin_path = PROJ_ROOT / "fst_mla_unified.xclbin"
    if main_xclbin.exists():
        shutil.copy2(str(main_xclbin), str(xclbin_path))
        main_xclbin.unlink()
    else:
        raise RuntimeError("main.xclbin not produced by aiecc")

    print(f"SUCCESS: {xclbin_path}: {xclbin_path.stat().st_size}B ({dt:.1f}s)")
    return xclbin_path


def extract_per_kernel_insts():
    """Extract named runtime_sequence instruction binaries.

    aiecc produces a default NPU instruction binary for the whole module, but we
    need one payload per kernel.  The intermediate MLIR that still contains the
    named `aiex.runtime_sequence` ops is dumped by --dump-intermediates; we run
    aie-translate on that file with --aie-sequence-name.
    """
    prj = PROJ_ROOT / "fst_mla_unified.mlir.prj"
    if not prj.exists():
        raise RuntimeError(f"aiecc project directory not found: {prj}")

    # aie-translate needs the fully NPU-lowered MLIR (main_npu_lowered.mlir).
    # input_with_addresses.mlir is too early and produces only a 16-byte header.
    lowered_mlir = prj / "main_npu_lowered.mlir"

    print(f"\nExtracting per-kernel insts from {lowered_mlir}")

    kernel_names = ["qc", "kvc", "qk", "sv", "oa", "ob", "wq_b", "k_pe"]
    for name in kernel_names:
        out_bin = PROJ_ROOT / f"fst_mla_{name}_insts.bin"
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
    print("=== Unified MLA Generator ===\n")
    t_total = time.perf_counter()

    mlir_text = generate_unified_mlir()
    mlir_text = unify_link_with(mlir_text)
    mlir_path = PROJ_ROOT / "fst_mla_unified.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)
    print(f"\nUnified MLIR: {len(mlir_text)} chars -> {mlir_path}")

    compile_unified(mlir_text)
    extract_per_kernel_insts()

    print(f"\n=== Total time: {time.perf_counter() - t_total:.1f}s ===")


if __name__ == "__main__":
    main()
