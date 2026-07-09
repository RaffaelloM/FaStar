#!/usr/bin/env python3
"""Compile ALL MLA kernels into a single fst_mla_unified.xclbin.

Each kernel is compiled in a separate subprocess (IRON JIT segfaults with
multiple @iron.jit functions). The raw MLIR from each is combined with
proper symbol renaming, deduplication, and a unified runtime_sequence.

Dimensions:
  K_FULL_DIM = 32768 (= nope_dim + MLA_N_HEADS * rope_dim = 28672 + 4096)
  S_MAX = 512 (max sequence length for QK/SV)
  D_FIX = 1024 (MLA_KV_DECOMP)
"""
import os, sys, time, shutil, subprocess, json, tempfile
from pathlib import Path
from collections import OrderedDict
import re

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
M_FIX = 8
K_FULL_DIM = 32768
D_FIX = 1024
S_MAX = 512

GEMM_KERNELS = [
    ("qc",  8, 4096, 1024),
    ("kvc", 8, 4096, 512),
    ("oa",  8, 4096, 4096),
    ("ob",  8, 4096, 4096),
    ("wq_b", 8, 1024, 2048),   # Q expansion tile (full N=32768 tiled by host)
    ("k_pe", 8, 4096, 2048),   # k_pe projection tile (full N=4096 tiled by host)
]

def gen_gemm_subprocess_script(name, m, k, n, out_file):
    proj = str(PROJ_ROOT)
    return f'''
import os, sys, json, shutil
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
sys.path.insert(0, "{proj}/Source/IRON-devel")
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
set_current_device(NPU2())
TILE_M, TILE_K, TILE_N = 8, 64, 64
@iron.jit
def gemm_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N
    from aie.iron import kernels
    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=el, output_dtype=el, vectorized=False)
    z = mm.zero
    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="{name}_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="{name}_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="{name}_Cout", depth=2)
    def core(of_a, of_b, of_c, zero_k, mm_k):
        for _ in range_(M//m * N//n):
            ec = of_c.acquire(1); zero_k(ec)
            for _ in range_(K//k):
                ea = of_a.acquire(1); eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1); of_b.release(1)
            of_c.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm])
    at = TensorAccessPattern((M,K),0,[N//n,K//k,m,k],[0,k,K,1])
    bt = TensorAccessPattern((K,N),0,[N//n,K//k,k,n],[n,k*N,N,1])
    ct = TensorAccessPattern((M,N),0,[1,N//n,m,n],[m*N,n,N,1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(M,K), np.dtype[el]], np.ndarray[(K,N), np.dtype[el]], np.ndarray[(M,N), np.dtype[el]]) as (a,b,c):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()
design = gemm_op.specialize(M={m}, K={k}, N={n}, el=bfloat16)
mlir = design.as_mlir()
with open("{out_file}", "w") as f:
    f.write(mlir)
# Compile to generate .o files, then copy them and the insts.bin to project root
xclbin, insts = design.compile()
cache_dir = Path(xclbin).parent
proj = Path("{proj}")
copied_o = []
for o in cache_dir.glob("*.o"):
    dst = proj / o.name
    if not dst.exists():
        shutil.copy2(str(o), str(dst))
    copied_o.append(str(dst.name))
insts_dst = proj / f"fst_mla_{name}_insts.bin"
shutil.copy2(str(insts), str(insts_dst))
print(json.dumps({{"status": "ok", "chars": len(mlir), "o_files": copied_o, "insts": str(insts_dst.name)}}))
'''

def gen_qk_subprocess_script(m, k_full, s, out_file):
    proj = str(PROJ_ROOT)
    return f'''
import os, sys, json, shutil
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
sys.path.insert(0, "{proj}/Source/IRON-devel")
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
set_current_device(NPU2())
PROJ_ROOT = Path("{proj}")
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
INCLUDE_DIRS = [str(AIE_KERNEL_DIR.parent), str(PROJ_ROOT)]
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
        include_dirs=INCLUDE_DIRS)
    z = ExternalFunction("fst_qk_zero",
        source_file=str(PROJ_ROOT / "fst_qk_zero.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS)
    scale_k = ExternalFunction("fst_qk_scale",
        source_file=str(PROJ_ROOT / "fst_qk_scale.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]], np.int32],
        include_dirs=INCLUDE_DIRS)
    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="qk_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="qk_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="qk_Cout", depth=2)
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
design = qk_op.specialize(M={m}, K_full={k_full}, S={s}, el=bfloat16)
mlir = design.as_mlir()
with open("{out_file}", "w") as f:
    f.write(mlir)
# Compile to generate .o files
xclbin, insts = design.compile()
cache_dir = Path(xclbin).parent
proj = Path("{proj}")
copied_o = []
for o in cache_dir.glob("*.o"):
    dst = proj / o.name
    if not dst.exists():
        shutil.copy2(str(o), str(dst))
    copied_o.append(str(dst.name))
insts_dst = proj / "fst_mla_qk_insts.bin"
shutil.copy2(str(insts), str(insts_dst))
print(json.dumps({{"status": "ok", "chars": len(mlir), "o_files": copied_o, "insts": str(insts_dst.name)}}))
'''

def gen_sv_subprocess_script(m, s, d, out_file):
    proj = str(PROJ_ROOT)
    return f'''
import os, sys, json, shutil
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
sys.path.insert(0, "{proj}/Source/IRON-devel")
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
set_current_device(NPU2())
PROJ_ROOT = Path("{proj}")
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
INCLUDE_DIRS = [str(AIE_KERNEL_DIR.parent), str(PROJ_ROOT)]
TILE_M, TILE_K, TILE_N = 8, 64, 64
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
        include_dirs=INCLUDE_DIRS)
    z = ExternalFunction("fst_sv_zero",
        source_file=str(PROJ_ROOT / "fst_sv_zero.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]]],
        include_dirs=INCLUDE_DIRS)
    scale_k = ExternalFunction("fst_sv_scale",
        source_file=str(PROJ_ROOT / "fst_sv_scale.cc"),
        arg_types=[np.ndarray[(m*n,), np.dtype[el]], np.int32],
        include_dirs=INCLUDE_DIRS)
    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="sv_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="sv_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="sv_Cout", depth=2)
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
design = sv_op.specialize(M={m}, S={s}, D={d}, el=bfloat16)
mlir = design.as_mlir()
with open("{out_file}", "w") as f:
    f.write(mlir)
# Compile to generate .o files
xclbin, insts = design.compile()
cache_dir = Path(xclbin).parent
proj = Path("{proj}")
copied_o = []
for o in cache_dir.glob("*.o"):
    dst = proj / o.name
    if not dst.exists():
        shutil.copy2(str(o), str(dst))
    copied_o.append(str(dst.name))
insts_dst = proj / "fst_mla_sv_insts.bin"
shutil.copy2(str(insts), str(insts_dst))
print(json.dumps({{"status": "ok", "chars": len(mlir), "o_files": copied_o, "insts": str(insts_dst.name)}}))
'''


def run_subprocess(script, label):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(script)
        f.flush()
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJ_ROOT)
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-500:]}")
            return None
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {"status": "ok"}
    finally:
        os.unlink(tmp_path)


def extract_device_body(mlir_text):
    """Extract the aie.device body, including runtime_sequence and all cores."""
    lines = mlir_text.split('\n')
    device_start = -1
    depth = 0
    for i, line in enumerate(lines):
        if 'aie.device(npu2)' in line or 'aie.device(npu1)' in line:
            device_start = i
            depth = line.count('{') - line.count('}')
            continue
        if device_start >= 0:
            depth += line.count('{') - line.count('}')
            if depth == 0:
                return '\n'.join(lines[device_start+1:i])
    return ""


def extract_runtime_sequence(body):
    """Split device body into (decls_and_cores, rt_arg_decls, rt_arg_names, rt_body).

    decls_and_cores: everything before aie.runtime_sequence (tiles, objectfifos, core blocks)
    rt_arg_decls: list of (arg_name, arg_type) e.g. [("%arg0", "memref<8x4096xbf16>"), ...]
    rt_arg_names: list of just the names ["%arg0", "%arg1", "%arg2"]
    rt_body: the DMA operations inside runtime_sequence (text)
    """
    idx = body.find('aie.runtime_sequence(')
    if idx < 0:
        return body, [], [], ""

    decls = body[:idx].rstrip()

    # Find args
    args_start = body.index('(', idx) + 1
    args_end = body.index(')', args_start)
    args_str = body[args_start:args_end]

    # Parse args: "%arg0: memref<8x4096xbf16>, %arg1: memref<4096x1024xbf16>"
    rt_arg_decls = []
    rt_arg_names = []
    for arg in args_str.split(','):
        arg = arg.strip()
        if ':' in arg:
            name, typ = arg.split(':', 1)
            rt_arg_decls.append((name.strip(), typ.strip()))
            rt_arg_names.append(name.strip())

    # Find runtime_sequence body (balanced braces)
    body_open = body.index('{', args_end)
    depth = 1
    i = body_open + 1
    while depth > 0 and i < len(body):
        if body[i] == '{':
            depth += 1
        elif body[i] == '}':
            depth -= 1
        i += 1
    rt_body = body[body_open + 1:i - 1].strip()

    return decls, rt_arg_decls, rt_arg_names, rt_body


def rename_decls_and_cores(decls_and_cores, prefix):
    """Rename device-level symbols and core-internal symbols for a kernel.
    Skip func.func private (handled separately). Don't rename func.call targets.
    """
    lines = decls_and_cores.split('\n')
    result = []
    seen_syms = {}

    for line in lines:
        if 'func.func private' in line:
            continue

        # ObjectFifo names (@qc_Ain, @kvc_Ain, etc.) are already unique across
        # kernels, and func.call targets are shared, so do not rename @ symbols.

        def rename_pct(m):
            name = m.group(1)
            return f"%{name}_{prefix}"

        line = re.sub(r'%([a-zA-Z_][a-zA-Z0-9_]*)', rename_pct, line)
        # Also rename numeric SSA values like %0, %1, %2 (aie.core results)
        # Use valid named SSA: %0 -> %v0_kernel, %1 -> %v1_kernel, etc.
        line = re.sub(r'%(\d+)\b', lambda m: f"%v{m.group(1)}_{prefix}", line)

        result.append(line)

    return '\n'.join(result)


def rename_rt_body(rt_body, original_arg_names, prefix):
    """Rename references inside runtime_sequence body:
    - @-prefixed ObjectFifo names get prefix
    - %arg0, %arg1, etc. get _suffix
    - %1, %2, %3 (numeric SSA task results) get _suffix
    """
    result = rt_body

    # ObjectFifo names (@qc_Ain etc.) are already unique; do not rename them.

    # Rename arg references: %arg0 -> %arg0_kernel, etc.
    for arg_name in original_arg_names:
        escaped = re.escape(arg_name)
        result = re.sub(escaped + r'\b', f"{arg_name}_{prefix}", result)

    # Rename numeric SSA values: %1 -> %v1_kernel, %2 -> %v2_kernel, etc.
    # These are DMA task results (aiex.dma_configure_task_for return values)
    result = re.sub(r'%(\d+)\b', lambda m: f"%v{m.group(1)}_{prefix}", result)

    return result


def deduplicate_func_decls(bodies_with_prefix):
    """Collect all func.func private declarations, deduplicate by signature."""
    seen = {}
    decls = []
    for prefix, body in bodies_with_prefix:
        for line in body.split('\n'):
            if 'func.func private' in line:
                m = re.search(r'func\.func\s+private\s+(@?\S+)\(([^)]*)\)', line)
                if m:
                    key = (m.group(1), m.group(2))
                    if key not in seen:
                        seen[key] = True
                        decls.append(line.strip())
    return '\n'.join(f'    {d}' for d in decls)


def main():
    print("=== Compile MLA Unified XCLBIN (Fixed Combination) ===\n")
    t_total = time.perf_counter()
    proj = str(PROJ_ROOT)

    # Phase 1: Generate each kernel's MLIR in a separate subprocess
    kernel_milrs = OrderedDict()

    for name, m, k, n in GEMM_KERNELS:
        print(f"Generating {name} ({m}x{k}x{n})...", flush=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False, dir='/tmp') as out_f:
            out_file = out_f.name
        src = gen_gemm_subprocess_script(name, m, k, n, out_file)
        info = run_subprocess(src, name)
        if info is None:
            print(f"  FAILED"); return
        with open(out_file) as f:
            kernel_milrs[name] = f.read()
        os.unlink(out_file)
        print(f"  {info.get('chars', '?')} chars")

    print(f"Generating qk ({M_FIX}x{K_FULL_DIM}x{S_MAX})...", flush=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False, dir='/tmp') as out_f:
        out_file = out_f.name
    src = gen_qk_subprocess_script(M_FIX, K_FULL_DIM, S_MAX, out_file)
    info = run_subprocess(src, "qk")
    if info is None:
        print("  FAILED"); return
    with open(out_file) as f:
        kernel_milrs["qk"] = f.read()
    os.unlink(out_file)
    print(f"  {info.get('chars', '?')} chars")

    print(f"Generating sv ({M_FIX}x{S_MAX}x{D_FIX})...", flush=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False, dir='/tmp') as out_f:
        out_file = out_f.name
    src = gen_sv_subprocess_script(M_FIX, S_MAX, D_FIX, out_file)
    info = run_subprocess(src, "sv")
    if info is None:
        print("  FAILED"); return
    with open(out_file) as f:
        kernel_milrs["sv"] = f.read()
    os.unlink(out_file)
    print(f"  {info.get('chars', '?')} chars")

    # Phase 2: Extract device bodies and split into parts
    print("\n=== Combining MLIR ===")
    kernel_parts = OrderedDict()
    for name, full_mlir in kernel_milrs.items():
        body = extract_device_body(full_mlir)
        if not body:
            print(f"  WARNING: empty body for {name}")
            continue
        decls, rt_arg_decls, rt_arg_names, rt_body = extract_runtime_sequence(body)
        kernel_parts[name] = (decls, rt_arg_decls, rt_arg_names, rt_body)
        print(f"  {name}: decls={len(decls)} chars, rt_args={len(rt_arg_decls)}, rt_body={len(rt_body)} chars")

    # Collect func declarations for dedup (from raw decls, before renaming)
    func_decl_map = {}
    for name, (decls, _, _, _) in kernel_parts.items():
        for line in decls.split('\n'):
            if 'func.func private' in line:
                m = re.search(r'func\.func\s+private\s+(@?\S+)\(([^)]*)\)', line)
                if m:
                    key = (m.group(1), m.group(2))
                    if key not in func_decl_map:
                        func_decl_map[key] = line.strip()

    # Rename device-level symbols in each kernel's decls+cores
    renamed_parts = []
    for name, (decls, _, _, _) in kernel_parts.items():
        renamed = rename_decls_and_cores(decls, name)
        renamed_parts.append(renamed)

    # Merge runtime_sequence: rename args and body per kernel
    all_rt_arg_decls = []
    all_rt_bodies = []
    for name, (_, rt_arg_decls, _, rt_body) in kernel_parts.items():
        # Rename arg declarations
        renamed_arg_decls = [(f"{n}_{name}", t) for n, t in rt_arg_decls]
        all_rt_arg_decls.extend(renamed_arg_decls)
        # Rename body references
        original_arg_names = [n for n, _ in rt_arg_decls]
        renamed_body = rename_rt_body(rt_body, original_arg_names, name)
        all_rt_bodies.append(renamed_body)

    # Phase 3: Build combined MLIR
    func_decls_str = '\n'.join(f'    {d}' for d in func_decl_map.values())
    parts_str = '\n'.join(renamed_parts)
    rt_args_str = ', '.join(f'{n}: {t}' for n, t in all_rt_arg_decls)
    rt_body_str = '\n'.join(all_rt_bodies)

    combined_mlir = f"""module {{
  aie.device(npu2) {{
    // === Shared function declarations ===
{func_decls_str}
    // === Kernel declarations and cores ===
{parts_str}
    // === Merged runtime sequence ===
    aie.runtime_sequence({rt_args_str}) {{
{rt_body_str}
    }}
  }}
}}"""

    print(f"\nCombined MLIR: {len(combined_mlir)} chars")

    mlir_path = PROJ_ROOT / "fst_mla_unified.mlir"
    with open(mlir_path, 'w') as f:
        f.write(combined_mlir)
    print(f"MLIR written to {mlir_path}")

    # Phase 4: Compile with aiecc
    print("\n=== Compiling with aiecc ===")
    peano_dir = os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie")
    cmd = [
        "aiecc",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--no-xchesscc",
        "--no-xbridge",
        f"--peano={peano_dir}",
        str(mlir_path),
        "-o", str(PROJ_ROOT / "fst_mla_unified")
    ]
    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t0

    if result.stdout:
        last_lines = result.stdout.strip().split('\n')[-10:]
        for l in last_lines:
            print(f"  stdout: {l}")
    if result.stderr:
        last_lines = result.stderr.strip().split('\n')[-10:]
        for l in last_lines:
            print(f"  stderr: {l}")

    xclbin_file = PROJ_ROOT / "fst_mla_unified.xclbin"
    main_xclbin = PROJ_ROOT / "main.xclbin"
    if main_xclbin.exists():
        shutil.copy2(str(main_xclbin), str(xclbin_file))
        xclbin_sz = xclbin_file.stat().st_size
        print(f"\nSUCCESS: {xclbin_file}: {xclbin_sz}B ({dt:.1f}s)")
        main_xclbin.unlink()
    else:
        print(f"\nFAILED - xclbin not created (return code {result.returncode})")

    total_dt = time.perf_counter() - t_total
    print(f"\n=== Total time: {total_dt:.1f}s ===")


if __name__ == "__main__":
    main()
