#!/usr/bin/env python3
"""Compile all FFN kernels into a single fst_ffn_unified.xclbin.

This reduces hw_context usage from 5 to 1 for FFN operations.
Kernels included: gemm (gate), gemm_down (down), dequant, silu, mul
"""
import os, sys, time, shutil, subprocess
from pathlib import Path
from collections import OrderedDict
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
set_current_device(NPU2())

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
AIE_KERNEL_DIR = Path(config.cxx_header_path()) / "aie_kernels"
INCLUDE_DIRS = [str(AIE_KERNEL_DIR.parent)]

# ── Kernel definitions ─────────────────────────────────────────────────────

# --- Gate/Up GEMM (32x4096x2048) ---
TILE_M, TILE_K, TILE_N = 32, 64, 64

@iron.jit
def gate_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N
    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=el, output_dtype=el, vectorized=False)
    z = mm.zero
    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el)], name="gate_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="gate_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el)], name="gate_Cout", depth=2)
    def core(of_a, of_b, of_c, zero_k, mm_k):
        for _ in range_(M//m * N//n):
            ec = of_c.acquire(1); zero_k(ec)
            for _ in range_(K//k):
                ea = of_a.acquire(1); eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1); of_b.release(1)
            of_c.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm])
    at = TensorAccessPattern((M, K), 0, [N//n, K//k, m, k], [0, k, K, 1])
    bt = TensorAccessPattern((K, N), 0, [N//n, K//k, k, n], [n, k*N, N, 1])
    ct = TensorAccessPattern((M, N), 0, [1, N//n, m, n], [m*N, n, N, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(M, K), np.dtype[el]], np.ndarray[(K, N), np.dtype[el]], np.ndarray[(M, N), np.dtype[el]]) as (a, b, c):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# --- Down GEMM (32x2048x4096) ---
@iron.jit
def down_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N
    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=el, output_dtype=el, vectorized=False)
    z = mm.zero
    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el)], name="down_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el)], name="down_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="down_Cout", depth=2)
    def core(of_a, of_b, of_c, zero_k, mm_k):
        for _ in range_(M//m * N//n):
            ec = of_c.acquire(1); zero_k(ec)
            for _ in range_(K//k):
                ea = of_a.acquire(1); eb = of_b.acquire(1)
                mm_k(ea, eb, ec)
                of_a.release(1); of_b.release(1)
            of_c.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm])
    at = TensorAccessPattern((M, K), 0, [N//n, K//k, m, k], [0, k, K, 1])
    bt = TensorAccessPattern((K, N), 0, [N//n, K//k, k, n], [n, k*N, N, 1])
    ct = TensorAccessPattern((M, N), 0, [1, N//n, m, n], [m*N, n, N, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(M, K), np.dtype[el]], np.ndarray[(K, N), np.dtype[el]], np.ndarray[(M, N), np.dtype[el]]) as (a, b, c):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# --- Dequant V4 (multi-core) ---
BLOCK_BYTES = 17
BLOCK_ELEMS = 32
BLOCKS_PER_CALL = 4
DMA_IN_BYTES = BLOCKS_PER_CALL * BLOCK_BYTES
DMA_OUT_ELEMS = BLOCKS_PER_CALL * BLOCK_ELEMS
PROJ_BLOCKS = 262144
TOTAL_BLOCKS = 3 * PROJ_BLOCKS
NUM_COLS = 8
NUM_ROWS = 2
NUM_CORES = NUM_COLS * NUM_ROWS

@iron.jit
def dequant_op(input0: In, output: Out, *, total_blocks: CompileTime[int]):
    in_dtype = np.uint8
    out_dtype = bfloat16
    calls = (total_blocks // NUM_CORES) // BLOCKS_PER_CALL
    total_in_bytes = total_blocks * BLOCK_BYTES
    total_out_elems = total_blocks * BLOCK_ELEMS
    in_tensor_ty = np.ndarray[(total_in_bytes,), np.dtype[in_dtype]]
    out_tensor_ty = np.ndarray[(total_out_elems,), np.dtype[out_dtype]]
    in_call_ty = np.ndarray[(DMA_IN_BYTES,), np.dtype[in_dtype]]
    out_call_ty = np.ndarray[(DMA_OUT_ELEMS,), np.dtype[out_dtype]]

    dequant_kernel = ExternalFunction("fst_dequant_v4_4096",
        source_file="fst_dequant_v4_vectorized_kernel.cc",
        arg_types=[in_call_ty, out_call_ty],
        include_dirs=[config.cxx_header_path()],
        object_file_name="fst_dequant_v4_4096.o")

    f_in = [ObjectFifo(in_call_ty, name=f"deq_in_{i}", depth=2) for i in range(NUM_CORES)]
    f_out = [ObjectFifo(out_call_ty, name=f"deq_out_{i}", depth=2) for i in range(NUM_CORES)]

    def make_core_fn(idx):
        def core_fn(of_in, of_out, kernel):
            for _ in range_(calls):
                ei = of_in.acquire(1); eo = of_out.acquire(1)
                kernel(ei, eo)
                of_in.release(1); of_out.release(1)
        return core_fn

    workers = [Worker(make_core_fn(i), [f_in[i].cons(), f_out[i].prod(), dequant_kernel]) for i in range(NUM_CORES)]

    taps_in = [TensorAccessPattern(
        tensor_dims=(1, total_in_bytes),
        offset=i * calls * DMA_IN_BYTES,
        sizes=[8, 48, 32, DMA_IN_BYTES],
        strides=[48 * 32 * DMA_IN_BYTES, 32 * DMA_IN_BYTES, DMA_IN_BYTES, 1]
    ) for i in range(NUM_CORES)]

    taps_out = [TensorAccessPattern(
        tensor_dims=(1, total_out_elems),
        offset=i * calls * DMA_OUT_ELEMS,
        sizes=[8, 48, 32, DMA_OUT_ELEMS],
        strides=[48 * 32 * DMA_OUT_ELEMS, 32 * DMA_OUT_ELEMS, DMA_OUT_ELEMS, 1]
    ) for i in range(NUM_CORES)]

    rt = Runtime()
    with rt.sequence(in_tensor_ty, out_tensor_ty) as (A, C):
        rt.start(*workers); tg = rt.task_group()
        for i in range(NUM_CORES):
            rt.fill(f_in[i].prod(), A, tap=taps_in[i], task_group=tg)
        for i in range(NUM_CORES):
            rt.drain(f_out[i].cons(), C, tap=taps_out[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


# --- SiLU activation ---
@iron.jit
def silu_op(A: In, B: Out, *, N: CompileTime[int], el: CompileTime[type]):
    TILE = 1024
    silu_k = kernels.silu(tile_size=TILE)
    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el)], name="silu_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="silu_Bout", depth=2)
    def core(of_a, of_b, silu_fn):
        for _ in range_(N // TILE):
            ea = of_a.acquire(1); eb = of_b.acquire(1)
            silu_fn(ea, eb)
            of_a.release(1); of_b.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.prod(), silu_k])
    tap = TensorAccessPattern((N,), 0, [1, N // TILE, TILE], [0, TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]], np.ndarray[(N,), np.dtype[el]]) as (a, b):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=tap, task_group=tg)
        rt.drain(fifo_B.cons(), b, tap=tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# --- Element-wise multiply ---
@iron.jit
def mul_op(A: In, B: In, C: Out, *, N: CompileTime[int], el: CompileTime[type]):
    TILE = 1024
    mul_k = kernels.mul(tile_size=TILE, dtype=el, vectorized=True)
    fifo_A = ObjectFifo(np.ndarray[(TILE,), np.dtype[el)], name="mul_Ain", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="mul_Bin", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(TILE,), np.dtype[el]], name="mul_Cout", depth=2)
    def core(of_a, of_b, of_c, mul_fn):
        for _ in range_(N // TILE):
            ea = of_a.acquire(1); eb = of_b.acquire(1); ec = of_c.acquire(1)
            mul_fn(ea, eb, ec)
            of_a.release(1); of_b.release(1); of_c.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), mul_k])
    tap = TensorAccessPattern((N,), 0, [1, N // TILE, TILE], [0, TILE, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(N,), np.dtype[el]], np.ndarray[(N,), np.dtype[el]], np.ndarray[(N,), np.dtype[el]]) as (a, b, c):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=tap, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=tap, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


# ── MLIR generation and combination ─────────────────────────────────────────

def extract_device_body(mlir_text):
    """Extract the aie.device body content from MLIR text."""
    lines = mlir_text.split('\n')
    device_start = -1
    depth = 0
    for i, line in enumerate(lines):
        if 'aie.device(npu2)' in line:
            device_start = i
            depth = 0
        elif device_start >= 0:
            depth += line.count('{') - line.count('}')
            if depth > 0 and line.strip() == '}':
                body_lines = lines[device_start+1:i]
                return '\n'.join(body_lines)
    return ""

def generate_and_combine_mlir():
    """Generate MLIR for all kernels and combine into single module."""
    print("=== Generating MLIR for all FFN kernels ===\n")

    kernels_mlir = OrderedDict()

    # Gate: 32x4096x2048
    print("Generating Gate (32x4096x2048)...", flush=True)
    design = gate_op.specialize(M=32, K=4096, N=2048, el=bfloat16)
    kernels_mlir["gate"] = extract_device_body(design.as_mlir())
    print(f"  {len(kernels_mlir['gate'])} chars")

    # Down: 32x2048x4096
    print("Generating Down (32x2048x4096)...", flush=True)
    design = down_op.specialize(M=32, K=2048, N=4096, el=bfloat16)
    kernels_mlir["down"] = extract_device_body(design.as_mlir())
    print(f"  {len(kernels_mlir['down'])} chars")

    # Dequant: 786432 blocks
    print("Generating Dequant (786432 blocks)...", flush=True)
    design = dequant_op.specialize(total_blocks=TOTAL_BLOCKS)
    kernels_mlir["dequant"] = extract_device_body(design.as_mlir())
    print(f"  {len(kernels_mlir['dequant'])} chars")

    # SiLU: 65536
    print("Generating SiLU (65536)...", flush=True)
    design = silu_op.specialize(N=65536, el=bfloat16)
    kernels_mlir["silu"] = extract_device_body(design.as_mlir())
    print(f"  {len(kernels_mlir['silu'])} chars")

    # Mul: 4096
    print("Generating Mul (4096)...", flush=True)
    design = mul_op.specialize(N=4096, el=bfloat16)
    kernels_mlir["mul"] = extract_device_body(design.as_mlir())
    print(f"  {len(kernels_mlir['mul'])} chars")

    print(f"\nTotal kernels: {len(kernels_mlir)}")

    # Combine into single MLIR module
    print("\n=== Combining MLIR modules ===")
    combined_lines = ['module {', '  aie.device(npu2) {']

    for name, body in kernels_mlir.items():
        combined_lines.append(f'    // === Kernel: {name} ===')
        for line in body.split('\n'):
            if line.strip():
                combined_lines.append(f'    {line}')

    combined_lines.append('  }', '}')

    combined_mlir = '\n'.join(combined_lines)
    print(f"Combined MLIR: {len(combined_mlir)} chars")

    return combined_mlir


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Compile FFN Unified XCLBIN ===\n")
    t_total = time.perf_counter()

    # Generate combined MLIR
    combined_mlir = generate_and_combine_mlir()

    # Write MLIR to file
    mlir_path = PROJ_ROOT / "fst_ffn_unified.mlir"
    with open(mlir_path, 'w') as f:
        f.write(combined_mlir)
    print(f"\nMLIR written to {mlir_path}")

    # Compile using aiecc
    print("\n=== Compiling with aiecc ===")
    xclbin_path = PROJ_ROOT / "fst_ffn_unified.xclbin"

    cmd = [
        "aiecc",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        str(mlir_path),
        "-o", str(xclbin_path.with_suffix(''))
    ]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t0

    # Show output
    if result.stdout:
        print("STDOUT:", result.stdout[:2000])
    if result.stderr:
        print("STDERR:", result.stderr[:2000])

    # Check results
    xclbin_file = PROJ_ROOT / "fst_ffn_unified.xclbin"
    insts_file = PROJ_ROOT / "insts.bin"

    if xclbin_file.exists():
        xclbin_sz = xclbin_file.stat().st_size
        insts_sz = insts_file.stat().st_size if insts_file.exists() else 0
        print(f"\n✓ SUCCESS:")
        print(f"  {xclbin_file}: {xclbin_sz}B")
        if insts_file.exists():
            print(f"  {insts_file}: {insts_sz}B")
            # Copy to expected name
            shutil.copy2(insts_file, PROJ_ROOT / "fst_ffn_unified_insts.bin")
            print(f"  Copied to fst_ffn_unified_insts.bin")
    else:
        print(f"\n✗ FAILED - xclbin not created")
        print(f"Return code: {result.returncode}")

    total_dt = time.perf_counter() - t_total
    print(f"\n=== Total time: {total_dt:.1f}s ===")


if __name__ == "__main__":
    main()
