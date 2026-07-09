#!/usr/bin/env python3
"""TRIAL: recompile qc/ob/wqb with output_dtype=float32 so the host K-split in
npu_gemm_mla_vec accumulates FP32 partials (no bf16 rounding between K-chunks).
The bf16-partial version caps at cos ~0.9986 (worst-element error ~3.3 under
K-chunk cancellation); fp32 partials should push cos >0.9999 vs the float32 ref.

The kernel's accfloat already computes each K-chunk partial in fp32; it only
rounds to bf16 on output.  output_dtype=float32 hands the fp32 partial straight
to the host, which sums them in fp32 and converts to bf16 once at the very end
(the engine's downstream still gets bf16).  Layout (b_col_maj, K_c=1024) is
UNCHANGED -- only the output dtype differs.

Writes fst_mla_{qc,ob,wqb}.xclbin + _insts.bin (overwrites).  Backups in /tmp.
"""
import os, shutil, importlib
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import (CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker,
                      kernels, str_to_dtype)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.utils import set_current_device
from aie.iron.device import NPU2

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
set_current_device(NPU2())


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def fst_mla_gemm_vec_fp32(
    A: In, B: In, C: Out, *,
    M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    m: CompileTime[int], k: CompileTime[int], n: CompileTime[int],
    b_col_maj: CompileTime[int] = 0):
    dtype_in = bfloat16
    dtype_out = np.float32           # <-- only change vs compile_mla_vec.py
    bcm = bool(b_col_maj)

    assert M % m == 0 and K % k == 0 and N % n == 0
    matmul_kernel = kernels.mm(dim_m=m, dim_k=k, dim_n=n,
                               input_dtype=dtype_in, output_dtype=dtype_out,
                               b_col_maj=bcm, use_chess=False,
                               emulate_bf16_mmul_with_bfp16=False)
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert m % r == 0 and k % s == 0 and n % t == 0

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    if bcm:
        b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
    else:
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, zero, matmul):
        loop = range_(tiles) if tiles > 1 else range(1)
        for _ in loop:
            elem_out = of_c.acquire(1)
            zero(elem_out)
            inner = range_(K_div_k) if K_div_k > 1 else range(1)
            for _ in inner:
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn,
                    [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
                    stack_size=0xD00)

    rows_per_block = 2
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n, prune_step=False)
    if bcm:
        b_tap = TensorTiler2D.group_tiler(
            (N, K), (n, k), (N_div_n, K_div_k), prune_step=False)[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K_div_k, N_div_n),
            tile_group_col_major=True, prune_step=False)[0]
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m, n), (rows_per_block // 2, N_div_n), prune_step=False)
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tgs = []
        for tile_row_block in range(iron.ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = (tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2)
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index],
                         task_group=tgs[-1], wait=True)
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    return Program(NPU2(), rt).resolve_program()


# Same shapes/sizes as compile_mla_qc_trial.py + wqb (K_c=1024, K_div_k<=32).
JOBS = [
    ("qc",  32, 1024, 1024, 16, 64, 32, 1),   # K=4096 host-looped in 4 chunks
    ("ob",  32, 1024, 2048, 16, 64, 32, 1),   # K=8192 host-looped in 8 chunks
    ("wqb", 32, 1024, 2048, 16, 64, 32, 1),   # K=1024 -> 1 chunk
]
for (NAME, M, K, N, TM, TK, TN, BCM) in JOBS:
    print(f"compile {NAME} fp32-out: M={M} K={K} N={N} tile {TM}x{TK}x{TN} "
          f"bcm={BCM} -> K_div_k={K//TK}", flush=True)
    assert K % TK == 0 and M % TM == 0 and N % TN == 0
    xclbin, insts = fst_mla_gemm_vec_fp32.specialize(
        M=M, K=K, N=N, m=TM, k=TK, n=TN, b_col_maj=BCM).compile()
    shutil.copy(xclbin, f"fst_mla_{NAME}.xclbin")
    shutil.copy(insts, f"fst_mla_{NAME}_insts.bin")
    print(f"  -> fst_mla_{NAME}.xclbin ({os.path.getsize(f'fst_mla_{NAME}.xclbin')}B)", flush=True)