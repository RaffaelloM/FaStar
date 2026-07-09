#!/usr/bin/env python3
"""FaStar DEVICE-SIDE K-SPLIT GEMM (Step 1 of the BO-to-BO restoration).

The host K-split fix (Phase 14, npu_gemm_mla_vec) is correct but reads dequanted
weights back to the CPU and host-loops K -- 0.04 tok/s.  This compiler emits
kernels that stream the FULL K reduction on-device in <=32-step BD groups, so the
engine can run a single kernel call with B already on a device BO (BO-to-BO, no
host K-loop, no readback).

Mechanism (replaces group_tiler's single K_div_k-step BD):
  * K_div_k = K // k_tile  (e.g. qc K=4096, k=32 -> 128)
  * K_CHUNK = 32            (per-BD K-step count -- well under the NPU2 <=64 limit)
  * KC      = K_div_k // K_CHUNK  (number of K-chunks per output tile)
  * B TAPs: step_tiler((N,K),(n,k), tile_group_repeats=(1, K_CHUNK)) ->
            N_div_n * KC TAPs, ordered (n=0,kc=0),(n=0,kc=1),...,(n=0,kc=KC-1),
            (n=1,kc=0),...  i.e. n-OUTER, kc-INNER -- matches the worker's
            per-output-tile `range_(K_div_k)` acquire order.
  * A TAPs: step_tiler((M,K),(m,k), tile_group_repeats=(1, K_CHUNK)) ->
            M_div_m * KC TAPs.  A is reused across N-tiles; we re-fill it per
            (n_tile, kc) with pattern_repeat=1 (small re-DMA, ~1 MB) so the A
            FIFO order also matches n-outer/kc-inner.  (pattern_repeat=N_div_n
            would invert to kc-outer/n-inner and mismatch the worker.)
  * Runtime: for each M-tile, for each n_tile, for each kc, fill A[m,kc] + B[n,kc]
    in one task_group, finish_task_group immediately (BD reuse -> only 2 BDs live
    per chunk).  Drain C per M-tile-row (wait=True) as today.
  * Worker core_fn is UNCHANGED: `for tile in range_(tiles): zero; for _ in
    range_(K_div_k): acquire A,B; matmul`.  The K-chunking is purely a DMA-side
    concern -- the FIFO element sequence the worker consumes is identical to the
    non-chunked kernel, only the BDs that feed it are split into <=32-step groups.

Layout (b_col_maj=1 reads B stored [N,K], native weight form -- NO transpose).
"""
import os, shutil
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

K_CHUNK = 32   # per-BD K-step count (NPU2 per-BD <=64; 32 is the proven-safe zone)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def fst_mla_gemm_kchunk(
    A: In, B: In, C: Out, *,
    M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    m: CompileTime[int], k: CompileTime[int], n: CompileTime[int],
    b_col_maj: CompileTime[int] = 0):
    dtype_in = bfloat16
    dtype_out = bfloat16
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
    assert K_div_k % K_CHUNK == 0
    KC = K_div_k // K_CHUNK

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
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]  # b_row_maj
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

    # K-chunked tilers: <=K_CHUNK K-steps per BD, n-outer/kc-inner order.
    a_seq = TensorTiler2D.step_tiler(
        (M, K), (m, k), (1, K_CHUNK), prune_step=False)          # M_div_m*KC TAPs
    if bcm:
        b_seq = TensorTiler2D.step_tiler(
            (N, K), (n, k), (1, K_CHUNK), prune_step=False)      # N_div_n*KC TAPs
    else:
        b_seq = TensorTiler2D.step_tiler(
            (K, N), (k, n), (K_CHUNK, 1),
            tile_group_col_major=True, prune_step=False)         # b_row_maj, kc-outer/n-inner
    rows_per_block = 2
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
                    m_tile = row_base + tile_row
                    for n_tile in range(N_div_n):
                        for kc in range(KC):
                            rt.fill(inA.prod(), A, tap=a_seq[m_tile * KC + kc],
                                    task_group=tgs[-1])
                            rt.fill(inB.prod(), B, tap=b_seq[n_tile * KC + kc],
                                    task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index],
                         task_group=tgs[-1], wait=True)
                c_index += 1
                if len(tgs) > 1:
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        if tgs:
            rt.finish_task_group(tgs[-1])
    return Program(NPU2(), rt).resolve_program()


# Prototype: qc only (M=32, K=4096, N=1024, b_col_maj) -> fst_mla_qck.xclbin.
# K_div_k = 128, KC = 4 (four 32-step BDs per output tile).
JOBS = [
    ("qck", 32, 4096, 1024, 32, 32, 32, 1),
]
for (NAME, M, K, N, TM, TK, TN, BCM) in JOBS:
    print(f"compile {NAME}: M={M} K={K} N={N} tile {TM}x{TK}x{TN} bcm={BCM} "
          f"-> K_div_k={K//TK} KC={(K//TK)//K_CHUNK} (K_CHUNK={K_CHUNK})", flush=True)
    assert K % TK == 0 and M % TM == 0 and N % TN == 0 and (K // TK) % K_CHUNK == 0
    xclbin, insts = fst_mla_gemm_kchunk.specialize(
        M=M, K=K, N=N, m=TM, k=TK, n=TN, b_col_maj=BCM).compile()
    shutil.copy(xclbin, f"fst_mla_{NAME}.xclbin")
    shutil.copy(insts, f"fst_mla_{NAME}_insts.bin")
    print(f"  -> fst_mla_{NAME}.xclbin ({os.path.getsize(f'fst_mla_{NAME}.xclbin')}B)", flush=True)