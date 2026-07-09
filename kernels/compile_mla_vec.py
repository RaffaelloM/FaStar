#!/usr/bin/env python3
"""FaStar MLA GEMMs — CANONICAL IRON single_core (kernels.mm + zero), VECTORIZED.

Replicates the proven FFN template (`fst_expert_gemm_vectorized.py`) but for the
7 MLA shapes.  The FFN proved this exact pattern (canonical kernels.mm + zero,
b_col_maj, no SET-then-ACC) is correct (probe 1.0).  MLA M=8 cannot use the 4x4
vectorized expansion (m%16==0 required), so we pad M=8 -> M=16 in the C++
dispatch and compile the projections at M=16.

Per-kernel layout (b_col_maj reads B stored [N,K], i.e. native weight form —
NO engine transpose needed for the projections whose weights are pre-transposed
to [K,N] on load; b_col_maj=1 means the engine should NOT transpose and instead
store/keep B as [N,K]):

  qc   : M=16, K=4096, N=1024  b_col_maj=1   (wq_a  native [N=1024,K=4096])
  kvc  : M=16, K=4096, N=512   b_col_maj=1   (wkv   native [N=512, K=4096])
  qk   : M=16, K=512,  N=128   b_col_maj=1   (kv_latent [S,512]=[N=S,K=512] direct)
  sv   : M=16, K=128,  N=512   b_row_maj=0   (kv_latent [S,512]=[K=S,N=512] direct)
  oa   : M=16, K=4096, N=1024  b_col_maj=1   (wo_a_g native [N=1024,K=4096])
  ob   : M=16, K=8192, N=2048  b_col_maj=1   (wo_b  native [N=4096,K=8192], engine 2 calls)
  wq_b : M=16, K=1024, N=2048  b_col_maj=1   (wq_b native [N=32768,K=1024], engine 16 calls)

qk/sv are compiled at M=16 (one M-tile, the engine dispatches M_LAT/16 tiles;
the probe verifies the single-tile shape; the engine M-tiles by re-calling with
A-row offsets).  This keeps the worker loop count identical per kernel family.

One-xclbin trick: every kernel is compiled standalone (own xclbin+insts), but we
copy the FIRST (qc) xclbin as fst_mla_unified.xclbin, mirroring the scalar
compile_mla_unified.py.  The probe tests the unified xclbin against EACH shape's
insts — if all pass, the engine registers all MLA kernels against the one
unified xclbin (1 hw_context, total stays at 5).  If any fail, separate xclbins
are needed (see fallback note at bottom).
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

set_current_device(NPU2())


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def fst_mla_gemm_vec(
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


# Consolidated to 4 xclbins (hw_context budget: 4 existing + 4 MLA = 8, the hard
# AMDXDNA limit).  The vectorized worker loop `range_(tiles)` is CompileTime-baked
# so distinct shapes need distinct xclbins (verified: qc vs kvc ELFs differ).
# Consolidation map (all M=16, m=16, M_div_m=1 — the proven FFN pattern; the
# engine M-tiles projections <=2 calls and qk/sv M_LAT/16 calls):
#   mla_qc   (16,4096,1024) b_col_maj -> qc, oa (native), kvc (pad N 512->1024)
#   mla_wqb  (16,1024,2048) b_col_maj -> wq_b (engine N-tiles 16x for N=32768)
#   mla_ob   (16,8192,2048) b_col_maj -> ob  (engine N-tiles 2x  for N=4096)
#   mla_qksv (16, 512, 512) b_row_maj -> qk (pad N S->512, B=kvT) + sv (pad K S->512, B=kv)
# Tile choices honor NPU2 limits:
#   - BD step <=64 per dim: N_div_n<=64, K_div_k<=128.
#   - AIE program mem ~64KB: (m/16)*(n/16)*(k/8)*16 <=256.
MLA_DIMS = [
    # M=16 standardization: wqb problem-M=16, tile m=16 (1 M-tile).  Narrowed to
    # wqb-only so this script does NOT overwrite fst_mla_{qc,ob,qksv}.xclbin --
    # qc/ob are deployed from compile_mla_qc_trial.py (K=1024 host-K-split) and
    # qksv stays M=2048 (out of scope).  (This script's qc/ob/qksv entries were
    # stale: qc K=4096, ob K=8192, neither matching the engine's K_c=1024.)
    ("wqb",  16, 1024, 2048, 16, 32, 32, 1),    # M=16, m=16 unroll 128; engine N-tiles 16x for N=32768
]


def aot_compile() -> None:
    for (name, M, K, N, tm, tk, tn, bcm) in MLA_DIMS:
        xclbin, insts = fst_mla_gemm_vec.specialize(
            M=M, K=K, N=N, m=tm, k=tk, n=tn, b_col_maj=bcm).compile()
        out_xcl = f"fst_mla_{name}.xclbin"
        out_ins = f"fst_mla_{name}_insts.bin"
        shutil.copy(xclbin, out_xcl)
        shutil.copy(insts, out_ins)
        print(f"  [{name}] {M}x{K}x{N} tile {tm}x{tk}x{tn} bcm={bcm} "
              f"-> {out_xcl} ({os.path.getsize(out_xcl)}B) + {out_ins} "
              f"({os.path.getsize(out_ins)}B)")
    print("\n4 MLA xclbins (qc, wqb, ob, qksv). Engine registers 4 MLA hw_contexts;")
    print("total = 4 (FFN/dequant/ew) + 4 MLA = 8 (the AMDXDNA hard limit).")


if __name__ == "__main__":
    aot_compile()