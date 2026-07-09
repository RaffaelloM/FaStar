#!/usr/bin/env python3
"""Compile unified MLA xclbins.

All MLA GEMMs (qc, kvc, qk, sv, oa, ob, wq_b) share the SAME AIE kernel pair:
a custom SET-then-ACC scalar bf16 matmul (fst_aie_mm.cc), NOT IRON's
kernels.mm.  The scalar kernels read A/B and write C in plain ROW-MAJOR
order, which matches the 4-dim row-major fill/drain TensorAccessPatterns
below.  Each kernel gets its own insts.bin with different (M,K,N) TAP
dimensions; all xclbins are byte-identical, so we copy the first as
fst_mla_unified.xclbin.

ROOT CAUSE OF THE ~3.7x / sign-flip NPU ERROR (fixed here): IRON's
matmul_scalar does C += A*B (read-modify-write) and relies on a separate
zero_scalar pass to clear the C ObjectFifo slot before the K-reduction.
IRON FORCES the C ObjectFifo to depth 2 (depth=3/8 requests are ignored),
so C slots are reused across N-tiles.  On AIE2 the read-modify-write at
n_local == 3 (mod 4) picks up the previous N-tile's residual in the reused
slot, producing a (1 + t//2)x error from N-tile 2 on (tile 2 = 2x, tile 4 =
3x, ...).  N=128 (N_div_n=2) passed because no slot reuse occurs; N>=192
failed.  The A-fill stride-0 theory was DISPROVEN (the kvcrep probe with
stride m*K gave the identical error).  The C-fifo depth theory was DISPROVEN
(depth=3/8 forced to 2, identical error).  The kernel math itself
(matmul_scalar/zero_scalar) is provably correct in isolation.

THE FIX: a custom fst_aie_mm.cc providing two sibling symbols compiled from
one .o:
  matmul_set_scalar_bf16_bf16 : C =  A*B  (SET, no read — overwrites residual)
  matmul_acc_scalar_bf16_bf16 : C += A*B  (ACC, identical to IRON matmul_scalar)
The worker calls SET for K-tile 0 and ACC for K-tiles 1+, and DOES NOT call
zero.  Because SET writes (not read-modify-writes) the C slot, the residual
in a reused slot is overwritten — no dependence on zero.  Verified by
fst_gemm_probe: A=ones, B=identity -> C all 1.0 for N=512 (was 2x/3x/4x
before); A=ones, B=ones -> C all 4096.  Same fix applies to FFN
(compile_ffn_unified.py) which uses the same 4-dim TAPs.

WHY SCALAR (not vectorized): kernels.mm vectorized selects the 4x4-expanded
bf16 microkernel (matmul_vectorized_4x4) which requires DIM_M % 16 == 0
(it expands z, z+1, z+2, z+3).  MLA needs M=8, which violates that and reads
out of bounds.  The scalar kernel reads A/B row-major (matching the fill) and
writes C row-major (matching the drain), so its math matches the CPU ref.
"""

import os, sys, time, shutil
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.iron.kernels._common import _make_extern
from pathlib import Path

set_current_device(NPU2())
PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

TILE_M, TILE_K, TILE_N = 8, 64, 64


@iron.jit
def mla_gemm_op(A: In, B: In, C: Out,
                *, M: CompileTime[int], K: CompileTime[int],
                  N: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N
    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n

    # Custom SET-then-ACC scalar bf16 matmul (fst_aie_mm.cc).  SET overwrites
    # the reused C slot (no read-modify-write), so no zero pass is needed and
    # no residual accumulates across N-tiles.  vectorized would need M%16==0.
    a_ty = np.ndarray[(m * k,), np.dtype[el]]
    b_ty = np.ndarray[(k * n,), np.dtype[el]]
    c_ty = np.ndarray[(m * n,), np.dtype[el]]
    flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}", "-Dbf16_bf16_ONLY"]
    src = str(PROJ_ROOT / "fst_aie_mm.cc")
    mm_set = _make_extern("matmul_set_scalar_bf16_bf16", src, [a_ty, b_ty, c_ty],
                          compile_flags=flags, use_chess=False,
                          shared_object_file_name="fst_aie_mm_setacc.o")
    mm_acc = _make_extern("matmul_acc_scalar_bf16_bf16", src, [a_ty, b_ty, c_ty],
                          compile_flags=flags, use_chess=False,
                          shared_object_file_name="fst_aie_mm_setacc.o")

    fifo_A = ObjectFifo(np.ndarray[(m * k,), np.dtype[el]], name="A", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k * n,), np.dtype[el]], name="B", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m * n,), np.dtype[el]], name="C", depth=2)

    def core(of_a, of_b, of_c, k_set, k_acc):
        for _ in range_(M_div_m * N_div_n) if (M_div_m * N_div_n) > 1 else range(1):
            ec = of_c.acquire(1)
            # K-tile 0: SET (overwrites any residual in the reused C slot).
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            k_set(ea, eb, ec)
            of_a.release(1)
            of_b.release(1)
            # K-tiles 1..end: ACC (accumulate into the SET result).
            for _ in range_(K_div_k - 1) if K_div_k > 1 else range(0):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                k_acc(ea, eb, ec)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), mm_set, mm_acc],
               stack_size=0xD00)

    # Row-major fill/drain TAPs — match the scalar kernel's row-major read/
    # write.  A [M,K] is shared across all N-tiles (N stride 0, like FFN);
    # B [K,N] shifts by n per N-tile and by k*N per K-tile; C [M,N] drains one
    # N-tile at a time.  These are the SAME TAPs compile_ffn_unified.py uses.
    if N_div_n <= 64:
        at = TensorAccessPattern((M, K), 0, [N_div_n, K_div_k, m, k], [0, k, K, 1])
        bt = TensorAccessPattern((K, N), 0, [N_div_n, K_div_k, k, n], [n, k * N, N, 1])
        ct = TensorAccessPattern((M, N), 0, [1, N_div_n, m, n], [m * N, n, N, 1])
    else:
        N_outer = 64
        N_inner = N_div_n // N_outer
        at = TensorAccessPattern((M, K), 0, [N_inner, N_outer, K_div_k, m], [0, 0, K, 1])
        bt = TensorAccessPattern((K, N), 0, [N_inner, N_outer, K_div_k, k], [N_outer * n, n, k * N, 1])
        ct = TensorAccessPattern((M, N), 0, [1, N_inner, N_outer, m * n], [m * N, N_outer * n, n, 1])

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


if __name__ == "__main__":
    # Dims (M, K, N) MUST match the engine's per-call GEMM shapes.  All kernels
    # use M=8 (M_PAD token tile).  K is the reduction dim, N the output dim.
    #   qk:  q[8,512]      @ k[512,128]   -> scores[8,128]   (K=kv_latent=512)
    #   sv:  scores[8,128] @ v[128,512]   -> out[8,512]      (N=head_dim=512)
    #   wq_b: qc[8,1024]   @ wq_b[1024,2048] -> q_full tile  (N=2048 tile)
    dims = [
        ("qc",  8, 4096, 1024),
        ("kvc", 8, 4096, 512),
        ("qk",  8, 512, 128),
        ("sv",  8, 128, 512),
        ("oa",  8, 1024, 4096),
        ("ob",  8, 1024, 4096),
        ("wq_b", 8, 1024, 2048),
    ]

    print("Compiling unified MLA kernels (SET-then-ACC scalar row-major)...", flush=True)
    xclbins = []
    for name, M, K, N in dims:
        t0 = time.perf_counter()
        spec = mla_gemm_op.specialize(M=M, K=K, N=N, el=bfloat16)
        xclbin, insts = spec.compile()
        tx = str(PROJ_ROOT / f"fst_mla_{name}.xclbin")
        ti = str(PROJ_ROOT / f"fst_mla_{name}_insts.bin")
        shutil.copy2(xclbin, tx)
        shutil.copy2(insts, ti)
        xclbins.append(tx)
        dt = time.perf_counter() - t0
        print(f"  {name}: {M}x{K}x{N} → {os.path.getsize(tx)}B xclbin + {os.path.getsize(ti)}B insts ({dt:.1f}s)", flush=True)

    shutil.copy2(xclbins[0], str(PROJ_ROOT / "fst_mla_unified.xclbin"))
    print(f"\nUnified MLA xclbin: fst_mla_unified.xclbin ({os.path.getsize(str(PROJ_ROOT / 'fst_mla_unified.xclbin'))} bytes)")