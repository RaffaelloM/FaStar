#!/usr/bin/env python3
"""FaStar MULTI-CORE BATCHED expert GEMM — 3x-stride (gate|up|down) layout.

Collapses the draft FFN's per-expert gate/up/down GEMM loop (6 routed experts ×
1 dispatch/op = 6 dispatches/op) into ONE multi-core dispatch where 6 AIE cores
each GEMM one expert's slice of a batched B BO — with NO host memcpy and NO
dequant-split.

KEY LAYOUT (vs compile_ffn_multicore.py): the dequant kernel emits each expert's
weights as gate|up|down contiguous (3 slices of proj_elems = N*K each).  We batch
the 6 experts' FULL dequant output into one big BO, expert i at element offset
i*3*N*K, and the multicore kernel's per-worker B TAP uses a 3x stride
(view [E*3N, K], offset i*3*N*K) so worker i reads expert i's GATE slice in place.
The engine calls the SAME xclbin/insts for "up" by passing the batched BO
sub-buffer at base offset +N*K (so worker i reads expert i's UP slice), exactly as
the single-core gemm/gemm2 share one xclbin with different B sub-buffers today.
Down uses its own xclbin (different shape) with base offset +2*proj_elems.

Layout (b_col_maj, matching fst_expert_gemm_vectorized.py b_col_maj=1):
    A : [M, K]              shared activation (replicated to all 6 workers)
    B : [E, 3, N, K] flat   batched dequanted weights, expert i gate at i*3*N*K,
                            up at i*3*N*K + N*K, down at i*3*N*K + 2*N*K
    C : [E*M, N] flat       batched output, expert i at rows [i*M, (i+1)*M)

Two xclbins compiled here:
  fst_expert_gemm_vec_mc.xclbin  gate/up  M=16 K=4096 N=2048 tile 16x128x64
  fst_expert_gemm_down_mc.xclbin down    M=16 K=2048 N=4096 tile 16x128x64

Engine calls each TWICE conceptually (gate vs up via sub-buffer base shift for
vec; down once).  +1 hw_context for the down xclbin (vec_mc already counted).
"""
import os, sys, shutil, time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
set_current_device(NPU2())

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
NUM_EXPERTS = 6
EL = bfloat16


@iron.jit
def expert_gemm_mc_op(A: In, B: In, C: Out, *,
                      M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
                      E: CompileTime[int],
                      m: CompileTime[int], k: CompileTime[int], n: CompileTime[int],
                      el: CompileTime[type],
                      a_batched: CompileTime[bool] = False):
    """Multi-core batched expert GEMM, 3x-stride (gate|up|down) B layout.

    A : a_batched=False (gate/up): [M, K] shared -> replicated to all workers
        a_batched=True  (down):     [E*M, K] per-worker -> worker i reads
        rows [i*M,(i+1)*M]  (down's A = silu*up DIFFERS per expert; gate/up A = h).
    B : [E*3N, K] view    expert i GATE at offset i*3*N*K (engine shifts sub-buffer
                         base by +N*K for up, +2*N*K for down-via-down-xclbin)
    C : [E*M, N]         expert i at rows [i*M, (i+1)*M]
    b_col_maj (B [N,K] per expert, transposed-B GEMM).
    """
    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n,
                    input_dtype=el, output_dtype=el,
                    b_col_maj=True, use_chess=False,
                    emulate_bf16_mmul_with_bfp16=False)
    z = mm.zero
    r, s, t = mm.mac_dims
    assert m % r == 0 and k % s == 0 and n % t == 0

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    a_ty = np.ndarray[(m, k), np.dtype[el]]
    b_ty = np.ndarray[(k, n), np.dtype[el]]
    c_ty = np.ndarray[(m, n), np.dtype[el]]

    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]   # b_col_maj
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

    def make_fifos(i):
        inA = ObjectFifo(a_ty, name=f"mc_inA_{i}", depth=2)
        inB = ObjectFifo(b_ty, name=f"mc_inB_{i}", depth=2)
        memC = ObjectFifo(c_ty, name=f"mc_memC_{i}", depth=2)
        memA = inA.cons().forward(name=f"mc_memA_{i}", dims_to_stream=a_dims)
        memB = inB.cons().forward(name=f"mc_memB_{i}", dims_to_stream=b_dims)
        outC = memC.cons().forward(name=f"mc_outC_{i}", dims_to_stream=c_dims)
        return inA, inB, memC, memA, memB, outC

    fifos = [make_fifos(i) for i in range(E)]

    def make_core(idx):
        def core_fn(of_a, of_b, of_c, zero_k, mm_k):
            loop = range_(tiles) if tiles > 1 else range(1)
            for _ in loop:
                ec = of_c.acquire(1); zero_k(ec)
                inner = range_(K_div_k) if K_div_k > 1 else range(1)
                for _ in inner:
                    ea = of_a.acquire(1); eb = of_b.acquire(1)
                    mm_k(ea, eb, ec)
                    of_a.release(1); of_b.release(1)
                of_c.release(1)
        return core_fn

    workers = [Worker(make_core(i),
                     [fifos[i][3].cons(), fifos[i][4].cons(), fifos[i][2].prod(),
                      z, mm],
                     stack_size=0xD00) for i in range(E)]

    # A: gate/up -> replicated [M,K] (shared h, offset 0 all workers).
    #     down   -> per-worker [E*M,K] (worker i reads its own 16 rows of silu*up).
    if a_batched:
        taps_A = [TensorAccessPattern((E * M, K), i * M * K,
                    [N_div_n, K_div_k, m, k], [0, k, K, 1]) for i in range(E)]
    else:
        taps_A = [TensorAccessPattern((M, K), 0,
                    [N_div_n, K_div_k, m, k], [0, k, K, 1]) for _ in range(E)]

    # B: 3x-stride (gate|up|down).  Worker i reads expert i's GATE slice at
    # offset i*3*N*K, walking [N,K] (N_div_n*N rows = exactly the gate block,
    # so it never bleeds into the up/down slices).  Engine shifts the sub-buffer
    # base for up (+N*K) and the down xclbin uses +2*N*K base.
    taps_B = [TensorAccessPattern(
                (E * 3 * N, K), i * 3 * N * K,
                [N_div_n, K_div_k, n, k],
                [n * K, k, K, 1]) for i in range(E)]

    # C: batched [E*M, N], worker i at row offset i*M.
    taps_C = [TensorAccessPattern(
                (E * M, N), i * M * N,
                [1, N_div_n, m, n],
                [m * N, n, N, 1]) for i in range(E)]

    A_ty = np.ndarray[(E * M * K,) if a_batched else (M * K,), np.dtype[el]]
    B_ty = np.ndarray[(E * 3 * K * N,), np.dtype[el]]   # 3x batched B
    C_ty = np.ndarray[(E * M * N,), np.dtype[el]]

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a, b, c):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(E):
            rt.fill(fifos[i][0].prod(), a, tap=taps_A[i], task_group=tg)
        for i in range(E):
            rt.fill(fifos[i][1].prod(), b, tap=taps_B[i], task_group=tg)
        for i in range(E):
            rt.drain(fifos[i][5].cons(), c, tap=taps_C[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def compile_one(name, M, K, N, tile_m, tile_k, tile_n, a_batched):
    print(f"\n=== Compile {name}: M={M} K={K} N={N} tile={tile_m}x{tile_k}x{tile_n} a_batched={a_batched} ===")
    t0 = time.perf_counter()
    out_xcl = PROJ_ROOT / f"{name}.xclbin"
    out_ins = PROJ_ROOT / f"{name}_insts.bin"
    try:
        xclbin, insts = expert_gemm_mc_op.specialize(
            M=M, K=K, N=N, E=NUM_EXPERTS,
            m=tile_m, k=tile_k, n=tile_n, el=EL, a_batched=a_batched).compile()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"✗ {name} COMPILE FAILED: {type(e).__name__}: {e}")
        return 1
    shutil.copy(xclbin, out_xcl)
    shutil.copy(insts, out_ins)
    dt = time.perf_counter() - t0
    print(f"✓ {name} COMPILED in {dt:.1f}s  xclbin={out_xcl.stat().st_size}B insts={out_ins.stat().st_size}B")
    return 0


def main():
    rc = 0
    # gate/up: M=16 K=4096 N=2048 tile 16x128x64 (A replicated)
    rc |= compile_one("fst_expert_gemm_vec_mc", 16, 4096, 2048, 16, 128, 64, False)
    # down:    M=16 K=2048 N=4096 tile 16x128x64 (A per-worker, E*M=96 rows)
    rc |= compile_one("fst_expert_gemm_down_mc", 16, 2048, 4096, 16, 128, 64, True)
    if rc == 0:
        print("\nNOTE: +1 hw_context for fst_expert_gemm_down_mc vs the single-core")
        print("down.  Engine retires the single-core gemm_down + dequant + gemm_vec")
        print("contexts when wiring these (net contexts unchanged or fewer).")
    return rc


if __name__ == "__main__":
    sys.exit(main())