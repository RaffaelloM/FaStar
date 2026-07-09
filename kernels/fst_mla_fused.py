#!/usr/bin/env python3
# fst_mla_fused.py — Fused Q_compress + Q_expand (DMA replay, no self-loop)
#
# Core 1: hidden @ wq_a → q_latent [8,1024] (16 tiles, sub-tiled)
#         DMA replays hidden+wq_a 64× to feed all Q_expand N-tiles
# Core 2: q_latent @ wq_b → q_full [8,4096] (64 tiles, sub-tiled)
#         Direct core-to-core FIFO, no self-loop needed
#
# The q_latent scratch FIFO (depth=16) connects both cores.
# Core 2 reads each tile once per N-tile (64×16 = 1024 total reads).
# Core 1 produces each tile once per N-tile (64×16 = 1024 total writes).
# DMA on Core 1 replays hidden+wq_a 64× via stride=0 TAP.

import numpy as np, time, shutil
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import (CompileTime, In, Out, ObjectFifo, Program, Runtime,
                      Worker, kernels)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
set_current_device(NPU2())

M_TOK, K_HID, N_LAT = 8, 4096, 1024
N_CHUNK = 4096
m, k, n = 8, 64, 64
qc_n, qc_k = N_LAT // n, K_HID // k   # 16 tiles, 64 K-steps
qe_n, qe_k = N_CHUNK // n, N_LAT // k # 64 tiles, 16 K-steps
REPLAY = qe_n  # 64 — DMA replays hidden+wq_a this many times


@iron.jit
def fst_mla_qc_qe_fused(
    hidden: In, wq_a: In, wq_b: In, q_full: Out, *,
    M: CompileTime[int], Kc: CompileTime[int], Nc: CompileTime[int],
    Ke: CompileTime[int], Nch: CompileTime[int],
    el: CompileTime[type],
):
    mm_k = kernels.mm(dim_m=m, dim_k=k, dim_n=n,
                      input_dtype=el, output_dtype=el, vectorized=True)
    zero = mm_k.zero
    r, s, t = mm_k.mac_dims

    ht_ty = np.ndarray[(m, k), np.dtype[el]]
    wt_ty = np.ndarray[(k, n), np.dtype[el]]
    st_ty = np.ndarray[(m, n), np.dtype[el]]
    ot_ty = np.ndarray[(m, n), np.dtype[el]]

    dH = [(m//r, r*k), (k//s, s), (r, k), (s, 1)]
    dW = [(k//s, s*n), (n//t, t), (s, n), (t, 1)]
    dO = [(m//r, r*n), (r, t), (n//t, r*t), (t, 1)]

    fH  = ObjectFifo(ht_ty, name="H_L3L1",  depth=2)
    fWa = ObjectFifo(wt_ty, name="Wa_L3L1", depth=2)
    fWb = ObjectFifo(wt_ty, name="Wb_L3L1", depth=2)
    fS  = ObjectFifo(st_ty, name="S_core", depth=qc_n)
    memO = ObjectFifo(ot_ty, name="memO")

    mH  = fH.cons().forward(name="mH",  dims_to_stream=dH)
    mWa = fWa.cons().forward(name="mWa", dims_to_stream=dW)
    mWb = fWb.cons().forward(name="mWb", dims_to_stream=dW)
    outO = memO.cons().forward(name="outO", dims_to_stream=dO)

    # Worker 1: produces q_latent tiles, replays 64× via DMA
    def w1(of_h, of_w, of_s, mm, z):
        for _ in range_(REPLAY) if REPLAY > 1 else range(1):
            for _ in range_(qc_n) if qc_n > 1 else range(1):
                s = of_s.acquire(1); z(s)
                for _ in range_(qc_k) if qc_k > 1 else range(1):
                    h = of_h.acquire(1); w = of_w.acquire(1)
                    mm(h, w, s)
                    of_h.release(1); of_w.release(1)
                of_s.release(1)

    # Worker 2: consumes q_latent, multiplies by wq_b → output
    def w2(of_s, of_wb, of_o, mm, z):
        for _ in range_(qe_n) if qe_n > 1 else range(1):
            o = of_o.acquire(1); z(o)
            for _ in range_(qe_k) if qe_k > 1 else range(1):
                s = of_s.acquire(1)
                w = of_wb.acquire(1)
                mm(s, w, o)
                of_wb.release(1); of_s.release(1)
            of_o.release(1)

    W1 = Worker(w1, [mH.cons(), mWa.cons(), fS.prod(), mm_k, zero],
                stack_size=0xC00)
    W2 = Worker(w2, [fS.cons(), mWb.cons(), memO.prod(), mm_k, zero],
                stack_size=0x1000)

    # TAPs — hidden/wq_a repeated REPLAY times (stride 0)
    ht = TensorAccessPattern(
        tensor_dims=(M, Kc), offset=0,
        sizes=[REPLAY, qc_n, qc_k, m, k],
        strides=[0, 0, k, Kc, 1],
    )
    wa = TensorAccessPattern(
        tensor_dims=(Kc, Nc), offset=0,
        sizes=[REPLAY, qc_n, qc_k, k, n],
        strides=[0, n, k*Nc, Nc, 1],
    )
    wb = TensorAccessPattern(
        tensor_dims=(Ke, Nch), offset=0,
        sizes=[qe_n, qe_k, k, n],
        strides=[n, k*Nch, Nch, 1],
    )
    ot = TensorAccessPattern(
        tensor_dims=(M, Nch), offset=0,
        sizes=[1, qe_n, m, n],
        strides=[m*Nch, n, Nch, 1],
    )

    H_ty = np.ndarray[(M*Kc,), np.dtype[el]]
    A_ty = np.ndarray[(Kc*Nc,), np.dtype[el]]
    B_ty = np.ndarray[(Ke*Nch,), np.dtype[el]]
    O_ty = np.ndarray[(M*Nch,), np.dtype[el]]

    rt = Runtime()
    with rt.sequence(H_ty, A_ty, B_ty, O_ty) as (H, Wa, Wb, O):
        rt.start(W1); rt.start(W2)
        tg = rt.task_group()
        rt.fill(fH.prod(),  H,  tap=ht, task_group=tg)
        rt.fill(fWa.prod(), Wa, tap=wa, task_group=tg)
        rt.fill(fWb.prod(), Wb, tap=wb, task_group=tg)
        rt.drain(outO.cons(), O, tap=ot, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def compile_and_bench():
    rng = np.random.RandomState(42)
    M, Kc, Nc, Ke = M_TOK, K_HID, N_LAT, N_LAT
    N_full = N_CHUNK * 8  # 32768

    hidden = (rng.randn(M, Kc).astype('f4') * 0.3).astype(bfloat16)
    wq_a   = (rng.randn(Kc, Nc).astype('f4') * 0.03).astype(bfloat16)
    wq_b_full = (rng.randn(Ke, N_full).astype('f4') * 0.01).astype(bfloat16)
    q_full_ref = (hidden.astype('f4') @ wq_a.astype('f4')) @ wq_b_full.astype('f4')

    Hb = iron.zeros(M*Kc, dtype=bfloat16, device="npu"); Hb.numpy()[:] = hidden.ravel()
    Wa = iron.zeros(Kc*Nc, dtype=bfloat16, device="npu"); Wa.numpy()[:] = wq_a.ravel()

    print("Compiling fused QC+QE (DMA replay, N_chunk=4096)...")
    t0 = time.perf_counter()
    compiled = fst_mla_qc_qe_fused.specialize(
        M=M, Kc=Kc, Nc=Nc, Ke=Ke, Nch=N_CHUNK, el=bfloat16,
    )
    x, ii = compiled.compile()
    dt_c = (time.perf_counter() - t0) * 1000
    shutil.copy(x, "fst_mla_qc_qe_fused.xclbin")
    shutil.copy(ii, "fst_mla_qc_qe_fused_insts.bin")
    print(f"  Compiled in {dt_c:.0f}ms  xclbin={x.stat().st_size}B")

    n_chunks = 8
    Wb_chunk = iron.zeros(Ke * N_CHUNK, dtype=bfloat16, device="npu")
    Ob_chunk = iron.zeros(M * N_CHUNK, dtype=bfloat16, device="npu")
    Wb_chunk.numpy()[:] = wq_b_full[:, :N_CHUNK].ravel()

    print("  Warmup...")
    compiled(Hb, Wa, Wb_chunk, Ob_chunk)

    n_runs = 5; times = []
    for run in range(n_runs + 1):
        t0 = time.perf_counter()
        for ci in range(n_chunks):
            n0 = ci * N_CHUNK
            Wb_chunk.numpy()[:] = wq_b_full[:, n0:n0+N_CHUNK].ravel()
            compiled(Hb, Wa, Wb_chunk, Ob_chunk)
        dt = (time.perf_counter() - t0) * 1000
        if run > 0:
            times.append(dt)

    dt = np.mean(times)
    err = np.abs(Ob_chunk.numpy().reshape(M, N_CHUNK).astype('f4') -
                 q_full_ref[:, :N_CHUNK]).max()
    macs = (M*Kc*Nc + M*Ke*N_full) * 2
    gflops = macs / 1e9 / (dt/1000) if dt > 0 else 0

    print(f"\n  Fused QC+QE  (DMA replay, {n_chunks}×{N_CHUNK})")
    print(f"  MACs:         {macs/1e6:.0f}M")
    print(f"  Latency:      {dt:.1f} ms  [{np.min(times):.1f}..{np.max(times):.1f}]")
    print(f"  Per-chunk:    {dt/n_chunks:.1f} ms")
    print(f"  GFLOPS:       {gflops:.1f}")
    print(f"  Max error:    {err:.2e}")
    print(f"  vs separate:  QC(2.4) + QE(22.3) = 24.7ms  "
          f"Δ={(1-dt/24.7)*100:.0f}%")
    return dt


if __name__ == "__main__":
    dt = compile_and_bench()
    print(f"\n  FUSED: {dt:.1f} ms total ({dt/8:.1f} ms/chunk)")
