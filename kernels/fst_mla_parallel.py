#!/usr/bin/env python3
# fst_mla_parallel.py — Multi-core MLA GEMM (8 workers for Q_expand, 2 for O_proj)
#
# Q_expand: 8 workers × N=4096 each (Ndn=64, avoids BD-64 limit)
# O_proj A: 2 workers × N=4096 each
# O_proj B: 2 workers × K=4096 each

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

m, kt, nt = 8, 64, 64


def bench_qe_8core():
    M, K, N = 8, 1024, 32768
    nw = 8; Np = N // nw  # 4096
    Ndn = Np // nt  # 64
    Kdk = K // kt     # 16
    tiles = (M // m) * Ndn  # 64

    @iron.jit
    def qe_8(
        A_buf: In, B_buf: In,
        C0: Out, C1: Out, C2: Out, C3: Out,
        C4: Out, C5: Out, C6: Out, C7: Out,
        *, Mv: CompileTime[int], Kv: CompileTime[int], Nv: CompileTime[int],
        el: CompileTime[type],
    ):
        mm_k = kernels.mm(dim_m=m, dim_k=kt, dim_n=nt,
                          input_dtype=el, output_dtype=el, vectorized=True)
        zero = mm_k.zero

        ht = np.ndarray[(m, kt), np.dtype[el]]
        wt = np.ndarray[(kt, nt), np.dtype[el]]
        ct = np.ndarray[(m, nt), np.dtype[el]]

        dH = [(m//4, 4*kt), (kt//8, 8), (4, kt), (8, 1)]
        dW = [(kt//8, 8*nt), (nt//8, 8), (8, nt), (8, 1)]
        dO = [(m//4, 4*nt), (4, 8), (nt//8, 4*8), (8, 1)]

        def worker_fn(of_a, of_b, of_c, mm, z):
            for _ in range_(tiles) if tiles > 1 else range(1):
                co = of_c.acquire(1); z(co)
                for _ in range_(Kdk) if Kdk > 1 else range(1):
                    ca = of_a.acquire(1); cb = of_b.acquire(1)
                    mm(ca, cb, co)
                    of_a.release(1); of_b.release(1)
                of_c.release(1)

        fA = []; mAs = []; fB = []; mBs = []; mCs = []
        Ws = []
        for wi in range(nw):
            fA.append(ObjectFifo(ht, name=f"A_{wi}", depth=2))
            mAs.append(fA[-1].cons().forward(name=f"mA_{wi}", dims_to_stream=dH))
            fB.append(ObjectFifo(wt, name=f"B_{wi}", depth=2))
            mBs.append(fB[-1].cons().forward(name=f"mB_{wi}", dims_to_stream=dW))
            mc = ObjectFifo(ct, name=f"memC_{wi}")
            mCs.append(mc.cons().forward(name=f"mC_{wi}", dims_to_stream=dO))
            Ws.append(Worker(worker_fn, [mAs[-1].cons(), mBs[-1].cons(),
                          mc.prod(), mm_k, zero], stack_size=0xD00))

        rt = Runtime()
        A_ty = np.ndarray[(Mv*Kv,), np.dtype[el]]
        B_ty = np.ndarray[(Kv*N,), np.dtype[el]]
        C_ty = np.ndarray[(Mv*Np,), np.dtype[el]]
        types = [A_ty, B_ty] + [C_ty]*nw

        with rt.sequence(*types) as seq:
            for w in Ws: rt.start(w)
            tg = rt.task_group()

            for wi in range(nw):
                rt.fill(fA[wi].prod(), seq[0], tap=TensorAccessPattern(
                    tensor_dims=(Mv, Kv), offset=0,
                    sizes=[Ndn, Kdk, m, kt], strides=[0, kt, Kv, 1],
                ), task_group=tg)
                rt.fill(fB[wi].prod(), seq[1], tap=TensorAccessPattern(
                    tensor_dims=(Kv, N), offset=wi * Np,
                    sizes=[Ndn, Kdk, kt, nt],
                    strides=[nt, kt * N, N, 1],
                ), task_group=tg)
                rt.drain(mCs[wi].cons(), seq[2 + wi], tap=TensorAccessPattern(
                    tensor_dims=(Mv, N), offset=wi * Np,
                    sizes=[1, Ndn, m, nt],
                    strides=[m * N, nt, N, 1],
                ), task_group=tg)

            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()

    print("Compiling Q_expand (8 workers)...")
    rng = np.random.RandomState(42)
    A = (rng.randn(M, K).astype('f4') * 0.3).astype(bfloat16)
    B = (rng.randn(K, N).astype('f4') * 0.01).astype(bfloat16)
    ref = A.astype('f4') @ B.astype('f4')

    Ab = iron.zeros(M*K, dtype=bfloat16, device="npu"); Ab.numpy()[:] = A.ravel()
    Bb = iron.zeros(K*N, dtype=bfloat16, device="npu"); Bb.numpy()[:] = B.ravel()
    Cbs = [iron.zeros(M*Np, dtype=bfloat16, device="npu") for _ in range(nw)]

    t0 = time.perf_counter()
    compiled = qe_8.specialize(Mv=M, Kv=K, Nv=Np, el=bfloat16)
    x, ii = compiled.compile()
    dt_c = (time.perf_counter() - t0) * 1000
    shutil.copy(x, "fst_mla_qe_8core.xclbin")
    print(f"  Compiled in {dt_c:.0f}ms  xclbin={x.stat().st_size}B")

    print("  Warmup...")
    compiled(Ab, Bb, *Cbs)

    n_runs = 5; times = []
    for _ in range(n_runs):
        for cb in Cbs: cb.numpy()[:] = 0
        t0 = time.perf_counter()
        compiled(Ab, Bb, *Cbs)
        times.append((time.perf_counter() - t0) * 1000)

    dt = np.mean(times)
    out = np.concatenate([cb.numpy().reshape(M, Np).astype('f4') for cb in Cbs], axis=1)
    err = np.abs(out - ref).max()
    macs = M * K * N * 2
    gflops = macs / 1e9 / (dt/1000) if dt > 0 else 0
    print(f"\n  Q_expand 8-core: {dt:.1f} ms  [{np.min(times):.1f}..{np.max(times):.1f}]")
    print(f"  GFLOPS: {gflops:.1f}  err={err:.2e}  vs 1-core(22.3): {22.3/dt:.1f}×")
    return dt


def bench_o_proj(name, Mv, Kv, N_total, nw):
    Np = N_total // nw
    Ndn = Np // nt
    Kdk = Kv // kt
    tiles = (Mv // m) * Ndn

    @iron.jit
    def kernel(
        A_buf: In, B_buf: In, C0: Out, C1: Out,
        *, Mv: CompileTime[int], Kv: CompileTime[int], Nv: CompileTime[int],
        el: CompileTime[type],
    ):
        mm_k = kernels.mm(dim_m=m, dim_k=kt, dim_n=nt,
                          input_dtype=el, output_dtype=el, vectorized=True)
        zero = mm_k.zero
        ht = np.ndarray[(m, kt), np.dtype[el]]
        wt = np.ndarray[(kt, nt), np.dtype[el]]
        ct = np.ndarray[(m, nt), np.dtype[el]]
        dH = [(m//4, 4*kt), (kt//8, 8), (4, kt), (8, 1)]
        dW = [(kt//8, 8*nt), (nt//8, 8), (8, nt), (8, 1)]
        dO = [(m//4, 4*nt), (4, 8), (nt//8, 4*8), (8, 1)]

        def wfn(of_a, of_b, of_c, mm, z):
            for _ in range_(tiles) if tiles > 1 else range(1):
                co = of_c.acquire(1); z(co)
                for _ in range_(Kdk) if Kdk > 1 else range(1):
                    ca = of_a.acquire(1); cb = of_b.acquire(1)
                    mm(ca, cb, co)
                    of_a.release(1); of_b.release(1)
                of_c.release(1)

        fA0 = ObjectFifo(ht, name="A0", depth=2); mA0 = fA0.cons().forward(name="mA0", dims_to_stream=dH)
        fB0 = ObjectFifo(wt, name="B0", depth=2); mB0 = fB0.cons().forward(name="mB0", dims_to_stream=dW)
        mc0 = ObjectFifo(ct, name="memC0"); mC0 = mc0.cons().forward(name="mC0", dims_to_stream=dO)

        fA1 = ObjectFifo(ht, name="A1", depth=2); mA1 = fA1.cons().forward(name="mA1", dims_to_stream=dH)
        fB1 = ObjectFifo(wt, name="B1", depth=2); mB1 = fB1.cons().forward(name="mB1", dims_to_stream=dW)
        mc1 = ObjectFifo(ct, name="memC1"); mC1 = mc1.cons().forward(name="mC1", dims_to_stream=dO)

        W0 = Worker(wfn, [mA0.cons(), mB0.cons(), mc0.prod(), mm_k, zero], stack_size=0xD00)
        W1 = Worker(wfn, [mA1.cons(), mB1.cons(), mc1.prod(), mm_k, zero], stack_size=0xD00)

        A_ty = np.ndarray[(Mv*Kv,), np.dtype[el]]
        B_ty = np.ndarray[(Kv*N_total,), np.dtype[el]]
        C_ty = np.ndarray[(Mv*Np,), np.dtype[el]]

        rt = Runtime()
        with rt.sequence(A_ty, B_ty, C_ty, C_ty) as (a, b, c0, c1):
            rt.start(W0); rt.start(W1)
            tg = rt.task_group()

            for wi, (fA, fB, mC, off) in enumerate([
                (fA0, fB0, mC0, 0), (fA1, fB1, mC1, Np)
            ]):
                rt.fill(fA.prod(), a, tap=TensorAccessPattern(
                    tensor_dims=(Mv, Kv), offset=0,
                    sizes=[Ndn, Kdk, m, kt], strides=[0, kt, Kv, 1],
                ), task_group=tg)
                rt.fill(fB.prod(), b, tap=TensorAccessPattern(
                    tensor_dims=(Kv, N_total), offset=off,
                    sizes=[Ndn, Kdk, kt, nt],
                    strides=[nt, kt * N_total, N_total, 1],
                ), task_group=tg)
                rt.drain(mC.cons(), [c0, c1][wi], tap=TensorAccessPattern(
                    tensor_dims=(Mv, N_total), offset=off,
                    sizes=[1, Ndn, m, nt],
                    strides=[m * N_total, nt, N_total, 1],
                ), task_group=tg)

            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()

    print(f"\nCompiling {name} (2 workers)...")
    rng = np.random.RandomState(43)
    A = (rng.randn(Mv, Kv).astype('f4') * 0.3).astype(bfloat16)
    B = (rng.randn(Kv, N_total).astype('f4') * 0.02).astype(bfloat16)
    ref = A.astype('f4') @ B.astype('f4')

    Ab = iron.zeros(Mv*Kv, dtype=bfloat16, device="npu"); Ab.numpy()[:] = A.ravel()
    Bb = iron.zeros(Kv*N_total, dtype=bfloat16, device="npu"); Bb.numpy()[:] = B.ravel()
    Cb0 = iron.zeros(Mv*Np, dtype=bfloat16, device="npu")
    Cb1 = iron.zeros(Mv*Np, dtype=bfloat16, device="npu")

    t0 = time.perf_counter()
    compiled = kernel.specialize(Mv=Mv, Kv=Kv, Nv=Np, el=bfloat16)
    x, ii = compiled.compile()
    dt_c = (time.perf_counter() - t0) * 1000
    shutil.copy(x, f"fst_mla_{name}_2core.xclbin")
    print(f"  Compiled in {dt_c:.0f}ms  xclbin={x.stat().st_size}B")

    print("  Warmup...")
    compiled(Ab, Bb, Cb0, Cb1)

    n_runs = 5; times = []
    for _ in range(n_runs):
        Cb0.numpy()[:] = 0; Cb1.numpy()[:] = 0
        t0 = time.perf_counter()
        compiled(Ab, Bb, Cb0, Cb1)
        times.append((time.perf_counter() - t0) * 1000)

    dt = np.mean(times)
    out = np.concatenate([Cb0.numpy().reshape(Mv, Np).astype('f4'),
                           Cb1.numpy().reshape(Mv, Np).astype('f4')], axis=1)
    err = np.abs(out - ref).max()
    macs = Mv * Kv * N_total * 2
    gflops = macs / 1e9 / (dt/1000) if dt > 0 else 0

    single = {"o_proj_a": 22.0, "o_proj_b": 14.9}.get(name, 0)
    print(f"  {name} 2-core: {dt:.1f} ms  [{np.min(times):.1f}..{np.max(times):.1f}]")
    print(f"  GFLOPS: {gflops:.1f}  err={err:.2e}  vs 1-core: {single/dt:.1f}×")
    return dt


if __name__ == "__main__":
    t_qe = bench_qe_8core()
    t_oa = bench_o_proj("o_proj_a", 8, 4096, 8192, 2)
    t_ob = bench_o_proj("o_proj_b", 8, 8192, 4096, 2)

    total = 2.4 + t_qe + 1.0 + 0.3 + 0.4 + 0.3 + t_oa + t_ob
    print(f"\n{'='*70}")
    print(f"MULTI-CORE MLA PROJECTION:")
    print(f"  Q compress:    2.4 ms  (1 core)")
    print(f"  Q expand:     {t_qe:.1f} ms  (8 cores)")
    print(f"  KV compress:   1.0 ms  (1 core)")
    print(f"  Q@K^T:         0.3 ms  (1 core)")
    print(f"  Softmax:       0.4 ms  (host)")
    print(f"  scores@V:      0.3 ms  (1 core)")
    print(f"  O proj A:     {t_oa:.1f} ms  (2 cores)")
    print(f"  O proj B:     {t_ob:.1f} ms  (2 cores)")
    print(f"  {'─'*40}")
    print(f"  TOTAL:        {total:.1f} ms")
    print(f"  vs 1-core:    63.7ms → {total:.1f}ms  ({63.7/total:.1f}×)")
    print(f"  Tokens/s:     {8000/total:.0f}")
