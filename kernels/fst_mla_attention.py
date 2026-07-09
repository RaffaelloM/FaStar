#!/usr/bin/env python3
# fst_mla_attention.py -- IRON MLA GQA Attention xclbin (XDNA2 / NPU2)
#
# Three-worker pipeline, fully on-NPU:
#   W1 (score):  Q @ K_t  →  raw scores         (2 in: Q,Kt;  1 out: S1)
#   W2 (softmax): softmax(scores) + attn_sinks   (2 in: S2,Snk; 1 out: S2s)
#   W3 (value):  softmax @ V  →  output          (2 in: S2s,V;  1 out: O)
#
# Each tile has ≤2 Shim-DMA inputs and ≤2 outputs, avoiding the
# "requires 3 input/1 output DMA channels" placement error.
#
# Dimensions: M=8 tokens, 64 Q heads, head_dim=512, seq_len=128

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

M_TOKENS, N_HEADS, HEAD_DIM, SEQ_LEN = 8, 64, 512, 128
FLAT = M_TOKENS * N_HEADS

TILE_M, TILE_K, TILE_N = 32, 32, 128
Q_GROUPS = FLAT // TILE_M; K_STEPS = HEAD_DIM // TILE_K; V_STEPS = K_STEPS
m, k, n = TILE_M, TILE_K, TILE_N


@iron.jit
def fst_mla_attention(
    Q_flat: In, K_t: In, V_kv: In,
    Sinks: In,                       # [FLAT] BF16 attn_sinks, one per Q vector
    Scores_buf: In,                  # scratch [flat, seq_len] DDR
    O_flat: Out,
    *, n_head: CompileTime[int], head_dim: CompileTime[int],
    seq_len: CompileTime[int], m_tok: CompileTime[int],
    element_type: CompileTime[type],
):
    el = element_type
    flat = m_tok * n_head

    # ── Kernels ───────────────────────────────────────────────────────
    score_mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=el,
                          output_dtype=el, vectorized=False)
    value_mm = kernels.mm(dim_m=m, dim_k=n, dim_n=k, input_dtype=el,
                          output_dtype=el, vectorized=False)
    softmax_krnl = ExternalFunction(
        "fst_mla_softmax_32x128", source_file="fst_mla_attention_kernel.cc",
        arg_types=[np.ndarray[(m*n,), np.dtype[el]],
                   np.ndarray[(m,),   np.dtype[el]]],
        include_dirs=[aie_config.cxx_header_path()],
        object_file_name="fst_mla_softmax.o",
    )

    # ── Types ─────────────────────────────────────────────────────────
    Q_ty  = np.ndarray[(flat*head_dim,), np.dtype[el]]
    K_ty  = np.ndarray[(head_dim*seq_len,), np.dtype[el]]
    V_ty  = np.ndarray[(seq_len*head_dim,), np.dtype[el]]
    D_ty  = np.ndarray[(flat,), np.dtype[el]]
    S_ty  = np.ndarray[(flat*seq_len,), np.dtype[el]]

    q_ty  = np.ndarray[(m*k,), np.dtype[el]]    # [1024]
    k_ty  = np.ndarray[(k*n,), np.dtype[el]]    # [4096]
    v_ty  = np.ndarray[(n*k,), np.dtype[el]]    # [4096]
    s_ty  = np.ndarray[(m*n,), np.dtype[el]]    # [4096]
    o_ty  = np.ndarray[(m*k,), np.dtype[el]]    # [1024]
    snk_ty = np.ndarray[(m,), np.dtype[el]]     # [32]

    # ── FIFOs ─────────────────────────────────────────────────────────
    fq  = ObjectFifo(q_ty,  name="Q_L3L1",   depth=2)
    fkt = ObjectFifo(k_ty,  name="Kt_L3L1",  depth=2)
    fs1 = ObjectFifo(s_ty,  name="S1_L1L3",  depth=2)  # W1 → DDR
    fs2 = ObjectFifo(s_ty,  name="S2_L3L1",  depth=2)  # DDR → W2
    fs3 = ObjectFifo(s_ty,  name="S3_L1L1",  depth=2)  # W2 → W3 (core)
    fv  = ObjectFifo(v_ty,  name="V_L3L1",   depth=2)  # DDR → W3
    fsn = ObjectFifo(snk_ty, name="Snk_L3L1",depth=1)  # DDR → W2
    fo  = ObjectFifo(o_ty,  name="O_L1L3",   depth=2)  # W3 → DDR

    # ── Worker 1: Q @ K_t → scores  (2 in: Q,Kt; 1 out: S1) ─────────
    def w1(of_q, of_kt, of_s, mm_s):
        for _ in range_(Q_GROUPS) if Q_GROUPS > 1 else range(1):
            s = of_s.acquire(1)
            for i in range_(m*n) if m*n > 1 else range(1): s[i] = 0
            for _ in range_(K_STEPS) if K_STEPS > 1 else range(1):
                q = of_q.acquire(1); k = of_kt.acquire(1)
                mm_s(q, k, s)
                of_q.release(1); of_kt.release(1)
            of_s.release(1)

    # ── Worker 2: softmax  (2 in: S2,Snk; 1 out: S3 core) ──────────
    def w2(of_s, of_snk, of_s3, softmax):
        for _ in range_(Q_GROUPS) if Q_GROUPS > 1 else range(1):
            s = of_s.acquire(1); snk = of_snk.acquire(1)
            s3 = of_s3.acquire(1)
            # Copy s → s3 then softmax s3 (avoids modify-in-place on input)
            for i in range_(m*n) if m*n > 1 else range(1): s3[i] = s[i]
            softmax(s3, snk)
            of_s.release(1); of_snk.release(1); of_s3.release(1)

    # ── Worker 3: scores @ V → output  (2 in: S3,V; 1 out: O) ──────
    def w3(of_s, of_v, of_o, mm_v):
        for _ in range_(Q_GROUPS) if Q_GROUPS > 1 else range(1):
            s = of_s.acquire(1)
            for _ in range_(V_STEPS) if V_STEPS > 1 else range(1):
                o = of_o.acquire(1)
                for i in range_(m*k) if m*k > 1 else range(1): o[i] = 0
                v = of_v.acquire(1)
                mm_v(s, v, o)
                of_v.release(1); of_o.release(1)
            of_s.release(1)

    W1 = Worker(w1, [fq.cons(), fkt.cons(), fs1.prod(), score_mm])
    W2 = Worker(w2, [fs2.cons(), fsn.cons(), fs3.prod(), softmax_krnl],
                stack_size=0x4000)
    W3 = Worker(w3, [fs3.cons(), fv.cons(), fo.prod(), value_mm])

    # ── TAPs ─────────────────────────────────────────────────────────
    q_tap = TensorAccessPattern(
        tensor_dims=(flat, head_dim), offset=0,
        sizes=[Q_GROUPS, K_STEPS, m, k],
        strides=[m*head_dim, k, head_dim, 1],
    )
    k_tap = TensorAccessPattern(
        tensor_dims=(head_dim, seq_len), offset=0,
        sizes=[Q_GROUPS, K_STEPS, k, n],
        strides=[0, k*n, n, 1],
    )
    s_tap = TensorAccessPattern(
        tensor_dims=(flat, seq_len), offset=0,
        sizes=[Q_GROUPS, m, n],
        strides=[m*seq_len, seq_len, 1],
    )
    v_tap = TensorAccessPattern(
        tensor_dims=(seq_len, head_dim), offset=0,
        sizes=[Q_GROUPS, V_STEPS, n, k],
        strides=[0, k, head_dim, 1],
    )
    snk_tap = TensorAccessPattern(
        tensor_dims=(flat,), offset=0, sizes=[Q_GROUPS, m], strides=[m, 1],
    )
    o_tap = TensorAccessPattern(
        tensor_dims=(flat, head_dim), offset=0,
        sizes=[Q_GROUPS, V_STEPS, m, k],
        strides=[m*head_dim, k, head_dim, 1],
    )

    # ── Runtime ──────────────────────────────────────────────────────
    rt = Runtime()
    with rt.sequence(Q_ty, K_ty, V_ty, D_ty, S_ty, Q_ty) as (A, B, C, D, E, F):
        rt.start(W1); rt.start(W2); rt.start(W3)

        tg1 = rt.task_group()
        rt.fill(fq.prod(),   A, tap=q_tap,   task_group=tg1)
        rt.fill(fkt.prod(),  B, tap=k_tap,   task_group=tg1)
        rt.drain(fs1.cons(), E, tap=s_tap,   task_group=tg1, wait=True)
        rt.finish_task_group(tg1)

        tg2 = rt.task_group()
        rt.fill(fs2.prod(),  E, tap=s_tap,   task_group=tg2)
        rt.fill(fsn.prod(),  D, tap=snk_tap, task_group=tg2)
        rt.fill(fv.prod(),   C, tap=v_tap,   task_group=tg2)
        rt.drain(fo.cons(),  F, tap=o_tap,   task_group=tg2, wait=True)
        rt.finish_task_group(tg2)

    return Program(NPU2(), rt).resolve_program()


# ── Host helpers ───────────────────────────────────────────────────────
def prepare_k_t(K): return np.ascontiguousarray(K.T).astype(bfloat16)

def ref_mla(Q, K, V, sinks, M_tok=8, n_head=64, head_dim=512):
    flat = M_tok * n_head
    ks = 1.0 / np.sqrt(float(head_dim))
    Qf, Kf, Vf, Sf = Q.astype('f4'), K.astype('f4'), V.astype('f4'), sinks.astype('f4')
    out = np.zeros((flat, head_dim), dtype='f4')
    for i in range(flat):
        h = i % n_head; qh = Qf[i]
        s = np.dot(qh, Kf.T) * ks
        mx = max(Sf[h], s.max())
        se = np.exp(s - mx); de = np.exp(Sf[h] - mx) + se.sum()
        out[i] = (se / de) @ Vf
    return out

def aot_compile():
    x, i = fst_mla_attention.specialize(
        n_head=N_HEADS, head_dim=HEAD_DIM, seq_len=SEQ_LEN,
        m_tok=M_TOKENS, element_type=bfloat16).compile()
    print(f"AOT compiled fst_mla_attention {M_TOKENS}x{N_HEADS}x{HEAD_DIM} "
          f"seq_len={SEQ_LEN} (NPU2)\n  xclbin: {x}\n  insts: {i}")

def test_npu():
    print("=" * 60)
    print("MLA GQA Attention NPU Test")
    print("=" * 60)
    import time
    rng = np.random.RandomState(42)
    M, H, D, S = M_TOKENS, N_HEADS, HEAD_DIM, SEQ_LEN
    flat = M * H

    Q_bf16 = (rng.randn(flat, D).astype('f4') * 0.3).astype(bfloat16)
    K_bf16 = (rng.randn(S, D).astype('f4') * 0.3).astype(bfloat16)
    V_bf16 = (rng.randn(S, D).astype('f4') * 0.3).astype(bfloat16)
    snk_bf16 = (rng.randn(H).astype('f4') * 0.5 - 1.0).astype(bfloat16)
    snk_f = np.tile(snk_bf16, M).astype(bfloat16)
    Kt = prepare_k_t(K_bf16)

    t0 = time.perf_counter()
    ref = ref_mla(Q_bf16, K_bf16, V_bf16, snk_bf16)
    t_ref = (time.perf_counter() - t0) * 1000

    set_current_device(NPU2())
    Ab = iron.zeros(flat*D, dtype=bfloat16, device="npu"); Ab.numpy()[:] = Q_bf16.ravel()
    Bb = iron.zeros(D*S, dtype=bfloat16, device="npu"); Bb.numpy()[:] = Kt.ravel()
    Cb = iron.zeros(S*D, dtype=bfloat16, device="npu"); Cb.numpy()[:] = V_bf16.ravel()
    Db = iron.zeros(flat, dtype=bfloat16, device="npu"); Db.numpy()[:] = snk_f
    Eb = iron.zeros(flat*S, dtype=bfloat16, device="npu"); Eb.numpy()[:] = 0
    Fb = iron.zeros(flat*D, dtype=bfloat16, device="npu"); Fb.numpy()[:] = 0

    print("  Warmup...")
    fst_mla_attention(Ab, Bb, Cb, Db, Eb, Fb,
                      n_head=H, head_dim=D, seq_len=S, m_tok=M, element_type=bfloat16)

    n_runs = 5; times = []
    for _ in range(n_runs):
        Eb.numpy()[:] = 0; Fb.numpy()[:] = 0
        t1 = time.perf_counter()
        fst_mla_attention(Ab, Bb, Cb, Db, Eb, Fb,
                          n_head=H, head_dim=D, seq_len=S, m_tok=M, element_type=bfloat16)
        times.append((time.perf_counter() - t1) * 1000)

    out_npu = Fb.numpy().reshape(flat, D).astype('f4')
    err = np.abs(out_npu - ref)
    max_err, max_rel = err.max(), err.max() / (np.abs(ref).max() + 1e-10)

    print(f"\n  CPU ref:          {t_ref:.1f} ms")
    print(f"  NPU mean ({n_runs}): {np.mean(times):.1f} ms  [{np.min(times):.1f}..{np.max(times):.1f}]")
    print(f"  Max abs error:   {max_err:.5e}")
    print(f"  Max rel error:   {max_rel:.5e}")
    ok = max_rel < 0.1
    print(f"  STATUS:          {'PASS' if ok else 'FAIL'}")
    return ok, np.mean(times)

if __name__ == "__main__":
    ok, lat = test_npu()
    if ok:
        print(f"\n  LATENCY: {lat:.1f} ms  M={M_TOKENS} seq_len={SEQ_LEN}")
        aot_compile()
    else:
        print("\nFAILED.")
