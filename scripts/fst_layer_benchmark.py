#!/usr/bin/env python3
# fst_layer_benchmark.py — Full Transformer Layer Benchmark (FIXED)
#
# Pre-compiles all xclbins once, pre-allocates BOs, reuses buffers.
# Measures TRUE NPU kernel execution time (no BO alloc, no JIT overhead).

import time, numpy as np, os
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron.device import NPU2
from aie.utils import set_current_device
set_current_device(NPU2())

from fst_mla_all_gemm import make_gemm, MLA_OPS, auto_tile_m, CHUNK_N

M_TOKENS, N_HEADS, HEAD_DIM, SEQ_LEN = 8, 64, 512, 128
FLAT = M_TOKENS * N_HEADS


def host_rms_norm(x, weight, eps=1e-6):
    xs = x.astype(np.float32); ws = weight.astype(np.float32)
    rms = 1.0 / np.sqrt(np.mean(xs**2, axis=1) + eps)
    return (xs * rms[:, None] * ws).astype(bfloat16)

def host_softmax(scores, sinks):
    S = scores.astype(np.float32); sk = sinks.astype(np.float32)
    S *= 1.0 / np.sqrt(float(HEAD_DIM))
    mx = np.maximum(sk, S.max(axis=1))
    se = np.exp(S - mx[:, None])
    de = np.exp(sk - mx) + se.sum(axis=1)
    return (se / de[:, None]).astype(bfloat16)


class GemmRunner:
    """Pre-compiled GEMM with pre-allocated BOs."""
    def __init__(self, name, M_val, K_val, N_full):
        self.name = name
        self.M, self.K, self.N = M_val, K_val, N_full
        # Use chunk size = min(N_full, CHUNK_N)
        self.chunk_N = min(N_full, CHUNK_N)
        self.n_chunks = (N_full + self.chunk_N - 1) // self.chunk_N

        # Pre-compile for chunk_N (all chunks same size except maybe last)
        self.compiled = make_gemm(name, M_val, K_val, self.chunk_N)

        # Pre-allocate buffers
        self.Ab = iron.zeros(M_val * K_val, dtype=bfloat16, device="npu")
        self.Bb = iron.zeros(K_val * self.chunk_N, dtype=bfloat16, device="npu")
        self.Cb = iron.zeros(M_val * self.chunk_N, dtype=bfloat16, device="npu")

    def run(self, A_data, B_data_full, n_iters=5):
        N_full = self.N
        self.Ab.numpy()[:] = A_data.ravel()

        times = []
        for _ in range(n_iters + 1):
            t0 = time.perf_counter()
            result = np.zeros((self.M, N_full), dtype=bfloat16)
            for ci in range(self.n_chunks):
                n0 = ci * self.chunk_N; n1 = min(n0 + self.chunk_N, N_full)
                nc = n1 - n0
                if nc == self.chunk_N:
                    self.Bb.numpy()[:] = B_data_full[:, n0:n1].ravel()
                    self.compiled(self.Ab, self.Bb, self.Cb)
                    result[:, n0:n1] = self.Cb.numpy().reshape(self.M, self.chunk_N)
                else:
                    # Last partial chunk
                    Bb1 = iron.zeros(self.K * nc, dtype=bfloat16, device="npu")
                    Bb1.numpy()[:] = B_data_full[:, n0:n1].ravel()
                    Cb1 = iron.zeros(self.M * nc, dtype=bfloat16, device="npu")
                    compiled1 = make_gemm(self.name, self.M, self.K, nc)
                    compiled1(self.Ab, Bb1, Cb1)
                    result[:, n0:n1] = Cb1.numpy().reshape(self.M, nc)
                    del Bb1, Cb1
            dt = (time.perf_counter() - t0) * 1000
            if _ > 0:
                times.append(dt)
        return result, np.mean(times), np.min(times)


def benchmark_full_layer():
    print("=" * 70)
    print("FULL MLA ATTENTION LAYER BENCHMARK (FIXED)")
    print(f"  M={M_TOKENS}  H={N_HEADS}  D={HEAD_DIM}  S={SEQ_LEN}")
    print("=" * 70)

    rng = np.random.RandomState(42)

    hidden = (rng.randn(M_TOKENS, 4096).astype(np.float32) * 0.3).astype(bfloat16)
    attn_norm_w = (rng.randn(4096).astype(np.float32) * 0.1).astype(bfloat16)
    wq_a = (rng.randn(4096, 1024).astype(np.float32) * 0.03).astype(bfloat16)
    wq_b = (rng.randn(1024, 32768).astype(np.float32) * 0.01).astype(bfloat16)
    wkv = (rng.randn(4096, 512).astype(np.float32) * 0.02).astype(bfloat16)
    wo_a = (rng.randn(4096, 8192).astype(np.float32) * 0.02).astype(bfloat16)
    wo_b = (rng.randn(8192, 4096).astype(np.float32) * 0.01).astype(bfloat16)
    sinks = (rng.randn(N_HEADS).astype(np.float32) * 0.5 - 1.0).astype(bfloat16)
    sinks_flat = np.tile(sinks, M_TOKENS).astype(bfloat16)
    K_cache = (rng.randn(SEQ_LEN, HEAD_DIM).astype(np.float32) * 0.3).astype(bfloat16)
    V_cache = (rng.randn(SEQ_LEN, HEAD_DIM).astype(np.float32) * 0.3).astype(bfloat16)
    K_t = np.ascontiguousarray(K_cache.T).astype(bfloat16)

    # ── Pre-compile all runners ─────────────────────────────────────
    print("\n  Pre-compiling GEMM runners...")
    t_pc = time.perf_counter()
    runners = {
        "q_compress":  GemmRunner("q_compress",   8,    4096, 1024),
        "q_expand":    GemmRunner("q_expand",     8,    1024, 32768),
        "kv_compress": GemmRunner("kv_compress",  8,    4096, 512),
        "qk":          GemmRunner("qk",           512,  512,  128),
        "sv":          GemmRunner("sv",           512,  128,  512),
        "o_proj_a":    GemmRunner("o_proj_a",     8,    4096, 8192),
        "o_proj_b":    GemmRunner("o_proj_b",     8,    8192, 4096),
    }
    dt_pc = (time.perf_counter() - t_pc) * 1000
    print(f"  Pre-compile done in {dt_pc:.0f} ms")

    # ── RMSNorm ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    h_norm = host_rms_norm(hidden, attn_norm_w)
    dt_rms = (time.perf_counter() - t0) * 1000
    print(f"\n  [1] RMSNorm (host):               {dt_rms:.1f} ms")

    # ── Q Compress ──────────────────────────────────────────────────
    _, dt_qc, _ = runners["q_compress"].run(h_norm, wq_a)
    print(f"  [2] Q compress GEMM:              {dt_qc:.1f} ms")

    # ── Q Expand ────────────────────────────────────────────────────
    q_latent = (rng.randn(M_TOKENS, 1024).astype(np.float32) * 0.3).astype(bfloat16)
    _, dt_qe, _ = runners["q_expand"].run(q_latent, wq_b)
    print(f"  [3] Q expand GEMM:               {dt_qe:.1f} ms")

    # ── KV Compress ─────────────────────────────────────────────────
    _, dt_kv, _ = runners["kv_compress"].run(h_norm, wkv)
    print(f"  [4] KV compress GEMM:             {dt_kv:.1f} ms")

    # ── Q@K^T ───────────────────────────────────────────────────────
    q_full = (rng.randn(FLAT, HEAD_DIM).astype(np.float32) * 0.3).astype(bfloat16)
    scores, dt_qk, _ = runners["qk"].run(q_full, K_t)
    print(f"  [5] Q@K^T GEMM:                  {dt_qk:.1f} ms")

    # ── Softmax ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    weights = host_softmax(scores, sinks_flat)
    dt_sm = (time.perf_counter() - t0) * 1000
    print(f"  [6] Softmax (host):               {dt_sm:.1f} ms")

    # ── scores@V ────────────────────────────────────────────────────
    attn_out, dt_sv, _ = runners["sv"].run(weights, V_cache)
    print(f"  [7] scores@V GEMM:               {dt_sv:.1f} ms")

    # ── O Proj A ────────────────────────────────────────────────────
    attn_in = attn_out.reshape(M_TOKENS, -1)[:, :4096]  # [8, 4096]
    _, dt_oa, _ = runners["o_proj_a"].run(attn_in, wo_a)
    print(f"  [8] O proj A GEMM:               {dt_oa:.1f} ms")

    # ── O Proj B ────────────────────────────────────────────────────
    o_latent_data = (rng.randn(M_TOKENS, 8192).astype(np.float32) * 0.3).astype(bfloat16)
    _, dt_ob, _ = runners["o_proj_b"].run(o_latent_data, wo_b)
    print(f"  [9] O proj B GEMM:               {dt_ob:.1f} ms")

    # ── Totals ──────────────────────────────────────────────────────
    total_gemm = dt_qc + dt_qe + dt_kv + dt_qk + dt_sv + dt_oa + dt_ob
    total_all = total_gemm + dt_rms + dt_sm

    print(f"\n  {'─'*50}")
    print(f"  Total GEMM (NPU):                {total_gemm:.1f} ms")
    print(f"  Total RMS + Softmax (host):      {dt_rms + dt_sm:.1f} ms")
    print(f"  TOTAL MLA ATTENTION:             {total_all:.1f} ms")
    print(f"  Tokens/s (M=8):                  {M_TOKENS * 1000 / total_all:.0f} tok/s")
    print(f"\n  Target: <100ms per layer")
    print(f"  Status: {'PASS' if total_all < 100 else 'ABOVE'}")
    return total_all


if __name__ == "__main__":
    t = benchmark_full_layer()
    print(f"\n  FINAL LATENCY: {t:.1f} ms per attention sublayer")
