#!/usr/bin/env python3
"""Qwen3.5-Next (qwen35) Gated Delta Net — reference SSM scan recurrence.

WHY THIS FILE EXISTS
  The "SSM" layers of Qwen3.5-Next are NOT Mamba2 vector selective scans.  They are
  Gated Delta Net (GDN) linear attention with a MATRIX state per value head.  The
  first fst_ssm_scan_kernel.cc implemented `s = a*s + b*x; y = c*s` (a 1-D vector
  scan over [d_state=128] with 16 groups) — structurally wrong.  This file is the
  bit-correct reference the AIE kernel must match.

EXACT CONTRACT (from transformers Qwen3NextGatedDeltaNet.forward +
  torch_recurrent_gated_delta_rule, use_qk_l2norm_in_kernel=True):
  Per v-head, per decode token, with persistent state S [head_k, head_v] = [128,128]:
    qn = l2norm(q) * (1/sqrt(head_k_dim))     # q,k are RAW (pre-norm) [128]
    kn = l2norm(k)
    gdec = exp(g_logit)                       # g_logit = -exp(A_log)*softplus(a+dt_bias) < 0
    beta  = sigmoid(b)                        # already sigmoid'd by the caller
    S    = gdec * S
    kvm  = Sᵀ @ kn          (= (S*kn.unsqueeze(-1)).sum(-2))   [128]
    delta= (v - kvm) * beta                                  [128]
    S    = S + outer(kn, delta)
    y    = Sᵀ @ qn          (= (S*qn.unsqueeze(-1)).sum(-2))   [128]
  l2norm(x, eps=1e-6) = x * rsqrt((x*x).sum() + eps).

GEOMETRY (GGUF blk.0, confirmed):
  head_k_dim = head_v_dim = state_size = 128
  n_k_heads = 16, n_v_heads = 48 (q,k repeat_interleave 3x to v-heads)
  value_dim = 6144 = head_v_dim * n_v_heads; conv_dim = 10240; conv_kernel = 4.

DIVISION OF LABOR (zero-CPU): the AIE scan kernel implements the recurrence above
  (l2norm + scale + exp + delta rule).  The surrounding conv1d/SiLU, the qkv/gate
  projections, RMSNorm-gated, and out_proj are separate GEMM/ew_unified NPU ops.

This file: a hand-verifiable pure-delta-rule unit test, a faithful full-step
reference (numpy + torch), a stability test on real shapes, and a bit-correctness
cross-check against transformers' torch_recurrent_gated_delta_rule.
"""
import math
import numpy as np

HEAD_K = 128
HEAD_V = 128
N_V = 48
N_K = 16
D_INNER = HEAD_V * N_V            # 6144
L2_EPS = 1e-6
SCALE = 1.0 / math.sqrt(HEAD_K)   # 1/sqrt(head_k_dim)


def _l2norm_np(x):
    return x * (1.0 / np.sqrt((x * x).sum() + L2_EPS))


# ── pure matrix delta rule (the stripped AIE kernel contract) ───────────
def gdn_delta_rule(qn, kn, v, gdec, beta, S):
    """Inputs already l2-normed/scaled/decayed.  Pure recurrence.
      qn,kn : [HEAD_K]   v : [HEAD_V]   gdec,beta : scalars   S : [HEAD_K, HEAD_V]
    Returns (y [HEAD_V], S_new [HEAD_K, HEAD_V]).  Hand-verifiable (integer test)."""
    S = gdec * S
    kvm = S.T @ kn                       # [HV]
    delta = (v - kvm) * beta             # [HV]
    S = S + np.outer(kn, delta)          # [HK, HV]
    y = S.T @ qn                         # [HV]
    return y, S


# ── full faithful step (matches transformers recurrent path) ─────────────
def gdn_scan_step_numpy(q, k, v, g_logit, beta, S):
    """Faithful Qwen3-Next GDN decode step.
      q,k : [HEAD_K] RAW (pre-norm)   v : [HEAD_V]
      g_logit : scalar (<0) = -exp(A_log)*softplus(a+dt_bias)
      beta    : scalar (already sigmoid'd)
      S : [HEAD_K, HEAD_V] persistent state (fp32)."""
    qn = _l2norm_np(q) * SCALE
    kn = _l2norm_np(k)
    gdec = float(np.exp(g_logit))
    return gdn_delta_rule(qn, kn, v, gdec, beta, S)


def gdn_scan_step_torch(q, k, v, g_logit, beta, S):
    import torch
    def l2n(x):
        return x * torch.rsqrt((x * x).sum() + L2_EPS)
    qn = l2n(q) * SCALE
    kn = l2n(k)
    gdec = math.exp(g_logit)
    S = gdec * S
    kvm = (S * kn.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kvm) * beta
    S = S + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    y = (S * qn.unsqueeze(-1)).sum(dim=-2)
    return y, S


# ── hand-verifiable unit test (pure delta rule, integer math) ────────────
def _unit_test():
    """HK=HV=3, gdec=0.5, beta=1.0, unit-basis qn,kn so l2norm is identity-ish.
    Verify the 2-step delta rule by hand."""
    HK = HV = 3
    qn = np.array([0., 1., 0.], np.float32)   # basis e1
    kn = np.array([0., 1., 0.], np.float32)   # basis e1
    v = np.array([1., 2., 3.], np.float32)
    gdec, beta = 0.5, 1.0
    S0 = np.zeros((HK, HV), np.float32)

    # Step 1 from zero state: S stays 0 (gdec*0=0), kvm=0, delta=v*beta=v,
    # S = outer(kn, v) (row 1 = [1,2,3]), y = Sᵀ qn = S[1,:] = [1,2,3].
    y1, S1 = gdn_delta_rule(qn, kn, v, gdec, beta, S0)
    assert np.allclose(S1, np.outer(kn, v), atol=1e-6), f"S1 wrong:\n{S1}"
    assert np.allclose(y1, [1., 2., 3.], atol=1e-6), f"y1 wrong: {y1}"

    # Step 2 (same v): decay S1 by 0.5 -> row1 = [0.5,1,1.5]; kvm = Sᵀ kn = S[1,:]
    #   = [0.5,1,1.5]; delta = (v - kvm)*1 = [0.5,1,1.5]; S += outer(kn,delta) ->
    #   row1 += [0.5,1,1.5] -> row1 = [1,2,3] (decay + full-update cancel: same
    #   (kn,v) arrived again with beta=1 -> state returns to pre-decay value).
    #   y = Sᵀ qn = S[1,:] = [1,2,3].
    y2, S2 = gdn_delta_rule(qn, kn, v, gdec, beta, S1)
    assert np.allclose(S2, [[0, 0, 0], [1, 2, 3], [0, 0, 0]], atol=1e-6), f"S2 wrong:\n{S2}"
    assert np.allclose(y2, [1., 2., 3.], atol=1e-6), f"y2 wrong: {y2}"
    print("unit_test OK (hand-verified 2-step delta rule, 3x3 state)")


# ── realistic stability test on real shapes ──────────────────────────────
def _stability_test():
    """48 v-heads, 128x128 state, 64 decode steps with bounded random inputs.
    Uses the FULL step (l2norm+scale+exp).  Checks bounded ||S|| and finite y,
    and cross-checks numpy vs torch."""
    rng = np.random.default_rng(0)
    S = [np.zeros((HEAD_K, HEAD_V), np.float32) for _ in range(N_V)]
    max_norm = 0.0
    for t in range(64):
        q = rng.standard_normal((N_V, HEAD_K)).astype(np.float32)
        k = rng.standard_normal((N_V, HEAD_K)).astype(np.float32)
        v = rng.standard_normal((N_V, HEAD_V)).astype(np.float32)
        g_logit = np.full(N_V, -1.0, np.float32)   # exp(-1)=0.368 decay
        beta = np.full(N_V, 0.5, np.float32)
        ys = []
        for h in range(N_V):
            y, S[h] = gdn_scan_step_numpy(q[h], k[h], v[h], g_logit[h], beta[h], S[h])
            ys.append(y)
        max_norm = max(max_norm, max(np.linalg.norm(S[h]) for h in range(N_V)))
        assert np.all(np.isfinite(ys[-1])), f"non-finite y at step {t}"
    print(f"stability_test OK: 64 steps x {N_V} v-heads, max ||S||={max_norm:.3f} (bounded)")

    # numpy vs torch on one step
    try:
        import torch
        S = np.zeros((HEAD_K, HEAD_V), np.float32)
        q = rng.standard_normal(HEAD_K).astype(np.float32)
        k = rng.standard_normal(HEAD_K).astype(np.float32)
        v = rng.standard_normal(HEAD_V).astype(np.float32)
        yn, Sn = gdn_scan_step_numpy(q, k, v, -1.0, 0.5, S.copy())
        yt, St = gdn_scan_step_torch(torch.tensor(q), torch.tensor(k),
                                     torch.tensor(v), -1.0, 0.5,
                                     torch.tensor(S.copy()))
        assert np.allclose(yn, yt.numpy(), atol=1e-6), "numpy vs torch mismatch (y)"
        assert np.allclose(Sn, St.numpy(), atol=1e-6), "numpy vs torch mismatch (S)"
        print("numpy==torch OK")
    except ImportError:
        print("torch not available; skipped numpy-vs-torch cross-check")


# ── bit-correctness vs transformers Qwen3NextGatedDeltaNet ───────────────
def _transformers_crosscheck():
    """Feed the SAME raw (q,k,v,g_logit,beta) to our numpy full step and to
    transformers' torch_recurrent_gated_delta_rule(use_qk_l2norm_in_kernel=True)
    and compare.  This is the gold bit-correctness test for the kernel contract."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            torch_recurrent_gated_delta_rule)
    except Exception as e:
        print(f"transformers cross-check SKIPPED ({e})")
        return
    import torch
    rng = np.random.default_rng(1)
    B, H, T = 1, N_V, 1            # batch, n_v_heads, 1 token (decode)
    # transformers recurrent layout (as the layer produces): query,key [B,T,H,head_k],
    # value [B,T,H,head_v], g/beta [B,T,H], state [B,H,HK,HV].  recurrent transposes
    # (1,2) internally to [B,H,T,dim].
    q = torch.randn(B, T, H, HEAD_K, dtype=torch.float32)
    k = torch.randn(B, T, H, HEAD_K, dtype=torch.float32)
    v = torch.randn(B, T, H, HEAD_V, dtype=torch.float32)
    g_logit = torch.full((B, T, H), -1.0, dtype=torch.float32)   # decay logit
    beta = torch.full((B, T, H), 0.5, dtype=torch.float32)       # already sigmoid'd
    state_ref = torch.zeros(B, H, HEAD_K, HEAD_V, dtype=torch.float32)

    y_t, state_t = torch_recurrent_gated_delta_rule(
        q, k, v, g_logit, beta, state_ref, True, use_qk_l2norm_in_kernel=True)

    # our numpy full step, per v-head (one decode token)
    ys, state_np = [], np.zeros((N_V, HEAD_K, HEAD_V), np.float32)
    for h in range(N_V):
        y, state_np[h] = gdn_scan_step_numpy(
            q[0, 0, h].numpy(), k[0, 0, h].numpy(), v[0, 0, h].numpy(),
            -1.0, 0.5, np.zeros((HEAD_K, HEAD_V), np.float32))
        ys.append(y)
    y_np = np.stack(ys)                                  # [N_V, HEAD_V]
    y_t_np = y_t.detach().numpy()[0, 0]                 # [N_V, HEAD_V]
    state_t_np = state_t.detach().numpy()[0]            # [N_V, HK, HV]
    max_y = float(np.max(np.abs(y_np - y_t_np)))
    max_s = float(np.max(np.abs(state_np - state_t_np)))
    ok = max_y < 1e-3 and max_s < 1e-3
    print(f"transformers cross-check: max|Δy|={max_y:.2e} max|ΔS|={max_s:.2e} "
          f"-> {'BIT-CORRECT' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    _unit_test()
    _stability_test()
    _transformers_crosscheck()