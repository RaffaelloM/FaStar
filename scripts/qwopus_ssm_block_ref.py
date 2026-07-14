#!/usr/bin/env python3
"""qwopus_ssm_block_ref.py — HF-faithful reference for the Qwen3.5-Next SSM (GDN) block.

Loads the C++ engine dumps (FST_SSM_DUMP at layer 0, pos 0: dequanted fp32 weights
+ input residual + every intermediate) from a dump dir, RECOMPUTES the block two ways,
and reports the FIRST diverging tensor vs the engine:

  (1) Ported reference — the block math ported verbatim from transformers
      modeling_qwen3_next.py (Qwen3NextGatedDeltaNet.forward + helpers), using the
      FLAT GGUF weight layout the engine uses (attn_qkv = [q|k|v] flat, etc.).
      Catches MATH/algorithm bugs (wrong op, order, scale, eps, activation, residual).

  (2) HF module ground truth — constructs a real transformers Qwen3NextGatedDeltaNet,
      loads the dumped weights into it (reconstructing HF's grouped in_proj_qkvz /
      in_proj_ba from the flat attn_qkv/attn_gate/ssm_alpha/ssm_beta), and runs its
      actual forward.  Catches WEIGHT-LAYOUT / orientation bugs the ported ref shares.

The first tensor whose ported-ref AND engine agree but HF-module disagrees = layout bug.
The first tensor whose ported-ref disagrees with engine = math bug.

Usage: python3 qwopus_ssm_block_ref.py /tmp/ssm_dump
"""
import sys, os, math, numpy as np

HD, HV, NK, NV, REP, CK = 5120, 128, 16, 48, 3, 4
KDIM, VDIM, CDIM = NK * HV, NV * HV, 2 * (NK * HV) + (NV * HV)  # 2048, 6144, 10240
EPS = 1e-6
SCALE = 1.0 / math.sqrt(HV)


def load(path, n):
    a = np.fromfile(path, dtype=np.float32, count=n)
    if a.size != n:
        raise SystemExit(f"{path}: expected {n} floats, got {a.size}")
    return a


def rmsnorm(x, w, eps=EPS):
    var = float((x * x).mean())
    return x * (1.0 / math.sqrt(var + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    # match engine: log1p(exp(x)), x>20 -> x
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def l2norm(x, eps=EPS):
    return x * (1.0 / math.sqrt(float((x * x).sum()) + eps))


def gdn_step(qn, kn, v, gdec, beta, S):
    """One recurrent step, S=[HK,HV]. Matches torch_recurrent_gated_delta_rule."""
    S = gdec * S
    kvm = S.T @ kn
    delta = (v - kvm) * beta
    S = S + np.outer(kn, delta)
    y = S.T @ qn
    return y, S


def ported_ref(d):
    """Recompute every intermediate from h_in + weights. Returns dict of arrays."""
    h_in = load(d + "/h_in.bin", HD)
    w_an = load(d + "/w_attn_norm.bin", HD)
    W_qkv = load(d + "/w_ssm_qkv.bin", CDIM * HD).reshape(CDIM, HD)
    W_gate = load(d + "/w_ssm_gate.bin", VDIM * HD).reshape(VDIM, HD)
    W_alpha = load(d + "/w_ssm_alpha.bin", NV * HD).reshape(NV, HD)
    W_beta = load(d + "/w_ssm_beta.bin", NV * HD).reshape(NV, HD)
    W_out = load(d + "/w_ssm_out.bin", HD * VDIM).reshape(HD, VDIM)
    W_conv = load(d + "/w_ssm_conv1d.bin", CK * CDIM)        # ggml [CK,CDIM] col-major: W[c,t]=data[t+c*CK]
    ssm_a = load(d + "/w_ssm_a.bin", NV)
    dt_bias = load(d + "/w_ssm_dt_bias.bin", NV)
    w_norm = load(d + "/w_ssm_norm.bin", HV)

    an = rmsnorm(h_in, w_an)
    qkv = W_qkv @ an
    zgate = W_gate @ an
    avec = W_alpha @ an   # a
    bvec = W_beta @ an    # b

    # conv1d: engine indexes W[c*CK + t] = data[t + c*CK]; state init zeros, shift toward 0,
    # insert newest at CK-1, out = sum_t W[c*CK+t]*state[t][c]; silu.
    state = np.zeros((CK, CDIM), np.float32)
    qkvc = np.zeros(CDIM, np.float32)
    for c in range(CDIM):
        for r in range(CK - 1):
            state[r, c] = state[r + 1, c]
        state[CK - 1, c] = qkv[c]
        y = 0.0
        for t in range(CK):
            y += W_conv[c * CK + t] * state[t, c]
        qkvc[c] = y * sigmoid(y)

    q_raw = qkvc[0:KDIM]
    k_raw = qkvc[KDIM:2 * KDIM]
    v_vec = qkvc[2 * KDIM:CDIM]

    sp = softplus(avec + dt_bias)
    g_logit = ssm_a * sp          # ssm_a is precomputed -exp(A_log)
    gdec = np.exp(g_logit)
    beta = sigmoid(bvec)

    qn = np.zeros((NV, HV), np.float32)
    kn = np.zeros((NV, HV), np.float32)
    for hk in range(NK):
        qn_h = l2norm(q_raw[hk * HV:(hk + 1) * HV])
        kn_h = l2norm(k_raw[hk * HV:(hk + 1) * HV])
        for r in range(REP):
            v = hk * REP + r
            qn[v] = qn_h * SCALE
            kn[v] = kn_h

    y_scan = np.zeros((NV, HV), np.float32)
    for v in range(NV):
        S0 = np.zeros((HV, HV), np.float32)
        y, _ = gdn_step(qn[v], kn[v], v_vec[v * HV:(v + 1) * HV], gdec[v], beta[v], S0)
        y_scan[v] = y

    # RMSNorm-gated: var=mean(y^2); y*=rsqrt(var+eps); y*=w_norm; y*=silu(z)
    y_ng = np.zeros((NV, HV), np.float32)
    for v in range(NV):
        yv = y_scan[v].copy()
        zv = zgate[v * HV:(v + 1) * HV]
        var = float((yv * yv).mean())
        yv = yv * (1.0 / math.sqrt(var + EPS)) * w_norm
        yv = yv * silu(zv)
        y_ng[v] = yv

    out = W_out @ y_ng.reshape(-1)
    h_out = h_in + out
    return dict(an=an, qkv=qkv, zgate=zgate, avec=avec, bvec=bvec, qkvc=qkvc,
                gdec=gdec, beta=beta, qn=qn, kn=kn, vvec=v_vec.reshape(NV, HV),
                y_scan=y_scan, y_normgated=y_ng, out=out, h_out=h_out)


def cmp(name, ref, eng, tol_abs=1e-4, tol_rel=1e-3):
    ref = np.asarray(ref, np.float32).reshape(-1)
    eng = np.asarray(eng, np.float32).reshape(-1)
    d = np.abs(ref - eng)
    rel = d / np.maximum(np.maximum(np.abs(ref), np.abs(eng)), 1e-6)
    ma, mr = float(d.max()), float(rel.max())
    ok = ma < tol_abs or mr < tol_rel
    flag = "OK" if ok else "DIVERGE"
    print(f"  [{flag}] {name:14s} max|Δ|={ma:.4e} maxrel={mr:.4e} (n={ref.size})")
    return ok, ma, mr


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ssm_dump"
    print(f"== ported reference vs engine dump ({d}) ==")
    R = ported_ref(d)
    E = dict(
        an=load(d + "/an.bin", HD),
        qkv=load(d + "/qkv.bin", CDIM), zgate=load(d + "/zgate.bin", VDIM),
        avec=load(d + "/avec.bin", NV), bvec=load(d + "/bvec.bin", NV),
        qkvc=load(d + "/qkvc.bin", CDIM),
        gdec=load(d + "/gdec.bin", NV), beta=load(d + "/beta.bin", NV),
        qn=load(d + "/qn.bin", NV * HV), kn=load(d + "/kn.bin", NV * HV),
        vvec=load(d + "/vvec.bin", NV * HV),
        y_scan=load(d + "/y_scan.bin", NV * HV),
        y_normgated=load(d + "/y_normgated.bin", NV * HV),
        out=load(d + "/out.bin", HD), h_out=load(d + "/h_out.bin", HD),
    )
    order = ["an", "qkv", "zgate", "avec", "bvec", "qkvc", "gdec", "beta",
             "qn", "kn", "vvec", "y_scan", "y_normgated", "out", "h_out"]
    first_div = None
    for k in order:
        ok, ma, mr = cmp(k, R[k], E[k])
        if not ok and first_div is None:
            first_div = k
    print(f"\nFIRST DIVERGENCE (ported ref vs engine): {first_div}")
    # h_out "divergence" is residual-stream quantization (16-bit storage), not a bug —
    # out matches to 3e-6. Always run the HF module ground-truth check to catch a
    # shared layout/orientation bug the ported ref cannot see.
    hf_module_check(d, R, E)


def hf_module_check(d, R, E):
    """Ground truth via transformers Qwen3NextGatedDeltaNet, staged comparison.

    Runs HF's own sub-methods step by step, dumping each intermediate, and compares
    to the engine dumps.  The first stage that diverges localizes the layout bug.
    Pre-conv q/k/v/z/b/a are compared to the engine's PRE-conv qkv/zgate/avec/bvec,
    which disambiguates reconstruction risk: if HF's grouped projection (after
    fix_ordering) reproduces the engine's flat projection, the reconstruction is
    consistent and the bug is downstream of the projection.
    """
    try:
        import torch
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextGatedDeltaNet, torch_causal_conv1d_update,
            torch_recurrent_gated_delta_rule, l2norm as hf_l2norm,
        )
    except Exception as e:
        print(f"  HF module check SKIPPED ({e})")
        return
    print(f"\n== HF Qwen3NextGatedDeltaNet staged ground truth ==")
    h_in = load(d + "/h_in.bin", HD)
    W_qkv = load(d + "/w_ssm_qkv.bin", CDIM * HD).reshape(CDIM, HD)
    W_gate = load(d + "/w_ssm_gate.bin", VDIM * HD).reshape(VDIM, HD)
    W_alpha = load(d + "/w_ssm_alpha.bin", NV * HD).reshape(NV, HD)
    W_beta = load(d + "/w_ssm_beta.bin", NV * HD).reshape(NV, HD)
    W_conv = load(d + "/w_ssm_conv1d.bin", CK * CDIM)
    ssm_a = load(d + "/w_ssm_a.bin", NV)        # = -exp(A_log)
    dt_bias = load(d + "/w_ssm_dt_bias.bin", NV)
    w_norm = load(d + "/w_ssm_norm.bin", HV)
    A_log = np.log(-ssm_a)

    cfg = Qwen3NextConfig(
        hidden_size=HD, num_hidden_layers=1,
        linear_num_value_heads=NV, linear_num_key_heads=NK,
        linear_key_head_dim=HV, linear_value_head_dim=HV,
        linear_conv_kernel_dim=CK, rms_norm_eps=EPS, hidden_act="silu",
        layer_types=["linear_attention"], num_attention_heads=1, num_key_value_heads=1,
        head_dim=HV, rope_theta=1e7, max_position_embeddings=32768,
    )
    mod = Qwen3NextGatedDeltaNet(cfg, 0).eval()
    # Reconstruct HF in_proj_qkvz [16384, HD] (grouped per k-head) from flat attn_qkv+attn_gate.
    proj_qkvz = np.zeros((2 * KDIM + 2 * VDIM, HD), np.float32)
    for g in range(NK):
        base = g * (2 * HV + 2 * (NV // NK) * HV)  # 1024 per group
        proj_qkvz[base + 0:        base + HV]       = W_qkv[0 * KDIM + g * HV : 0 * KDIM + (g + 1) * HV]   # q
        proj_qkvz[base + HV:       base + 2 * HV]   = W_qkv[1 * KDIM + g * HV : 1 * KDIM + (g + 1) * HV]   # k
        for r in range(REP):
            vh = g * REP + r
            proj_qkvz[base + 2 * HV + r * HV : base + 2 * HV + (r + 1) * HV] = W_qkv[2 * KDIM + vh * HV : 2 * KDIM + (vh + 1) * HV]   # v
            proj_qkvz[base + (2 + REP) * HV + r * HV : base + (2 + REP) * HV + (r + 1) * HV] = W_gate[vh * HV : (vh + 1) * HV]          # z
    # in_proj_ba [96, HD]: per group [b(3) | a(3)]
    proj_ba = np.zeros((2 * NV, HD), np.float32)
    for g in range(NK):
        for r in range(REP):
            vh = g * REP + r
            proj_ba[g * (2 * REP) + r]            = W_beta[vh]    # b
            proj_ba[g * (2 * REP) + REP + r]      = W_alpha[vh]   # a
    conv_w = np.zeros((CDIM, 1, CK), np.float32)
    for c in range(CDIM):
        for t in range(CK):
            conv_w[c, 0, t] = W_conv[c * CK + t]
    with torch.no_grad():
        mod.in_proj_qkvz.weight.copy_(torch.tensor(proj_qkvz))
        mod.in_proj_ba.weight.copy_(torch.tensor(proj_ba))
        mod.conv1d.weight.copy_(torch.tensor(conv_w))
        mod.dt_bias.copy_(torch.tensor(dt_bias))
        mod.A_log.copy_(torch.tensor(A_log))
        mod.norm.weight.copy_(torch.tensor(w_norm))
        mod.out_proj.weight.copy_(torch.tensor(load(d + "/w_ssm_out.bin", HD * VDIM).reshape(HD, VDIM)))
        # The GDN module receives the INPUT-LAYERNORMED hidden state (the decoder layer
        # applies input_layernorm before calling self_attn).  The engine projects `an =
        # rmsnorm(h_in, attn_norm)`, so feed the module the dumped normed input `an`, NOT
        # the raw residual h_in.  (h_in is still used for the residual-add comparison.)
        x = torch.tensor(E["an"]).reshape(1, 1, HD)

        # --- staged forward, mirroring mod.forward but exposing every intermediate ---
        pqkvz = mod.in_proj_qkvz(x)              # [1,1,16384]
        pba   = mod.in_proj_ba(x)                # [1,1,96]
        q, k, v, z, b, a = mod.fix_query_key_value_ordering(pqkvz, pba)
        q, k, v = (xx.reshape(xx.shape[0], xx.shape[1], -1) for xx in (q, k, v))
        # HF pre-conv projections (flat) vs engine PRE-conv qkv/zgate/avec/bvec
        hf_q, hf_k, hf_v = q.reshape(-1).numpy(), k.reshape(-1).numpy(), v.reshape(-1).numpy()
        hf_z, hf_b, hf_a = z.reshape(-1).numpy(), b.reshape(-1).numpy(), a.reshape(-1).numpy()
        eng_qkv = E["qkv"]
        cmp("HF q  vs eng qkv[0:2048]",   hf_q, eng_qkv[0:KDIM])
        cmp("HF k  vs eng qkv[2048:4096]", hf_k, eng_qkv[KDIM:2 * KDIM])
        cmp("HF v  vs eng qkv[4096:10240]", hf_v, eng_qkv[2 * KDIM:CDIM])
        cmp("HF z  vs eng zgate",          hf_z, E["zgate"])
        cmp("HF b  vs eng bvec",           hf_b, E["bvec"])
        cmp("HF a  vs eng avec",           hf_a, E["avec"])

        # --- conv1d (decode single-token path) ---
        mixed = torch.cat((q, k, v), dim=-1).transpose(1, 2)   # [1, CDIM, 1]
        conv_state = torch.zeros(1, CDIM, CK, dtype=mixed.dtype)
        mixed_conv = torch_causal_conv1d_update(mixed, conv_state, mod.conv1d.weight.squeeze(1), None, "silu")
        hf_qkvc = mixed_conv.reshape(-1).numpy()
        cmp("HF qkvc (post-conv) vs eng qkvc", hf_qkvc, E["qkvc"])

        # --- split, g/beta, repeat_interleave, scan ---
        hf_qc = hf_qkvc[0:KDIM]; hf_kc = hf_qkvc[KDIM:2 * KDIM]; hf_vc = hf_qkvc[2 * KDIM:CDIM]
        beta = torch.sigmoid(torch.tensor(hf_b)).reshape(1, 1, NV)          # [B,T,nV]
        g = (-torch.exp(torch.tensor(A_log))) * torch.nn.functional.softplus(
            torch.tensor(hf_a) + torch.tensor(dt_bias))
        g = g.reshape(1, 1, NV)                                              # [B,T,nV]
        # q/k: [B,T,NK,HV] -> repeat_interleave REP on dim=2 -> [B,T,NV,HV]
        qt = torch.tensor(hf_qc).reshape(1, 1, NK, HV).repeat_interleave(REP, dim=2)
        kt = torch.tensor(hf_kc).reshape(1, 1, NK, HV).repeat_interleave(REP, dim=2)
        vt = torch.tensor(hf_vc).reshape(1, 1, NV, HV)
        # recurrent expects [B,T,nV,HV]; internally transposes to [B,nV,T,HV]
        core, _ = torch_recurrent_gated_delta_rule(qt, kt, vt, g, beta,
                                                   None, False, use_qk_l2norm_in_kernel=True)
        hf_y = core.reshape(-1).numpy()                                      # core returns [B,T,nV,HV]
        cmp("HF y_scan vs eng y_scan", hf_y, E["y_scan"])

        # --- RMSNorm-gated + out_proj (both [B,T,nV,HV]) ---
        zt = torch.tensor(hf_z).reshape(1, 1, NV, HV)
        yng = mod.norm(core, zt)
        hf_yng = yng.reshape(-1).numpy()
        cmp("HF y_normgated vs eng y_normgated", hf_yng, E["y_normgated"])
        out = mod.out_proj(yng.reshape(1, 1, -1))
        hf_out = out.reshape(-1).numpy()
        cmp("HF out (o_proj) vs eng out", hf_out, E["out"])
        print(f"\n  engine out[:6]={E['out'][:6]}")
        print(f"  HF     out[:6]={hf_out[:6]}")
        dh = float(np.max(np.abs((h_in + hf_out) - E["h_out"])))
        print(f"  HF residual (h_in+out) vs eng h_out: max|Δ|={dh:.4e}")


if __name__ == "__main__":
    main()