#!/usr/bin/env python3
"""qwopus_ssm_recur_ref.py — RECURRENT validation of the SSM block across positions.

The single-position dump (pos 0) is bit-correct (qwopus_ssm_block_ref.py), but at
pos 0 the conv state and the GDN S state are both ZERO, so the state-CARRY path is
never exercised.  A wiring bug in how conv_state / S persist across positions is
invisible at pos 0 and catastrophic over a sequence (the empty-assistant-turn loop).

This script replays HF Qwen3NextGatedDeltaNet recurrently over the ENGINE's
per-position layer-0 residuals (dumped by FST_SSM_DUMP at every prefill position),
maintaining conv_state + recurrent_state across positions exactly as HF's decode
path does, and compares y_scan / y_normgated / out at EACH position.  The first
diverging position localizes a state-carry (wiring) bug.

Feed the module the engine's dumped `an_p` (post-rmsnorm) directly, so the rmsnorm
is not a source of divergence — only projection + conv1d-state + scan-state + normgated
+ o_proj are tested recurrently.

Usage: python3 qwopus_ssm_recur_ref.py /tmp/ssm_dump
"""
import sys, os, math, glob, numpy as np

HD, HV, NK, NV, REP, CK = 5120, 128, 16, 48, 3, 4
KDIM, VDIM, CDIM = NK * HV, NV * HV, 2 * (NK * HV) + (NV * HV)  # 2048, 6144, 10240
EPS = 1e-6


def load(path, n):
    a = np.fromfile(path, dtype=np.float32, count=n)
    if a.size != n:
        raise SystemExit(f"{path}: expected {n} floats, got {a.size}")
    return a


def positions(d):
    s = set()
    for f in glob.glob(d + "/h_in_*.bin"):
        s.add(int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0]))
    return sorted(s)


def cmp(name, ref, eng, tol_abs=1e-3, tol_rel=2e-2):
    ref = np.asarray(ref, np.float32).reshape(-1)
    eng = np.asarray(eng, np.float32).reshape(-1)
    d = np.abs(ref - eng)
    rel = d / np.maximum(np.maximum(np.abs(ref), np.abs(eng)), 1e-6)
    ma, mr = float(d.max()), float(rel.max())
    ok = ma < tol_abs or mr < tol_rel
    return ok, ma, mr


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ssm_dump"
    import torch
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet, torch_causal_conv1d_update,
        torch_recurrent_gated_delta_rule,
    )

    W_qkv = load(d + "/w_ssm_qkv.bin", CDIM * HD).reshape(CDIM, HD)
    W_gate = load(d + "/w_ssm_gate.bin", VDIM * HD).reshape(VDIM, HD)
    W_alpha = load(d + "/w_ssm_alpha.bin", NV * HD).reshape(NV, HD)
    W_beta = load(d + "/w_ssm_beta.bin", NV * HD).reshape(NV, HD)
    W_conv = load(d + "/w_ssm_conv1d.bin", CK * CDIM)
    ssm_a = load(d + "/w_ssm_a.bin", NV)        # = -exp(A_log)
    dt_bias = load(d + "/w_ssm_dt_bias.bin", NV)
    w_norm = load(d + "/w_ssm_norm.bin", HV)
    W_out = load(d + "/w_ssm_out.bin", HD * VDIM).reshape(HD, VDIM)
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
        proj_qkvz[base + 0:        base + HV]       = W_qkv[0 * KDIM + g * HV : 0 * KDIM + (g + 1) * HV]
        proj_qkvz[base + HV:       base + 2 * HV]   = W_qkv[1 * KDIM + g * HV : 1 * KDIM + (g + 1) * HV]
        for r in range(REP):
            vh = g * REP + r
            proj_qkvz[base + 2 * HV + r * HV : base + 2 * HV + (r + 1) * HV] = W_qkv[2 * KDIM + vh * HV : 2 * KDIM + (vh + 1) * HV]
            proj_qkvz[base + (2 + REP) * HV + r * HV : base + (2 + REP) * HV + (r + 1) * HV] = W_gate[vh * HV : (vh + 1) * HV]
    proj_ba = np.zeros((2 * NV, HD), np.float32)
    for g in range(NK):
        for r in range(REP):
            vh = g * REP + r
            proj_ba[g * (2 * REP) + r]            = W_beta[vh]
            proj_ba[g * (2 * REP) + REP + r]      = W_alpha[vh]
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
        mod.out_proj.weight.copy_(torch.tensor(W_out))

    pos = positions(d)
    print(f"== recurrent HF replay vs engine ({d}), {len(pos)} positions ==")
    conv_state = torch.zeros(1, CDIM, CK, dtype=torch.float32)
    rec_state = torch.zeros(1, NV, HV, HV, dtype=torch.float32)
    first_div = None
    with torch.no_grad():
        for p in pos:
            an_p = load(d + f"/an_{p}.bin", HD)
            x = torch.tensor(an_p).reshape(1, 1, HD)
            pqkvz = mod.in_proj_qkvz(x)
            pba = mod.in_proj_ba(x)
            q, k, v, z, b, a = mod.fix_query_key_value_ordering(pqkvz, pba)
            q, k, v = (xx.reshape(xx.shape[0], xx.shape[1], -1) for xx in (q, k, v))
            mixed = torch.cat((q, k, v), dim=-1).transpose(1, 2)   # [1, CDIM, 1]
            mixed_conv = torch_causal_conv1d_update(mixed, conv_state,
                                                    mod.conv1d.weight.squeeze(1), None, "silu")
            qkv_c = mixed_conv.reshape(-1).numpy()
            qc = qkv_c[0:KDIM]; kc = qkv_c[KDIM:2 * KDIM]; vc = qkv_c[2 * KDIM:CDIM]
            beta = torch.sigmoid(b.reshape(-1)).reshape(1, 1, NV)
            g = (-torch.exp(torch.tensor(A_log))) * torch.nn.functional.softplus(
                a.reshape(-1) + torch.tensor(dt_bias))
            g = g.reshape(1, 1, NV)
            qt = torch.tensor(qc).reshape(1, 1, NK, HV).repeat_interleave(REP, dim=2)
            kt = torch.tensor(kc).reshape(1, 1, NK, HV).repeat_interleave(REP, dim=2)
            vt = torch.tensor(vc).reshape(1, 1, NV, HV)
            core, rec_state = torch_recurrent_gated_delta_rule(
                qt, kt, vt, g, beta, rec_state, True, use_qk_l2norm_in_kernel=True)
            hf_y = core.reshape(-1).numpy()
            zt = z.reshape(1, 1, NV, HV)
            hf_yng = mod.norm(core, zt).reshape(-1).numpy()
            hf_out = mod.out_proj(torch.tensor(hf_yng).reshape(1, 1, -1)).reshape(-1).numpy()

            eng_y = load(d + f"/y_scan_{p}.bin", NV * HV)
            eng_yng = load(d + f"/y_normgated_{p}.bin", NV * HV)
            eng_out = load(d + f"/out_{p}.bin", HD)
            eng_s_pre = load(d + f"/S_pre_{p}.bin", NV * HV * HV) if os.path.exists(d + f"/S_pre_{p}.bin") else None
            eng_s_post = load(d + f"/S_post_{p}.bin", NV * HV * HV) if os.path.exists(d + f"/S_post_{p}.bin") else None
            hf_s_pre = rec_state.reshape(-1).numpy().copy()
            ok_y,   my,   ry = cmp(f"y_scan",   hf_y,   eng_y)
            ok_yng, myng, rng = cmp(f"y_ng",    hf_yng, eng_yng)
            ok_out, mout, ro = cmp(f"out",      hf_out, eng_out)
            ok_spre = ok_spost = True; mspre = mspost = 0.0
            if eng_s_pre is not None:
                ok_spre, mspre, _ = cmp(f"S_pre", hf_s_pre, eng_s_pre, tol_abs=1e-2, tol_rel=1e-2)
            if eng_s_post is not None:
                ok_spost, mspost, _ = cmp(f"S_post", rec_state.reshape(-1).numpy(), eng_s_post, tol_abs=1e-2, tol_rel=1e-2)
            tag = "OK" if (ok_y and ok_yng and ok_out and ok_spre and ok_spost) else "DIVERGE"
            print(f"  pos {p:2d} [{tag}] y max|Δ|={my:.3e} | y_ng {myng:.3e} | out {mout:.3e} | "
                  f"S_pre {mspre:.3e} | S_post {mspost:.3e}")
            if not (ok_y and ok_yng and ok_out and ok_spre and ok_spost) and first_div is None:
                first_div = p
    print(f"\nFIRST DIVERGING POSITION: {first_div}")
    if first_div is None:
        print("SSM block is bit-correct recurrently across ALL dumped positions.")
        print("=> The incoherence is NOT in the SSM block. Look in GQA / FFN / MTP / lm_head / template.")


if __name__ == "__main__":
    main()