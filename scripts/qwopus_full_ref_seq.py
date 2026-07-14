#!/usr/bin/env python3
"""qwopus_full_ref_seq.py — multi-token prefill reference with STATE CARRY.

Layer-major forward over a sequence of token ids, carrying:
  - SSM (GDN): conv1d buffer [CK, CDIM] + recurrent S state [NV, HV, HV] per SSM layer.
  - GQA: KV cache (accumulate k/v per position; causal attention over keys 0..pos).
This exercises the pos>0 paths that the 1-token (pos-0) reference cannot: RoPE at
nonzero position, SSM conv/S state carry, and GQA MULTI-KEY attention.

Layer-major (load each layer's weights once, sweep all positions) is mathematically
identical to position-major (the engine's loop) because each layer's state is updated
sequentially across positions and h[p] at layer L only depends on h[p] at layer L-1.

Compares the final-position logits against the C++ engine's FST_HEAD_DUMP
(head_p0_logits.f32 = first head call = after prefill = pos n-1).

Usage:
  python3 scripts/qwopus_full_ref_seq.py --tokens 248045,846,198,9419,11,1092,513,488,30,248046,198,248045,74455,198,248068,271,248069,271 --dump logs/ref_dump_tmpl
"""
import argparse, math, os, gc
import numpy as np
import qwopus_full_ref as R
from qwopus_full_ref import (FST, GLOBAL, dequant_mxfp4, bf16, rmsnorm, silu, sigmoid,
                             softplus, l2norm, rope_head, load_layer_weights,
                             HV, NK, NV, REP, CK, KDIM, VDIM, CDIM, SCALE,
                             TID_EMBED, TID_OUTPUT_NORM, TID_LM_HEAD)


def ssm_step_stateful(an, W, conv_st, S_st):
    """Update conv_st [CK,CDIM] and S_st [NV,HV,HV] in place; return block out [hd]."""
    qkv = W["qkv"] @ an
    zgate = W["gate"] @ an
    avec = W["alpha"] @ an
    bvec = W["beta"] @ an
    # vectorized conv1d with state
    conv_st[:-1, :] = conv_st[1:, :]          # shift toward 0 (drop oldest)
    conv_st[CK - 1, :] = qkv                  # newest at CK-1
    Wc = W["conv"].reshape(CDIM, CK).T        # [CK, CDIM]: Wc[r,c]=Wconv[c*CK+r]
    y = (Wc * conv_st).sum(axis=0)            # [CDIM]
    qkvc = y * sigmoid(y)
    q_raw = qkvc[0:KDIM]; k_raw = qkvc[KDIM:2 * KDIM]; v_vec = qkvc[2 * KDIM:CDIM]
    sp = softplus(avec + W["dt_bias"])
    gdec = np.exp(W["ssm_a"] * sp)
    beta = sigmoid(bvec)
    qn = np.zeros((NV, HV), np.float32); kn = np.zeros((NV, HV), np.float32)
    for hk in range(NK):
        qn_h = l2norm(q_raw[hk * HV:(hk + 1) * HV]) * SCALE
        kn_h = l2norm(k_raw[hk * HV:(hk + 1) * HV])
        for r in range(REP):
            v = hk * REP + r
            qn[v] = qn_h; kn[v] = kn_h
    y_scan = np.zeros((NV, HV), np.float32)
    for vi in range(NV):
        S = gdec[vi] * S_st[vi]
        kvm = S.T @ kn[vi]
        delta = (v_vec[vi * HV:(vi + 1) * HV] - kvm) * beta[vi]
        S = S + np.outer(kn[vi], delta)
        S_st[vi] = S
        y_scan[vi] = S.T @ qn[vi]
    # RMSNorm-gated
    y_ng = np.zeros((NV, HV), np.float32)
    for vi in range(NV):
        yv = y_scan[vi]
        var = float((yv * yv).mean())
        yv = yv * (1.0 / math.sqrt(var + 1e-6)) * W["norm"]
        y_ng[vi] = yv * silu(zgate[vi * HV:(vi + 1) * HV])
    return (W["out"] @ y_ng.reshape(-1)).astype(np.float32)


def gqa_step_stateful(an, W, pos, kv, nq, nkv, dh, n_rot, base, eps):
    qrow = nq * dh; kvrow = nkv * dh
    qg = W["q"] @ an
    q = np.empty(qrow, np.float32); gate = np.empty(qrow, np.float32)
    for hq in range(nq):
        qp = qg[hq * 2 * dh : hq * 2 * dh + 2 * dh]
        q[hq * dh:(hq + 1) * dh] = qp[:dh]
        gate[hq * dh:(hq + 1) * dh] = qp[dh:]
    k = (W["k"] @ an).astype(np.float32)
    v = (W["v"] @ an).astype(np.float32)
    for hq in range(nq):
        q[hq * dh:(hq + 1) * dh] = rmsnorm(q[hq * dh:(hq + 1) * dh], W["q_norm"], eps)
    for hk in range(nkv):
        k[hk * dh:(hk + 1) * dh] = rmsnorm(k[hk * dh:(hk + 1) * dh], W["k_norm"], eps)
    for hq in range(nq):
        q[hq * dh:(hq + 1) * dh] = rope_head(q[hq * dh:(hq + 1) * dh], pos, n_rot, base)
    for hk in range(nkv):
        k[hk * dh:(hk + 1) * dh] = rope_head(k[hk * dh:(hk + 1) * dh], pos, n_rot, base)
    k = bf16(k); v = bf16(v)
    kv["k"][pos] = k           # append (bf16-truncated)
    kv["v"][pos] = v
    group = nq // nkv
    nkeys = pos + 1
    scale = 1.0 / math.sqrt(dh)
    out = np.zeros(qrow, np.float32)
    for hq in range(nq):
        kvh = hq // group
        qh = q[hq * dh:(hq + 1) * dh]
        scores = np.zeros(nkeys, np.float32)
        for kk in range(nkeys):
            scores[kk] = float(np.dot(qh, kv["k"][kk][kvh * dh:(kvh + 1) * dh])) * scale
        p = np.exp(scores - scores.max()); p = p / (p.sum() + 1e-20)
        oh = np.zeros(dh, np.float32)
        for kk in range(nkeys):
            oh += p[kk] * kv["v"][kk][kvh * dh:(kvh + 1) * dh]
        out[hq * dh:(hq + 1) * dh] = oh
    gated = out * sigmoid(gate)
    return (W["o"] @ gated).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwopus.fst")
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--dump", default="logs/ref_dump_seq")
    ap.add_argument("--n-rot", type=int, default=64)
    ap.add_argument("--layers", default="3,7,15,31,63",
                    help="GQA layers whose pos-1 attention to print (for multi-key check)")
    ap.add_argument("--decode", type=int, default=0,
                    help="greedy decode steps after prefill (feed back argmax, carry state)")
    args = ap.parse_args()
    os.makedirs(args.dump, exist_ok=True)
    toks = [int(x) for x in args.tokens.split(",")]
    print(f"[seq] {len(toks)} tokens: {toks}")

    fst = FST(args.model)
    c = fst.cfg
    HD = c["hidden"]; eps = c["rms_eps"]
    nq, nkv, dh = c["nq"], c["nkv"], c["head_dim"]
    base = c["rope_base"]
    n_trunk = c["num_layers"] - c["n_mtp"]
    print(f"[cfg] hidden={HD} trunk={n_trunk} nq={nq} nkv={nkv} dh={dh} base={base}")

    # embeddings for all tokens (dequant embed once, take rows)
    emb_raw, qtype, ndim, sh = fst._raw(TID_EMBED, GLOBAL)
    s0, s1 = int(sh[0]), int(sh[1])
    Vocab = c["vocab"]
    if s0 == HD:
        emb = dequant_mxfp4(emb_raw, HD, Vocab); H = [bf16(emb[:, t].astype(np.float32)) for t in toks]; del emb
    else:
        emb = dequant_mxfp4(emb_raw, Vocab, HD); H = [bf16(emb[t, :].astype(np.float32)) for t in toks]; del emb
    gc.collect()
    print(f"[emb] h0 norm={float(np.linalg.norm(H[0])):.4f}  h_last norm={float(np.linalg.norm(H[-1])):.4f}")

    # per-layer state containers
    conv_state = {}   # L -> [CK, CDIM]
    S_state = {}      # L -> [NV, HV, HV]
    kv_cache = {}     # L -> {"k": [nmax], "v": [nmax]}

    # layer-major
    for L in range(n_trunk):
        kind = int(fst.layer_types[L])
        W = load_layer_weights(fst, L, kind)
        if kind == 0:
            conv_state[L] = np.zeros((CK, CDIM), np.float32)
            S_state[L] = np.zeros((NV, HV, HV), np.float32)
        else:
            kv_cache[L] = {"k": [None] * len(toks), "v": [None] * len(toks)}
        for p in range(len(toks)):
            an = rmsnorm(H[p], W["attn_norm"], eps)
            if kind == 0:
                sub = ssm_step_stateful(an, W, conv_state[L], S_state[L])
            else:
                sub = gqa_step_stateful(an, W, p, kv_cache[L], nq, nkv, dh, args.n_rot, base, eps)
            H[p] = bf16(H[p] + sub)
            fn = rmsnorm(H[p], W["ffn_norm"], eps)
            fout = (W["ffn_down"] @ (silu(W["ffn_gate"] @ fn) * (W["ffn_up"] @ fn))).astype(np.float32)
            H[p] = bf16(H[p] + fout)
        if L in (0, 1, 2, 10, 30, 63):
            print(f"[L{L:2d}] kind={kind} h_last norm={float(np.linalg.norm(H[-1])):.4f} "
                  f"maxabs={float(np.abs(H[-1]).max()):.4f}")
        del W; gc.collect()

    # final norm + lm_head on LAST position
    h_last = H[-1].astype(np.float32)
    out_norm = fst.read_vec(TID_OUTPUT_NORM, GLOBAL)
    lm = fst.read_mxfp4(TID_LM_HEAD, GLOBAL)

    def head_logits(hp):
        hn = rmsnorm(hp, out_norm, eps)
        return (lm @ hn).astype(np.float32), hn

    logits, hn = head_logits(h_last)
    h_last.tofile(os.path.join(args.dump, "ref_h_prenorm.f32"))
    hn.tofile(os.path.join(args.dump, "ref_hn.f32"))
    logits.tofile(os.path.join(args.dump, "ref_logits.f32"))
    print(f"\n[head] h_prenorm norm={float(np.linalg.norm(h_last)):.4f} hn norm={float(np.linalg.norm(hn)):.4f}")
    top = np.argsort(logits)[::-1][:8]
    print("[head] top-8 logits:")
    for i, t in enumerate(top):
        print(f"  top{i}: id={t} logit={float(logits[t]):.4f}")

    if args.decode > 0:
        # greedy decode, position-major, continuing from carried state.
        # State dicts (conv_state/S_state/kv_cache) hold post-prefill values.
        cur = int(top[0])
        print(f"\n[decode] start tid={cur} (greedy, {args.decode} steps)")
        for step in range(args.decode):
            pos = len(toks) + step          # abs position of this new token
            emb_raw2, _, _, sh2 = fst._raw(TID_EMBED, GLOBAL)
            s0_, s1_ = int(sh2[0]), int(sh2[1])
            if s0_ == HD:
                e2 = dequant_mxfp4(emb_raw2, HD, c["vocab"]); h_new = bf16(e2[:, cur].astype(np.float32)); del e2
            else:
                e2 = dequant_mxfp4(emb_raw2, c["vocab"], HD); h_new = bf16(e2[cur, :].astype(np.float32)); del e2
            for L in range(n_trunk):
                kind = int(fst.layer_types[L])
                W = load_layer_weights(fst, L, kind)
                an = rmsnorm(h_new, W["attn_norm"], eps)
                if kind == 0:
                    sub = ssm_step_stateful(an, W, conv_state[L], S_state[L])
                else:
                    kv_cache[L]["k"].append(None); kv_cache[L]["v"].append(None)
                    sub = gqa_step_stateful(an, W, pos, kv_cache[L], nq, nkv, dh, args.n_rot, base, eps)
                h_new = bf16(h_new + sub)
                fn = rmsnorm(h_new, W["ffn_norm"], eps)
                fout = (W["ffn_down"] @ (silu(W["ffn_gate"] @ fn) * (W["ffn_up"] @ fn))).astype(np.float32)
                h_new = bf16(h_new + fout)
                del W
            lg2, _ = head_logits(h_new)
            cur = int(np.argmax(lg2))
            t5 = np.argsort(lg2)[::-1][:5]
            print(f"[decode] step{step+1} pos={pos} -> tid={cur} top5={[(int(t),round(float(lg2[t]),2)) for t in t5]}")
            gc.collect()
    del lm; gc.collect()
    return logits


if __name__ == "__main__":
    main()