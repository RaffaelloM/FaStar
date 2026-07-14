#!/usr/bin/env python3
"""Qwen3.5-Next (qwen35) GQA + dense-FFN reference, for finding the downstream
incoherence bug.

WHY THIS FILE EXISTS
  The SSM (Gated Delta Net) block is proven bit-correct vs HF transformers, yet the
  engine's output is incoherent (`\\n<|im_start|>assistant<|im_end|>` loop).  The bug
  is therefore downstream: in the full-attention (GQA) layers, the dense SwiGLU FFN,
  the MTP/NextN head, lm_head, or the chat template.  This script isolates the FIRST
  GQA layer (Layer 3, the lowest L%4==3) at decode position 0.

METHOD (weight-identical, no HF model load needed)
  The engine, when run with FST_GQA_DUMP=<dir>, writes for L3 pos 0:
    - every intermediate tensor (residual_in, attn_norm_out, q/k/v/gate projections,
      post-q/k-norm, post-rope, attn scores/softmax/out, gated, o_proj, residuals,
      and the full FFN chain) as fp32 .bin
    - the RAW MXFP4 weight bytes for q/k/v/o/gate/up/down + the bf16/fp32 norms
    - L3_meta.txt with hd/nq/nkv/dh/n_rot/eps/rope_freq_base/scale
  This script dequants the dumped MXFP4 weights in numpy (same 17-byte-block layout
  the engine's mxfp4_matvec_f32 uses), recomputes the full GQA+FFN block with the
  EXACT transformers Qwen3NextAttention / Qwen3NextMLP math, and compares each stage
  to the engine dump.  The first diverging tensor is the bug.

  This is stronger than loading HF `qwen3_next` because it uses the EXACT weights
  and input the engine uses (the .fst is GGUF-derived; an HF model would be a
  different weight snapshot).  It tests the engine's MATH against an independent
  numpy implementation of the HF-spec math, on identical data.

HF MATH CONTRACT (transformers/models/qwen3_next/modeling_qwen3_next.py):
  Qwen3NextAttention:
    q_proj: hidden -> num_q*hd*2  (DOUBLED, view [..,num_q,hd*2], chunk(2,-1) -> q|gate)
    q_norm over head_dim (per-head, shared weight), k_norm over head_dim (shared)
    v_proj: NO norm
    scaling = head_dim**-0.5 = 1/sqrt(256) = 1/16
    RoPE: partial_rotary_factor=0.25 -> dim=64; NeoX rotate-half over first 64 dims,
          pair (j, j+32), j=0..31; dims [64,256) passthrough. freq_j = base^(-2j/64).
    GQA causal attention (num_q/num_kv = 6 q-heads per kv-head)
    attn_output *= sigmoid(gate)
    o_proj: num_q*hd -> hidden
  Qwen3NextMLP:
    down(silu(gate(x)) * up(x))
  Residuals accumulated in bf16 (engine stores h as bf16).

Usage:
  python3 scripts/qwopus_gqa_ref.py <dump_dir>   # default logs/gqa_dump
"""
import math
import os
import sys
import struct
import numpy as np

FP4_TABLE = np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
                      0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], np.float32)


def dequant_mxfp4(raw: bytes, out_dim: int, in_dim: int) -> np.ndarray:
    """Inverse of mxfp4_matvec_f32 / dequant_mxfp4_dense.  W[out,in] row-major;
    17-byte blocks of 32 along the IN dim: [e8m0 scale | 16 nibble bytes].
    element[i] = FP4_TABLE[nibble_i] * 2^(e8m0-127); e8m0==0 -> dead block -> 0."""
    groups = in_dim // 32
    raw = np.frombuffer(raw, np.uint8)
    assert raw.size == out_dim * groups * 17, f"{raw.size} != {out_dim}*{groups}*17"
    raw = raw.reshape(out_dim, groups, 17)
    sc = raw[:, :, 0].astype(np.int32)
    scale = np.where(sc == 0, 0.0, np.power(2.0, (sc - 127).astype(np.float32)))  # [out,groups]
    nb = raw[:, :, 1:17]                          # [out,groups,16]
    # 32 nibbles: for i in 0..31, byte=nb[i>>1], nib = (i&1)? high : low  ->
    # byte0_low,byte0_high,byte1_low,byte1_high,...  i.e. even idx = low, odd idx = high.
    even = (nb[:, :, 0:16] & 0x0F).astype(np.int32)          # nibbles 0,2,4,...,30
    odd  = ((nb[:, :, 0:16] >> 4) & 0x0F).astype(np.int32)    # nibbles 1,3,5,...,31
    nib = np.empty(raw.shape[:2] + (32,), np.int32)
    nib[:, :, 0::2] = even
    nib[:, :, 1::2] = odd
    vals = FP4_TABLE[nib] * scale[:, :, None]                # [out,groups,32]
    return vals.reshape(out_dim, in_dim).astype(np.float32)


def rmsnorm(x, w, eps):
    ss = float(np.dot(x, x)) / x.shape[0]
    return x * (1.0 / math.sqrt(ss + eps)) * w


def rope_head(x, pos, n_rot, base):
    """NeoX rotate-half over first n_rot dims (pair j,j+n_rot/2), passthrough rest."""
    half = n_rot // 2
    out = x.copy()
    for j in range(half):
        freq = 1.0 / (base ** (2.0 * j / n_rot))
        ang = pos * freq
        cf, sf = math.cos(ang), math.sin(ang)
        a, b = x[j], x[j + half]
        out[j]           = a * cf - b * sf
        out[j + half]    = b * cf + a * sf
    return out


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else "logs/gqa_dump"
    dump = os.path.abspath(dump)

    # meta
    meta = {}
    with open(os.path.join(dump, "L3_meta.txt")) as f:
        for line in f:
            parts = line.split()
            for i in range(0, len(parts) - 1, 2):
                meta[parts[i]] = parts[i + 1]
    lid = int(meta["lid"]); pos = int(meta["pos"])
    hd = int(meta["hd"]); nq = int(meta["nq"]); nkv = int(meta["nkv"]); dh = int(meta["dh"])
    n_rot = int(meta["n_rot"]); inter = int(meta["inter"])
    eps = float(meta["rms_eps"]); base = float(meta["rope_freq_base"]); scale = float(meta["scale"])
    qrow = nq * dh        # 6144
    kvrow = nkv * dh      # 1024
    print(f"[meta] L{lid} pos {pos} hd={hd} nq={nq} nkv={nkv} dh={dh} n_rot={n_rot} "
          f"inter={inter} eps={eps} base={base} scale={scale}")

    def load_f32(name):
        return np.fromfile(os.path.join(dump, f"L{lid}_p{pos}_{name}.bin"), np.float32)

    def load_raw(name):
        with open(os.path.join(dump, f"L{lid}_{name}.bin"), "rb") as f:
            return f.read()

    # ---- weights ----
    w_attn_norm = np.fromfile(os.path.join(dump, f"L{lid}_w_attn_norm.bin"), np.float32)
    w_ffn_norm  = np.fromfile(os.path.join(dump, f"L{lid}_w_ffn_norm.bin"), np.float32)
    w_q_norm    = np.fromfile(os.path.join(dump, f"L{lid}_w_q_norm.bin"), np.float32)
    w_k_norm    = np.fromfile(os.path.join(dump, f"L{lid}_w_k_norm.bin"), np.float32)
    Wq = dequant_mxfp4(load_raw("w_q_proj.mxfp4"), nq * dh * 2, hd)   # [12288,5120]
    Wk = dequant_mxfp4(load_raw("w_k_proj.mxfp4"), nkv * dh, hd)       # [1024,5120]
    Wv = dequant_mxfp4(load_raw("w_v_proj.mxfp4"), nkv * dh, hd)
    Wo = dequant_mxfp4(load_raw("w_o_proj.mxfp4"), hd, nq * dh)        # [5120,6144]
    Wg = dequant_mxfp4(load_raw("w_ffn_gate.mxfp4"), inter, hd)
    Wu = dequant_mxfp4(load_raw("w_ffn_up.mxfp4"), inter, hd)
    Wd = dequant_mxfp4(load_raw("w_ffn_down.mxfp4"), hd, inter)
    print(f"[w] Wq{Wq.shape} Wk{Wk.shape} Wo{Wo.shape} Wg{Wg.shape} Wd{Wd.shape}")

    # ---- input residual (fp32) ----
    h_in = load_f32("residual_in")
    assert h_in.size == hd

    results = {}   # name -> np.float32 array (ref)
    diffs = []     # (name, max_abs, max_rel)

    def cmp(name, ref, eng=None):
        if eng is None:
            eng = load_f32(name)
        if eng.shape != ref.shape:
            print(f"  !! {name}: SHAPE MISMATCH eng={eng.shape} ref={ref.shape}")
            return
        d = np.abs(eng - ref)
        mx = float(d.max()) if d.size else 0.0
        denom = np.maximum(np.abs(ref), 1e-6)
        rel = float((d / denom).max()) if d.size else 0.0
        diffs.append((name, mx, rel, eng, ref))
        tag = "OK " if mx < 1e-3 else "XX "
        print(f"  {tag}{name:24s} max|Δ|={mx:.4e} rel={rel:.4e}")

    # 1. attn_norm_out
    an = rmsnorm(h_in, w_attn_norm, eps)
    results["attn_norm_out"] = an
    cmp("attn_norm_out", an)

    # 2. q/k/v/gate projections.  C++ q_proj output is [h0_q(256)|h0_gate(256)|...].
    qg = Wq @ an                              # [12288]
    q = np.empty(qrow, np.float32); gate = np.empty(qrow, np.float32)
    for hq in range(nq):
        qp = qg[hq * 2 * dh : hq * 2 * dh + 2 * dh]
        q[hq * dh : (hq + 1) * dh]    = qp[:dh]
        gate[hq * dh : (hq + 1) * dh] = qp[dh:]
    k = (Wk @ an).astype(np.float32)          # [1024]
    v = (Wv @ an).astype(np.float32)
    cmp("q_proj", q); cmp("q_gate", gate); cmp("k_proj", k); cmp("v_proj", v)

    # 3. per-head q/k RMSNorm (shared q_norm/k_norm over head_dim)
    for hq in range(nq):
        q[hq * dh : (hq + 1) * dh] = rmsnorm(q[hq * dh : (hq + 1) * dh], w_q_norm, eps)
    for hk in range(nkv):
        k[hk * dh : (hk + 1) * dh] = rmsnorm(k[hk * dh : (hk + 1) * dh], w_k_norm, eps)
    cmp("q_postqknorm", q); cmp("k_postqknorm", k)

    # 4. RoPE (n_rot over first 64 dims, passthrough [64,256))
    for hq in range(nq):
        q[hq * dh : (hq + 1) * dh] = rope_head(q[hq * dh : (hq + 1) * dh], pos, n_rot, base)
    for hk in range(nkv):
        k[hk * dh : (hk + 1) * dh] = rope_head(k[hk * dh : (hk + 1) * dh], pos, n_rot, base)
    cmp("q_postrope", q); cmp("k_postrope", k)

    # 5. attention (M=1, causal, pos 0 -> nkeys=1).  GQA: 6 q-heads per kv-head.
    # Engine stores K/V in a BF16 cache and reads bf16f(f2bf(x)) — replicate that
    # round-trip here, otherwise the score/out divergence is just our missing quant.
    def bf16(x):  # engine f2bf is TRUNCATION: (bf16_t)(bits>>16), then bf16f = bits<<16.
        u = x.astype(np.float32).view(np.uint32)
        u = u & np.uint32(0xffff0000)        # drop low 16 mantissa bits (no rounding)
        return u.view(np.float32)
    k = bf16(k)
    v = bf16(v)
    nkeys = pos + 1
    group = nq // nkv
    kcache = k.reshape(nkv, dh)     # [nkv,dh]
    vcache = v.reshape(nkv, dh)
    out = np.zeros(qrow, np.float32)
    sc_dump = np.zeros((nq, nkeys), np.float32)
    pr_dump = np.zeros((nq, nkeys), np.float32)
    for hq in range(nq):
        kvh = hq // group
        qh = q[hq * dh : (hq + 1) * dh]
        scores = np.zeros(nkeys, np.float32)
        for kk in range(nkeys):
            kr = kcache[kvh]                       # only one key at pos 0
            s = float(np.dot(qh, kr)) * scale
            scores[kk] = s
        mx = scores.max()
        p = np.exp(scores - mx)
        p = p / (p.sum() + 1e-20)
        oh = np.zeros(dh, np.float32)
        for kk in range(nkeys):
            pv = p[kk]
            oh += pv * vcache[kvh]
        out[hq * dh : (hq + 1) * dh] = oh
        sc_dump[hq] = scores
        pr_dump[hq] = p
    cmp("attn_scores", sc_dump.reshape(-1))
    cmp("attn_softmax", pr_dump.reshape(-1))
    cmp("attn_out", out)

    # 6. gate + o_proj + residual
    gated = out * (1.0 / (1.0 + np.exp(-gate)))
    cmp("attn_gated", gated)
    o = (Wo @ gated).astype(np.float32)
    cmp("o_proj", o)
    # residual stored bf16: engine h_out = bf16(bf16(h_in) + o).  h_in was already bf16-cast
    # at dump time (residual_in = bf16f(h)), so residual_out = bf16(h_in + o).
    h_after_attn = bf16(h_in + o)
    cmp("attn_residual_out", h_after_attn)

    # ---- FFN ----
    h_ffn_in = h_after_attn   # engine: process_ffn_q35 receives h (bf16) after attn residual
    an_f = rmsnorm(h_ffn_in, w_ffn_norm, eps)
    cmp("ffn_norm_out", an_f, eng=load_f32("ffn_norm_out"))
    g = (Wg @ an_f).astype(np.float32)
    u = (Wu @ an_f).astype(np.float32)
    cmp("ffn_gate", g); cmp("ffn_up", u)
    s = g * (1.0 / (1.0 + np.exp(-g))) * u
    cmp("ffn_swiglu", s)
    ffn_out = (Wd @ s).astype(np.float32)
    cmp("ffn_down", ffn_out)
    h_after_ffn = bf16(h_ffn_in + ffn_out)
    cmp("ffn_residual_out", h_after_ffn)

    # ---- verdict ----
    print("\n=== first diverging tensor (max|Δ|>1e-3) ===")
    first = next((d for d in diffs if d[1] > 1e-3), None)
    if first is None:
        print("  NONE — GQA + FFN are bit-correct vs the numpy HF-spec reference.")
        print("  Bug is NOT in GQA/FFN.  Look in MTP/NextN, lm_head, final norm, or the chat template.")
    else:
        name, mx, rel, eng, ref = first
        print(f"  FIRST DIVERGE: {name}  max|Δ|={mx:.4e} rel={rel:.4e}")
        i = int(np.argmax(np.abs(eng - ref)))
        print(f"  worst index {i}: eng={eng[i]:.6f} ref={ref[i]:.6f}")
        print(f"  eng[0:6]={eng[:6]}")
        print(f"  ref[0:6]={ref[:6]}")


if __name__ == "__main__":
    main()