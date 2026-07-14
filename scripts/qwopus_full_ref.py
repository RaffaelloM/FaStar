#!/usr/bin/env python3
"""qwopus_full_ref.py — FULL-TRUNK numpy reference for Qwen3.5-Next (qwopus.fst).

Reads weights DIRECTLY from qwopus.fst (no HF model load, no engine dumps) and
runs the complete 64-trunk-layer forward pass for a 1-token prompt, then final
RMSNorm + lm_head -> next-token logits.  Prints top-5 logits and dumps the
residual stream after selected layers + the final h_prenorm / hn / logits to
logs/ref_dump/ so they can be compared against the C++ engine's
FST_HEAD_DUMP output (head_p0_h_prenorm.f32 / head_p0_hn.f32 / head_p0_logits.f32).

WHY: SSM, GQA and FFN are each proven bit-correct BLOCK-level at pos 0, yet the
engine echoes the input ("Hello" -> token 9419, residual-dominance).  The bug is
in TRUNK COMPOSITION: how the 64 layers chain via the residual stream.  This
script is the independent ground truth for that composition.

At pos 0, RoPE / conv1d-state / GDN S-state / KV-cache-multi-key are all no-ops
or single-self-key, so a 1-token prompt isolates PURE composition (residual
adds, norm application, layer-type routing, final norm, lm_head).

Math contracts (pure numpy, ported from qwopus_ssm_block_ref.py + qwopus_gqa_ref.py,
which were validated bit-correct vs HF transformers block-level):
  - RMSNorm: x * rsqrt(mean(x^2)+eps) * w
  - SSM (Gated Delta Net): rmsnorm -> qkv/gate/alpha/beta proj -> conv1d(silu) ->
    softplus(a+dt_bias)*ssm_a -> gdec, sigmoid(b)->beta -> l2norm q/k, repeat 3x ->
    delta-rule recurrent (S=0 at pos0) -> rmsnorm-gated(y,w_norm,silu(z)) -> out_proj
  - GQA: rmsnorm -> q_proj(doubled, q|gate) / k / v -> per-head q/k RMSNorm -> RoPE
    (noop at pos0) -> causal self-attn (1 key) -> out*sigmoid(gate) -> o_proj
  - FFN (every layer): rmsnorm(post_attn_norm) -> down(silu(gate(x))*up(x))
  - Residual stored bf16 TRUNCATION (engine h is bf16): h = bf16(h + sub)

Usage:
  python3 scripts/qwopus_full_ref.py [--model qwopus.fst] [--token 9419] [--dump logs/ref_dump]
"""
import argparse, math, os, struct, sys
import numpy as np

# ── FST format constants (mirror fst_converter.py) ──────────────────────────
HEADER_FMT = "<4sIIIIIIIIIIIQffQQQQQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SHARED_ENTRY_FMT = "<IHHHHIQQQQQQ"
SHARED_ENTRY_SIZE = struct.calcsize(SHARED_ENTRY_FMT)   # 64
FST_MAGIC = b"FST\x00"
QTYPE_Q8_0, QTYPE_BF16, QTYPE_F32, QTYPE_MXFP4 = 1, 2, 3, 4

TID_EMBED, TID_OUTPUT_NORM, TID_LM_HEAD = 0, 1, 2
TID_INPUT_NORM, TID_POST_ATTN_NORM = 3, 4
TID_Q_PROJ, TID_K_PROJ, TID_V_PROJ, TID_O_PROJ = 5, 6, 7, 8
TID_Q35_Q_NORM, TID_Q35_K_NORM = 60, 61
TID_Q35_FFN_GATE, TID_Q35_FFN_UP, TID_Q35_FFN_DOWN = 62, 63, 64
TID_Q35_SSM_QKV, TID_Q35_SSM_GATE = 65, 66
TID_Q35_SSM_CONV1D, TID_Q35_SSM_A = 67, 68
TID_Q35_SSM_ALPHA, TID_Q35_SSM_BETA = 69, 70
TID_Q35_SSM_DT_BIAS, TID_Q35_SSM_NORM, TID_Q35_SSM_OUT = 71, 72, 73
TID_Q35_LAYER_TYPES, TID_Q35_CFG = 78, 79

GLOBAL = 0xFFFF   # GLOBAL_LAYER in fst_converter.py

FP4_TABLE = np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
                      0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], np.float32)


# ── FST reader ──────────────────────────────────────────────────────────────
class FST:
    def __init__(self, path):
        self.f = open(path, "rb")
        hdr = struct.unpack_from(HEADER_FMT, self.f.read(HEADER_SIZE))
        assert hdr[0] == FST_MAGIC, f"bad magic {hdr[0]!r}"
        (self.magic, self.version, self.hidden, self.num_layers, self.num_experts,
         self.top_k, self.nq, self.nkv, self.head_dim, self.inter, self.n_shared_exp,
         _reserved, self.vocab, self.rms_eps, self.rope_base,
         self.dir_off, self.dir_count, self.bank_off, self.block_bytes,
         self.block_stride, self.expert_count, self.r1, self.r2) = hdr
        # read shared directory
        self.entries = {}   # (tid, layer) -> (qtype, ndim, shape0, shape1, shape2, off, len)
        self.f.seek(self.dir_off)
        raw = self.f.read(self.dir_count * SHARED_ENTRY_SIZE)
        for i in range(self.dir_count):
            e = struct.unpack_from(SHARED_ENTRY_FMT, raw, i * SHARED_ENTRY_SIZE)
            tid, lid, sub, qtype, ndim, _r, s0, s1, s2, off, length, _r2 = e
            self.entries[(tid, lid)] = (qtype, ndim, s0, s1, s2, off, length)
        # q35 config blob
        cfg = self.read_array(TID_Q35_CFG, GLOBAL)   # 23 floats
        self.cfg = {
            "hidden": int(cfg[0]), "num_layers": int(cfg[1]), "nq": int(cfg[2]),
            "nkv": int(cfg[3]), "head_dim": int(cfg[4]), "inter": int(cfg[5]),
            "vocab": int(cfg[6]), "rms_eps": float(cfg[7]), "rope_base": float(cfg[8]),
            "full_attn_interval": int(cfg[9]), "nextn_layers": int(cfg[10]),
            "ssm_state": int(cfg[11]), "ssm_group": int(cfg[12]), "ssm_conv": int(cfg[13]),
            "ssm_dt_rank": int(cfg[14]), "ssm_inner": int(cfg[15]),
            "rope_sec": [int(cfg[16]), int(cfg[17]), int(cfg[18]), int(cfg[19])],
            "n_ssm": int(cfg[20]), "n_full": int(cfg[21]), "n_mtp": int(cfg[22]),
        }
        self.layer_types = self.read_array(TID_Q35_LAYER_TYPES, GLOBAL).astype(int)  # [65]

    def _raw(self, tid, layer):
        meta = self.entries.get((tid, layer))
        if meta is None:
            raise KeyError(f"tensor tid={tid} layer={layer} not in .fst dir")
        qtype, ndim, s0, s1, s2, off, length = meta
        self.f.seek(off)
        return self.f.read(length), qtype, ndim, (s0, s1, s2)

    def read_array(self, tid, layer):
        raw, qtype, ndim, sh = self._raw(tid, layer)
        if qtype == QTYPE_F32:
            return np.frombuffer(raw, np.float32).copy()
        if qtype == QTYPE_BF16:
            u = np.frombuffer(raw, np.uint16)
            return (u.astype(np.uint32) << 16).view(np.float32).copy()
        raise ValueError(f"read_array: tid={tid} not F32/BF16 (qtype={qtype})")

    def read_mxfp4(self, tid, layer):
        """Return dequanted fp32 [out, in] for an MXFP4 weight."""
        raw, qtype, ndim, sh = self._raw(tid, layer)
        assert qtype == QTYPE_MXFP4, f"tid={tid} not MXFP4 (qtype={qtype})"
        out, in_ = int(sh[0]), int(sh[1])
        return dequant_mxfp4(raw, out, in_)

    def read_vec(self, tid, layer):
        """1-D F32/BF16 vector (norms, biases)."""
        return self.read_array(tid, layer)


def dequant_mxfp4(raw, out_dim, in_dim):
    """W[out,in] row-major; 17-byte blocks of 32 along IN: [e8m0|16 nibble bytes].
    element[i] = FP4_TABLE[nibble_i] * 2^(e8m0-127); e8m0==0 -> 0."""
    groups = in_dim // 32
    a = np.frombuffer(raw, np.uint8)
    assert a.size == out_dim * groups * 17, f"{a.size} != {out_dim}*{groups}*17"
    a = a.reshape(out_dim, groups, 17)
    sc = a[:, :, 0].astype(np.int32)
    scale = np.where(sc == 0, 0.0, np.power(2.0, (sc - 127).astype(np.float32)))
    nb = a[:, :, 1:17]
    even = (nb & 0x0F).astype(np.int32)
    odd = ((nb >> 4) & 0x0F).astype(np.int32)
    nib = np.empty((out_dim, groups, 32), np.int32)
    nib[:, :, 0::2] = even
    nib[:, :, 1::2] = odd
    vals = FP4_TABLE[nib] * scale[:, :, None]
    return vals.reshape(out_dim, in_dim).astype(np.float32)


# ── numerics ────────────────────────────────────────────────────────────────
def bf16(x):
    """Engine bf16 = TRUNCATION: drop low 16 mantissa bits (no rounding)."""
    u = np.asarray(x, np.float32).view(np.uint32)
    return (u & np.uint32(0xffff0000)).view(np.float32)


def rmsnorm(x, w, eps):
    var = float((x * x).mean())
    return (x * (1.0 / math.sqrt(var + eps)) * w).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0)))).astype(np.float32)


def l2norm(x, eps=1e-6):
    return (x * (1.0 / math.sqrt(float((x * x).sum()) + eps))).astype(np.float32)


# ── SSM (Gated Delta Net) sublayer ──────────────────────────────────────────
# Constants from qwen35 config: HV=128, NK=16, NV=48, REP=3, CK=4
HV, NK, NV, REP, CK = 128, 16, 48, 3, 4
KDIM, VDIM = NK * HV, NV * HV             # 2048, 6144
CDIM = 2 * KDIM + VDIM                    # 10240
SCALE = 1.0 / math.sqrt(HV)


def ssm_sublayer(an, W):
    """Return the SSM block output (to be added to residual). Pure numpy, pos-0."""
    qkv = W["qkv"] @ an                    # [10240]
    zgate = W["gate"] @ an                 # [6144]
    avec = W["alpha"] @ an                 # [48]
    bvec = W["beta"] @ an                  # [48]
    # conv1d (state=zeros at pos0 -> only newest tap CK-1 contributes), then silu
    wconv = W["conv"]                      # [CK, CDIM] col-major: W[c*CK+t]
    qkvc = np.zeros(CDIM, np.float32)
    for c in range(CDIM):
        y = wconv[c * CK + (CK - 1)] * qkv[c]      # state[0..CK-2]=0 at pos0
        qkvc[c] = y * sigmoid(y)
    q_raw = qkvc[0:KDIM]
    k_raw = qkvc[KDIM:2 * KDIM]
    v_vec = qkvc[2 * KDIM:CDIM]
    sp = softplus(avec + W["dt_bias"])
    gdec = np.exp(W["ssm_a"] * sp)         # ssm_a = -exp(A_log)
    beta = sigmoid(bvec)
    qn = np.zeros((NV, HV), np.float32)
    kn = np.zeros((NV, HV), np.float32)
    for hk in range(NK):
        qn_h = l2norm(q_raw[hk * HV:(hk + 1) * HV]) * SCALE
        kn_h = l2norm(k_raw[hk * HV:(hk + 1) * HV])
        for r in range(REP):
            v = hk * REP + r
            qn[v] = qn_h
            kn[v] = kn_h
    # recurrent delta rule, S=0 at pos0
    y_scan = np.zeros((NV, HV), np.float32)
    for vidx in range(NV):
        S = np.zeros((HV, HV), np.float32)
        S = gdec[vidx] * S
        delta = (v_vec[vidx * HV:(vidx + 1) * HV]) * beta[vidx]   # kvm=0 at pos0
        S = S + np.outer(kn[vidx], delta)
        y_scan[vidx] = S.T @ qn[vidx]
    # RMSNorm-gated
    y_ng = np.zeros((NV, HV), np.float32)
    for vidx in range(NV):
        yv = y_scan[vidx]
        var = float((yv * yv).mean())
        yv = yv * (1.0 / math.sqrt(var + 1e-6)) * W["norm"]
        y_ng[vidx] = yv * silu(zgate[vidx * HV:(vidx + 1) * HV])
    return (W["out"] @ y_ng.reshape(-1)).astype(np.float32)        # [5120]


# ── GQA sublayer ────────────────────────────────────────────────────────────
def rope_head(x, pos, n_rot, base):
    half = n_rot // 2
    out = x.copy()
    for j in range(half):
        freq = 1.0 / (base ** (2.0 * j / n_rot))
        ang = pos * freq
        cf, sf = math.cos(ang), math.sin(ang)
        a, b = x[j], x[j + half]
        out[j] = a * cf - b * sf
        out[j + half] = b * cf + a * sf
    return out.astype(np.float32)


def gqa_sublayer(an, W, pos, nq, nkv, dh, n_rot, base, eps):
    qrow = nq * dh
    qg = W["q"] @ an                       # [nq*dh*2]
    q = np.empty(qrow, np.float32); gate = np.empty(qrow, np.float32)
    for hq in range(nq):
        qp = qg[hq * 2 * dh : hq * 2 * dh + 2 * dh]
        q[hq * dh : (hq + 1) * dh] = qp[:dh]
        gate[hq * dh : (hq + 1) * dh] = qp[dh:]
    k = (W["k"] @ an).astype(np.float32)
    v = (W["v"] @ an).astype(np.float32)
    for hq in range(nq):
        q[hq * dh : (hq + 1) * dh] = rmsnorm(q[hq * dh : (hq + 1) * dh], W["q_norm"], eps)
    for hk in range(nkv):
        k[hk * dh : (hk + 1) * dh] = rmsnorm(k[hk * dh : (hk + 1) * dh], W["k_norm"], eps)
    for hq in range(nq):
        q[hq * dh : (hq + 1) * dh] = rope_head(q[hq * dh : (hq + 1) * dh], pos, n_rot, base)
    for hk in range(nkv):
        k[hk * dh : (hk + 1) * dh] = rope_head(k[hk * dh : (hk + 1) * dh], pos, n_rot, base)
    k = bf16(k); v = bf16(v)               # engine KV cache is bf16
    group = nq // nkv
    nkeys = pos + 1
    scale = 1.0 / math.sqrt(dh)
    out = np.zeros(qrow, np.float32)
    for hq in range(nq):
        kvh = hq // group
        qh = q[hq * dh : (hq + 1) * dh]
        scores = np.zeros(nkeys, np.float32)
        for kk in range(nkeys):
            scores[kk] = float(np.dot(qh, k[kvh * dh : (kvh + 1) * dh])) * scale
        p = np.exp(scores - scores.max())
        p = p / (p.sum() + 1e-20)
        oh = np.zeros(dh, np.float32)
        for kk in range(nkeys):
            oh += p[kk] * v[kvh * dh : (kvh + 1) * dh]
        out[hq * dh : (hq + 1) * dh] = oh
    gated = out * sigmoid(gate)
    return (W["o"] @ gated).astype(np.float32)


# ── FFN sublayer (every layer) ──────────────────────────────────────────────
def ffn_sublayer(x, W):
    g = W["ffn_gate"] @ x
    u = W["ffn_up"] @ x
    s = silu(g) * u
    return (W["ffn_down"] @ s).astype(np.float32)


# ── per-layer weight load ───────────────────────────────────────────────────
def load_layer_weights(fst, L, kind):
    W = {}
    W["attn_norm"] = fst.read_vec(TID_INPUT_NORM, L)
    W["ffn_norm"] = fst.read_vec(TID_POST_ATTN_NORM, L)
    W["ffn_gate"] = fst.read_mxfp4(TID_Q35_FFN_GATE, L)
    W["ffn_up"] = fst.read_mxfp4(TID_Q35_FFN_UP, L)
    W["ffn_down"] = fst.read_mxfp4(TID_Q35_FFN_DOWN, L)
    if kind == 0:  # SSM
        W["qkv"] = fst.read_mxfp4(TID_Q35_SSM_QKV, L)
        W["gate"] = fst.read_mxfp4(TID_Q35_SSM_GATE, L)
        W["conv"] = fst.read_vec(TID_Q35_SSM_CONV1D, L)
        W["ssm_a"] = fst.read_vec(TID_Q35_SSM_A, L)
        W["alpha"] = fst.read_mxfp4(TID_Q35_SSM_ALPHA, L)
        W["beta"] = fst.read_mxfp4(TID_Q35_SSM_BETA, L)
        W["dt_bias"] = fst.read_vec(TID_Q35_SSM_DT_BIAS, L)
        W["norm"] = fst.read_vec(TID_Q35_SSM_NORM, L)
        W["out"] = fst.read_mxfp4(TID_Q35_SSM_OUT, L)
    else:          # GQA (kind 1) — MTP (kind 2) not in trunk
        W["q"] = fst.read_mxfp4(TID_Q_PROJ, L)
        W["k"] = fst.read_mxfp4(TID_K_PROJ, L)
        W["v"] = fst.read_mxfp4(TID_V_PROJ, L)
        W["o"] = fst.read_mxfp4(TID_O_PROJ, L)
        W["q_norm"] = fst.read_vec(TID_Q35_Q_NORM, L)
        W["k_norm"] = fst.read_vec(TID_Q35_K_NORM, L)
    return W


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwopus.fst")
    ap.add_argument("--token", type=int, default=9419, help="token id for 'Hello'")
    ap.add_argument("--dump", default="logs/ref_dump")
    ap.add_argument("--layers", default="0,1,2,10,30,63",
                    help="comma list of layer indices whose post-ffn residual to dump")
    args = ap.parse_args()
    os.makedirs(args.dump, exist_ok=True)
    dump_layers = set(int(x) for x in args.layers.split(",") if x.strip())

    fst = FST(args.model)
    c = fst.cfg
    HD = c["hidden"]
    eps = c["rms_eps"]
    nq, nkv, dh = c["nq"], c["nkv"], c["head_dim"]
    n_rot = 64
    base = c["rope_base"]
    n_trunk = c["num_layers"] - c["n_mtp"]   # 65 - 1 = 64
    print(f"[cfg] hidden={HD} trunk={n_trunk} vocab={c['vocab']} "
          f"nq={nq} nkv={nkv} dh={dh} inter={c['inter']} eps={eps} base={base}")
    print(f"[cfg] layer_types[0:65] = {list(fst.layer_types[:65])}")
    print(f"[cfg] n_ssm={c['n_ssm']} n_full={c['n_full']} n_mtp={c['n_mtp']}")

    # ---- embedding lookup ----
    emb_raw, qtype, ndim, sh = fst._raw(TID_EMBED, GLOBAL)
    s0, s1 = int(sh[0]), int(sh[1])
    print(f"[emb] embed shape=({s0},{s1}) qtype={qtype}  token={args.token}")
    assert qtype == QTYPE_MXFP4
    if s0 == HD:           # [hidden, vocab]  -> column
        emb = dequant_mxfp4(emb_raw, HD, c["vocab"])     # [HD, vocab] (5 GB)
        h = emb[:, args.token].astype(np.float32)
        del emb
    else:                  # [vocab, hidden]  -> row
        emb = dequant_mxfp4(emb_raw, c["vocab"], HD)
        h = emb[args.token, :].astype(np.float32)
        del emb
    import gc; gc.collect()
    h = bf16(h)            # engine stores residual as bf16
    print(f"[emb] h0 norm={float(np.linalg.norm(h)):.4f} meanabs={float(np.abs(h).mean()):.4f}")
    h.tofile(os.path.join(args.dump, "ref_h0.f32"))

    # ---- trunk ----
    for L in range(n_trunk):
        kind = int(fst.layer_types[L])
        W = load_layer_weights(fst, L, kind)
        an = rmsnorm(h, W["attn_norm"], eps)
        if kind == 0:
            sub = ssm_sublayer(an, W)
        else:
            sub = gqa_sublayer(an, W, 0, nq, nkv, dh, n_rot, base, eps)
        h = bf16(h + sub)                          # attn residual (bf16 storage)
        fn = rmsnorm(h, W["ffn_norm"], eps)
        fout = ffn_sublayer(fn, W)
        h = bf16(h + fout)                         # ffn residual (bf16 storage)
        if L in dump_layers or L == n_trunk - 1:
            h.astype(np.float32).tofile(os.path.join(args.dump, f"ref_h_L{L}.f32"))
            print(f"[L{L:2d}] kind={kind} h norm={float(np.linalg.norm(h)):.4f} "
                  f"meanabs={float(np.abs(h).mean()):.4f} maxabs={float(np.abs(h).max()):.4f}")
        del W; gc.collect()

    # ---- final norm + lm_head ----
    h_prenorm = h.astype(np.float32).copy()
    out_norm = fst.read_vec(TID_OUTPUT_NORM, GLOBAL)
    hn = rmsnorm(h, out_norm, eps)
    lm = fst.read_mxfp4(TID_LM_HEAD, GLOBAL)            # [vocab, hidden]
    logits = (lm @ hn).astype(np.float32)
    del lm; gc.collect()
    h_prenorm.tofile(os.path.join(args.dump, "ref_h_prenorm.f32"))
    hn.tofile(os.path.join(args.dump, "ref_hn.f32"))
    logits.tofile(os.path.join(args.dump, "ref_logits.f32"))
    print(f"\n[head] h_prenorm norm={float(np.linalg.norm(h_prenorm)):.4f} "
          f"hn norm={float(np.linalg.norm(hn)):.4f}")
    top = np.argsort(logits)[::-1][:5]
    print("[head] top-5 logits:")
    for i, t in enumerate(top):
        print(f"  top{i}: id={t} logit={float(logits[t]):.4f}")
    print(f"[head] logit[{args.token}] (input token) = {float(logits[args.token]):.4f}")


if __name__ == "__main__":
    main()