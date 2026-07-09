#!/usr/bin/env python3
"""Pure-PyTorch reference of DeepSeek-V4-Flash-DSpark for a SINGLE BOS token (input_id=0).

Replaces the 6 tilelang kernels (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm,
sparse_attn, hc_split_sinkhorn) with mathematically-equivalent pure-torch code.
Loads real HF safetensors weights and runs the model forward in FLOAT32 (weights
dequantized to fp32; activations go through the SAME fp8/fp4 round-trip quantization
the HF model.py performs, so this faithfully reproduces the true HF model behavior).

For a single BOS at position 0:
  - RoPE is identity (cos=1, sin=0) -> skipped.
  - compress_ratio>0 layers: compressor returns None (seqlen=1 < ratio) and the
    indexer compress_topk_idxs is empty -> attention is over the single window KV
    position only, identical to compress_ratio=0 layers. So we skip compressor/indexer
    entirely (faithful: they contribute nothing for seqlen<ratio at start_pos=0).

Outputs per-layer residual maxabs/rms after attn and after ffn sublayers.
"""
import os, sys, json, math
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = "/home/raffaele/Progetti/FaStar/Source/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7"
MAX_LAYER = int(os.environ.get("HF_BOS_MAXLAYER", "42"))  # run 0..MAX_LAYER inclusive

# ---------- config ----------
DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM  # 448
Q_LORA = 1024
O_LORA = 1024
O_GROUPS = 8
MOE_INTER = 2048
N_EXPERTS = 256
N_ACTIVATED = 6
N_HASH_LAYERS = 3
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6
SOFTMAX_SCALE = HEAD_DIM ** -0.5
FP4_BLOCK = 32
FP8_BLOCK = 128
FP8_MAX = 448.0
COMPRESS_RATIOS = [0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]
VOCAB = 129280

FP4_LUT = torch.tensor([0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32)

# ---------- weight loading ----------
_index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
_weight_map = _index["weight_map"]
_shard_cache = {}  # path -> safe_open handle

def _shard_path(shard_name):
    return os.path.join(MODEL_DIR, shard_name)

def _open_shard(shard_name):
    if shard_name not in _shard_cache:
        _shard_cache[shard_name] = safe_open(_shard_path(shard_name), framework="pt")
    return _shard_cache[shard_name]

def get_tensor(key):
    shard = _weight_map[key]
    f = _open_shard(shard)
    return f.get_tensor(key)

# ---------- quantization helpers ----------
def e8m0_to_scale(s_e8m0):
    """e8m0 byte -> float scale = 2^(byte-127). byte 0 -> 2^-127 (essentially zero)."""
    b = s_e8m0.view(torch.uint8).to(torch.float32)
    return torch.pow(2.0, b - 127.0)

def fp4_dequant(weight_int8, scale_e8m0):
    """weight_int8: [N, K//2] int8 (fp4 packed 2-per-byte, low nibble first).
    scale_e8m0: [N, K//32] e8m0. Returns fp32 [N, K]."""
    N, Khalf = weight_int8.shape
    K = Khalf * 2
    byte = weight_int8.contiguous().view(torch.uint8).to(torch.int64)
    low = byte & 0xF            # [N, K//2]
    high = (byte >> 4) & 0xF   # [N, K//2]
    # interleave low,high along last dim -> [N, K]
    vals = torch.stack([low, high], dim=-1).reshape(N, K)
    vals_f = FP4_LUT[vals]  # [N, K]
    s = e8m0_to_scale(scale_e8m0)  # [N, K//32]
    s_exp = s.repeat_interleave(FP4_BLOCK, dim=1)  # [N, K]
    return vals_f * s_exp

def fp8_dequant(weight_fp8, scale_e8m0):
    """weight_fp8: [N, K] float8_e4m3fn. scale_e8m0: [N//128, K//128] e8m0.
    Returns fp32 [N, K]."""
    N, K = weight_fp8.shape
    w = weight_fp8.to(torch.float32)
    s = e8m0_to_scale(scale_e8m0)  # [N//128, K//128]
    s_exp = s.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)  # [N, K]
    return w * s_exp

def act_quant_rt(x, block):
    """Block-wise FP8(e4m3) quant+dequant round-trip with power-of-2 (e8m0) scale.
    Matches act_quant(inplace=True): s = 2^ceil(log2(amax/448)); quant clamp to [-448,448]."""
    orig_shape = x.shape
    K = orig_shape[-1]
    xb = x.reshape(-1, K).to(torch.float32)
    M = xb.size(0)
    nblk = K // block
    xblk = xb.reshape(M, nblk, block)
    amax = xblk.abs().amax(dim=-1)  # [M, nblk]
    amax = amax.clamp(min=1e-4)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / FP8_MAX)))  # power-of-2 scale
    s = s.clamp(min=2.0 ** -126)
    q = (xblk / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dq = q.to(torch.float32) * s.unsqueeze(-1)
    return dq.reshape(orig_shape)

def fp8_linear(x, weight_fp8, scale_e8m0):
    """Faithful fp8_gemm: act_quant(x,128) round-trip, then matmul with dequant weight."""
    x_q = act_quant_rt(x, FP8_BLOCK)  # [.., K] fp32 round-tripped
    W = fp8_dequant(weight_fp8, scale_e8m0)  # [N, K]
    return x_q @ W.T

def fp4_linear(x, weight_int8, scale_e8m0):
    """Faithful fp4_gemm: act_quant(x,128) round-trip, then matmul with dequant fp4 weight."""
    x_q = act_quant_rt(x, FP8_BLOCK)
    W = fp4_dequant(weight_int8, scale_e8m0)  # [N, K]
    return x_q @ W.T

# ---------- RMSNorm ----------
def rmsnorm(x, weight):
    var = x.square().mean(-1, keepdim=True)
    return (weight.to(torch.float32) * (x.to(torch.float32) * torch.rsqrt(var + NORM_EPS)))

# ---------- hc_split_sinkhorn (pure torch, matches kernel.py 372-427) ----------
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=HC_MULT, iters=HC_SINKHORN_ITERS, eps=HC_EPS):
    # mixes: [b,s, mix_hc] fp32. hc_scale [3], hc_base [mix_hc]
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])
    comb = mixes[..., 2*hc:].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[2*hc:].reshape(hc, hc)
    # row softmax (dim -1) + eps
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps
    # col normalize (dim -2)
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb

# ---------- hc_pre / hc_post ----------
def hc_pre(x, hc_fn, hc_scale, hc_base):
    # x: [b,s,hc,d]
    shape = x.shape
    xf = x.flatten(2).to(torch.float32)  # [b,s,hc*d]
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(xf, hc_fn) * rsqrt  # [b,s,mix_hc]
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)  # [b,s,d]
    return y, post, comb

def hc_post(x, residual, post, comb):
    # y[dst] = post[dst]*x + sum_src comb[src,dst]*residual[src]
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y  # [b,s,hc,d]

# ---------- sparse attention (single pos, pure torch, matches kernel.py) ----------
def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    # q: [b,s,h,d], kv: [b,n,d], attn_sink: [h], topk_idxs: [b,s,topk] (int, -1 = invalid)
    b, s, h, d = q.shape
    qf = q.to(torch.float32)
    kvf = kv.to(torch.float32)
    sinkf = attn_sink.to(torch.float32)
    out = torch.zeros(b, s, h, d, dtype=torch.float32)
    for bi in range(b):
        for si in range(s):
            idxs = topk_idxs[bi, si]
            valid = idxs[idxs >= 0]
            if valid.numel() == 0:
                continue
            kk = kvf[bi][valid]  # [t, d]
            scores = torch.einsum("hd,td->ht", qf[bi, si], kk) * scale  # [h, t]
            # online softmax with attn_sink
            scores_max = scores.amax(dim=-1, keepdim=True)  # [h,1]
            exp_s = torch.exp(scores - scores_max)  # [h,t]
            sum_exp = exp_s.sum(dim=-1, keepdim=True) + torch.exp(sinkf.unsqueeze(-1) - scores_max)  # [h,1]
            # weighted sum of kv
            # acc_o[h,d] = sum_t exp_s[h,t]*kk[t,d]
            acc_o = torch.einsum("ht,td->hd", exp_s, kk)  # [h,d]
            out[bi, si] = acc_o / sum_exp
    return out

# ---------- MLA attention for single BOS at pos 0 ----------
def attn_forward(x, L):
    # x: [1,1,4096] fp32. Returns [1,1,4096] fp32.
    p = f"layers.{L}.attn."
    wq_a_w = get_tensor(p + "wq_a.weight"); wq_a_s = get_tensor(p + "wq_a.scale")
    wq_b_w = get_tensor(p + "wq_b.weight"); wq_b_s = get_tensor(p + "wq_b.scale")
    wkv_w  = get_tensor(p + "wkv.weight");  wkv_s  = get_tensor(p + "wkv.scale")
    wo_a_w = get_tensor(p + "wo_a.weight"); wo_a_s = get_tensor(p + "wo_a.scale")
    wo_b_w = get_tensor(p + "wo_b.weight"); wo_b_s = get_tensor(p + "wo_b.scale")
    q_norm_w = get_tensor(p + "q_norm.weight")
    kv_norm_w = get_tensor(p + "kv_norm.weight")
    attn_sink = get_tensor(p + "attn_sink")

    qr = rmsnorm(fp8_linear(x, wq_a_w, wq_a_s), q_norm_w)  # [1,1,1024]
    q = fp8_linear(qr, wq_b_w, wq_b_s).view(1,1,N_HEADS,HEAD_DIM)  # [1,1,64,512]
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + NORM_EPS)
    # rope on last 64 dims at pos 0 -> identity, skip

    kv = rmsnorm(fp8_linear(x, wkv_w, wkv_s), kv_norm_w)  # [1,1,512]
    # rope on last 64 dims at pos 0 -> identity, skip
    # act_quant on non-rope dims (block 64, inplace round-trip)
    kv_nope = kv[..., :NOPE_HEAD_DIM]  # [1,1,448]
    kv_nope = act_quant_rt(kv_nope, 64)
    kv = torch.cat([kv_nope, kv[..., NOPE_HEAD_DIM:]], dim=-1)  # [1,1,512]

    # window topk for single BOS: [0]
    topk_idxs = torch.tensor([[0]], dtype=torch.int32)  # [1,1,1]
    # (compressor/indexer contribute nothing for seqlen=1 < ratio; skip faithfully)

    o = sparse_attn(q, kv, attn_sink, topk_idxs, SOFTMAX_SCALE)  # [1,1,64,512]
    # inverse rope on last 64 dims at pos 0 -> identity, skip

    # o projection: view [1,1,8,4096]; einsum with wo_a [8,1024,4096]; wo_b
    # FAITHFULNESS NOTE (CORRECTED): model.py declares wo_a as bfloat16 and uses wo_a.weight
    # DIRECTLY — TRUE, but convert.py:126 bakes the e8m0 block scale INTO the bf16 param at
    # convert time (weight = weight.float() * scale). So the bf16 param IS scaled. We load the
    # RAW fp8 checkpoint, so we MUST apply wo_a.scale via fp8_dequant (matching fst_converter
    # and convert.py). The prior "unscaled" note was wrong and inflated wo_a ~2000x.
    o = o.view(1,1,O_GROUPS,-1)  # [1,1,8,4096]
    wo_a = fp8_dequant(wo_a_w, wo_a_s).view(O_GROUPS, O_LORA, -1)  # [8,1024,4096] block-scaled
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)  # [1,1,8,1024]
    o = o.flatten(2)  # [1,1,8192]
    x_out = fp8_linear(o, wo_b_w, wo_b_s)  # [1,1,4096]
    return x_out

# ---------- MoE FFN ----------
def _expert_linear(x, w, s):
    """Dispatch on weight dtype: fp4 (int8 packed) -> fp4_linear; fp8 e4m3 -> fp8_linear."""
    if w.dtype == torch.int8:
        return fp4_linear(x, w, s)
    else:  # float8_e4m3fn
        return fp8_linear(x, w, s)

def expert_forward(x, w1_w, w1_s, w2_w, w2_s, w3_w, w3_s, weight=None):
    gate = _expert_linear(x, w1_w, w1_s)  # [1,1,inter]
    up = _expert_linear(x, w3_w, w3_s)
    if SWIGLU_LIMIT > 0:
        up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    h = F.silu(gate) * up
    if weight is not None:
        h = weight * h
    out = _expert_linear(h, w2_w, w2_s)  # [1,1,4096]
    return out

def ffn_forward(x, L, input_ids):
    p = f"layers.{L}.ffn."
    gate_w = get_tensor(p + "gate.weight")  # [256,4096] bf16
    # routing
    scores = (x.float() @ gate_w.float().T)  # [1,1,256] (x already fp32)
    scores = F.softplus(scores).sqrt()  # sqrtsoftplus
    original_scores = scores
    if L < N_HASH_LAYERS:
        tid2eid = get_tensor(p + "gate.tid2eid")  # [vocab,6] int64
        indices = tid2eid[input_ids.flatten()]  # [1,6]
    else:
        bias = get_tensor(p + "gate.bias")  # [256] fp32
        indices = (scores + bias).topk(N_ACTIVATED, dim=-1)[1]  # [1,1,6]
    # gather weights
    indices = indices.view(1, -1)  # [1,6]
    weights = original_scores.view(-1, 256).gather(1, indices)  # [1,6]
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    y = torch.zeros(1,1,DIM, dtype=torch.float32)
    for e in range(N_ACTIVATED):
        idx = int(indices[0, e].item())
        w = weights[0, e]
        ep = f"layers.{L}.ffn.experts.{idx}."
        w1_w = get_tensor(ep+"w1.weight"); w1_s = get_tensor(ep+"w1.scale")
        w2_w = get_tensor(ep+"w2.weight"); w2_s = get_tensor(ep+"w2.scale")
        w3_w = get_tensor(ep+"w3.weight"); w3_s = get_tensor(ep+"w3.scale")
        y = y + w * expert_forward(x, w1_w, w1_s, w2_w, w2_s, w3_w, w3_s)
    # shared expert (weight 1.0, no route_scale)
    sp = p + "shared_experts."
    sw1_w = get_tensor(sp+"w1.weight"); sw1_s = get_tensor(sp+"w1.scale")
    sw2_w = get_tensor(sp+"w2.weight"); sw2_s = get_tensor(sp+"w2.scale")
    sw3_w = get_tensor(sp+"w3.weight"); sw3_s = get_tensor(sp+"w3.scale")
    y = y + expert_forward(x, sw1_w, sw1_s, sw2_w, sw2_s, sw3_w, sw3_s)
    return y  # [1,1,4096]

# ---------- stats ----------
def stats(t):
    t = t.to(torch.float32)
    return float(t.abs().max().item()), float(torch.sqrt(t.square().mean()).item())

def main():
    print(f"[hf_bos_ref] running BOS (token 0) through layers 0..{MAX_LAYER}", flush=True)
    embed = get_tensor("embed.weight")  # [vocab,4096] bf16
    h0 = embed[0].to(torch.float32).view(1,1,DIM)  # [1,1,4096]
    # expand to 4 HC streams (identical copies)
    h = h0.unsqueeze(2).repeat(1,1,HC_MULT,1)  # [1,1,4,4096]
    print(f"[init] embed BOS: maxabs={stats(h0)[0]:.4f} rms={stats(h0)[1]:.4f}", flush=True)

    residuals = []
    exploded = False
    bounded = True
    for L in range(MAX_LAYER+1):
        try:
            attn_norm_w = get_tensor(f"layers.{L}.attn_norm.weight")
            ffn_norm_w  = get_tensor(f"layers.{L}.ffn_norm.weight")
            hc_attn_fn = get_tensor(f"layers.{L}.hc_attn_fn")
            hc_attn_scale = get_tensor(f"layers.{L}.hc_attn_scale")
            hc_attn_base = get_tensor(f"layers.{L}.hc_attn_base")
            hc_ffn_fn = get_tensor(f"layers.{L}.hc_ffn_fn")
            hc_ffn_scale = get_tensor(f"layers.{L}.hc_ffn_scale")
            hc_ffn_base = get_tensor(f"layers.{L}.hc_ffn_base")
        except KeyError as e:
            print(f"[L{L}] missing tensor {e}; stopping", flush=True)
            break

        residual = h
        x, post, comb = hc_pre(residual, hc_attn_fn.to(torch.float32), hc_attn_scale.to(torch.float32), hc_attn_base.to(torch.float32))
        x = rmsnorm(x, attn_norm_w)
        x = attn_forward(x, L)
        h = hc_post(x, residual, post, comb)
        a_maxabs, a_rms = stats(h)

        residual = h
        x, post, comb = hc_pre(residual, hc_ffn_fn.to(torch.float32), hc_ffn_scale.to(torch.float32), hc_ffn_base.to(torch.float32))
        x = rmsnorm(x, ffn_norm_w)
        x = ffn_forward(x, L, torch.tensor([[0]], dtype=torch.long))
        h = hc_post(x, residual, post, comb)
        f_maxabs, f_rms = stats(h)

        residuals.append({"layer": L, "attn_maxabs": a_maxabs, "attn_rms": a_rms, "ffn_maxabs": f_maxabs, "ffn_rms": f_rms})
        print(f"L{L:2d} after attn: maxabs={a_maxabs:12.4f} rms={a_rms:12.4f} | after ffn: maxabs={f_maxabs:12.4f} rms={f_rms:12.4f}", flush=True)
        if f_maxabs > 100:
            exploded = True
            bounded = False
        if f_maxabs > 1e30 or math.isnan(f_maxabs) or math.isinf(f_maxabs):
            print(f"[L{L}] NON-FINITE, stopping", flush=True)
            break
        # free per-layer shard handles we no longer need (keep memory bounded)
        # close shards for layers we've passed (keep current)
        # Actually keep handles; they're small. Experts tensors are freed by GC.
    last_layer = residuals[-1]["layer"] if residuals else -1

    output_token = None
    if last_layer == 42:
        try:
            # head: hc_head -> norm -> head.weight
            hc_head_fn = get_tensor("hc_head_fn").to(torch.float32)
            hc_head_base = get_tensor("hc_head_base").to(torch.float32)
            hc_head_scale = get_tensor("hc_head_scale").to(torch.float32)
            norm_w = get_tensor("norm.weight")
            head_w = get_tensor("head.weight").to(torch.float32)  # [vocab,4096]
            # hc_head: pre = sigmoid(mixes*scale + base) + eps; y = sum(pre * x)
            xf = h.flatten(2).to(torch.float32)
            rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
            mixes = F.linear(xf, hc_head_fn) * rsqrt
            pre = torch.sigmoid(mixes * hc_head_scale + hc_head_base) + HC_EPS
            hh = torch.sum(pre.unsqueeze(-1) * h, dim=2)  # [1,1,4096]
            hh = rmsnorm(hh, norm_w)
            logits = hh @ head_w.T  # [1,1,vocab]
            output_token = int(logits.argmax(dim=-1).item())
            print(f"[head] argmax token for BOS = {output_token}", flush=True)
        except Exception as e:
            print(f"[head] failed: {e}", flush=True)

    print("\n==== SUMMARY ====", flush=True)
    print(f"layers_run={len(residuals)} bounded={bounded} exploded={exploded} output_token={output_token}", flush=True)

    # write JSON for the orchestrator
    import json as _j
    result = {
        "layers_run": len(residuals),
        "residuals": residuals,
        "bounded": bounded,
        "exploded": exploded,
        "output_token": output_token,
        "notes": "Pure-pytorch faithful HF reproduction (fp32 compute, fp8/fp4 act+weight quantization reproduced). Single BOS at pos0: RoPE identity (skipped), compressor/indexer no-op (seqlen<ratio), attention over 1 KV window pos. swiglu_limit=10 kept; no other clamps added.",
        "script_path": "/home/raffaele/Progetti/FaStar/hf_bos_ref.py",
    }
    with open("/home/raffaele/Progetti/FaStar/hf_bos_ref_result.json", "w") as f:
        _j.dump(result, f, indent=2)
    print("result written to hf_bos_ref_result.json", flush=True)

if __name__ == "__main__":
    main()