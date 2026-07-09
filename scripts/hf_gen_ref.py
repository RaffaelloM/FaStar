#!/usr/bin/env python3
"""Pure-PyTorch GENERATION reference for DeepSeek-V4-Flash-DSpark.

Extends hf_bos_ref.py (single-BOS residual proof) to actually GENERATE tokens for a
real chat prompt, so we can compare the HF ground-truth output + first-decode logits
against the FaStar engine and localize the incoherence bug.

Ground truth = inference/model.py + inference/kernel.py (the deployed `generate.py`
path) with ALL 6 tilelang kernels replaced by mathematically-equivalent pure-torch:
  act_quant / fp4_act_quant / fp8_gemm / fp4_gemm / sparse_attn / hc_split_sinkhorn.
Real HF safetensors weights, float32 compute, fp8/fp4 act+weight round-trip reproduced.

Faithfulness locked in from inference/config.json (the config generate.py loads):
  - scale_fmt = ue8m0  -> power-of-2 act-quant scales  (act_quant_rt)
  - original_seq_len = 65536, rope_factor = 16, beta_fast=32, beta_slow=1  -> YaRN
    (compress layers base=160000; L0/L1 base=10000, original_seq_len=0 -> no YaRN).
    This MATCHES FaStar's build_rope_lut, so RoPE is NOT a divergence.
  - swiglu_limit=10, route_scale=1.5, n_hash_layers=3, window=128, index_topk=512.

KV Compressor (ratio=4 at L2,L4,...): ported FAITHFULLY from model.py Compressor
(overlap=True, ape, wkv/wgate bf16, norm, overlap_transform, incremental decode state).
The learned Indexer (ratio==4) is replaced by get_compress_topk_idxs — faithful because
n_comp = seqlen//ratio << index_topk=512 for our short prompt, so ALL compressed rows
are selected (causally masked); the learned scorer cannot drop any row below index_topk.
Toggle compressor via env HF_COMPRESSOR (default "1"=ON, faithful to deployed model;
"0" matches FaStar's current gated-off behavior for an A/B comparison).

Chat template (matches fst_main.cpp + encoding_dsv4.py):
  "<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>ab"   (ab = think_end = token 128822)
Prefill M = 14 tokens, then decode 5 tokens.

Outputs: per-step raw top-5 logits (token id + value) for prefill-last (pos 13) and each
decode position (14..18), greedy 5-token text, and a temp=0.6 sampled 5-token text.
"""
import os, sys, json, math
import numpy as _np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = "/home/raffaele/Progetti/FaStar/Source/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7"
PROMPT = os.environ.get("HF_PROMPT", "The importance of NPU technology in modern laptops is")
N_GEN = int(os.environ.get("HF_NGEN", "5"))
TEMPERATURE = float(os.environ.get("HF_TEMP", "0.6"))
USE_COMPRESSOR = os.environ.get("HF_COMPRESSOR", "1") == "1"
GREEDY = os.environ.get("HF_GREEDY", "1") == "1"   # deterministic modal trajectory (primary)
# seed for the sampled (non-greedy) trajectory, for reproducibility
SAMPLE_SEED = int(os.environ.get("HF_SAMPLE_SEED", "1234"))

# ---------- config (from inference/config.json) ----------
DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM   # 448
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
VOCAB = 129280
WINDOW = 128
INDEX_TOPK = 512
N_LAYERS = 43
COMPRESS_RATIOS = [0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]
# YaRN (inference/config.json: original_seq_len=65536, rope_factor=16)
ORIG_SEQ_LEN = 65536
ROPE_FACTOR = 16.0
BETA_FAST = 32
BETA_SLOW = 1
ROPE_THETA = 10000.0        # L0/L1 (compress_ratio==0)
COMPRESS_ROPE_THETA = 160000.0  # L2+ (compress_ratio>0)
# buffer sizes (our gen uses <=22 positions; sizes only affect allocation, not math)
MAX_SEQ = 4096
THINK_END_ID = 128822       # "ab" / </think>

FP4_LUT = torch.tensor([0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32)

# ---------- weight loading (from hf_bos_ref.py, verbatim) ----------
_index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
_weight_map = _index["weight_map"]
_shard_cache = {}

def _shard_path(shard_name): return os.path.join(MODEL_DIR, shard_name)
def _open_shard(shard_name):
    if shard_name not in _shard_cache:
        _shard_cache[shard_name] = safe_open(_shard_path(shard_name), framework="pt")
    return _shard_cache[shard_name]
def get_tensor(key):
    shard = _weight_map[key]
    return _open_shard(shard).get_tensor(key)

# ---------- quantization helpers (from hf_bos_ref.py, verbatim) ----------
def e8m0_to_scale(s_e8m0):
    b = s_e8m0.view(torch.uint8).to(torch.float32)
    return torch.pow(2.0, b - 127.0)

def fp4_dequant(weight_int8, scale_e8m0):
    N, Khalf = weight_int8.shape
    K = Khalf * 2
    byte = weight_int8.contiguous().view(torch.uint8).to(torch.int64)
    low = byte & 0xF
    high = (byte >> 4) & 0xF
    vals = torch.stack([low, high], dim=-1).reshape(N, K)
    vals_f = FP4_LUT[vals]
    s = e8m0_to_scale(scale_e8m0)
    s_exp = s.repeat_interleave(FP4_BLOCK, dim=1)
    return vals_f * s_exp

def fp8_dequant(weight_fp8, scale_e8m0):
    N, K = weight_fp8.shape
    w = weight_fp8.to(torch.float32)
    s = e8m0_to_scale(scale_e8m0)
    s_exp = s.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    return w * s_exp

def act_quant_rt(x, block):
    """Block FP8(e4m3) quant+dequant round-trip, POWER-OF-2 (ue8m0) scale.
    Faithful to model.py act_quant(scale_fmt='ue8m0', inplace=True)."""
    orig_shape = x.shape
    K = orig_shape[-1]
    xb = x.reshape(-1, K).to(torch.float32)
    M = xb.size(0)
    if K % block != 0:
        # pad to a multiple of block (should not happen for our 448/64/128 cases)
        pad = block - (K % block)
        xb = F.pad(xb, (0, pad)); K = xb.size(-1)
    nblk = K // block
    xblk = xb.reshape(M, nblk, block)
    amax = xblk.abs().amax(dim=-1).clamp(min=1e-4)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / FP8_MAX))).clamp(min=2.0 ** -126)
    q = (xblk / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dq = q.to(torch.float32) * s.unsqueeze(-1)
    return dq.reshape(orig_shape)

def fp8_linear(x, weight_fp8, scale_e8m0):
    x_q = act_quant_rt(x, FP8_BLOCK)
    W = fp8_dequant(weight_fp8, scale_e8m0)
    return x_q @ W.T

def fp4_linear(x, weight_int8, scale_e8m0):
    x_q = act_quant_rt(x, FP8_BLOCK)
    W = fp4_dequant(weight_int8, scale_e8m0)
    return x_q @ W.T

def rmsnorm(x, weight):
    var = x.square().mean(-1, keepdim=True)
    return (weight.to(torch.float32) * (x.to(torch.float32) * torch.rsqrt(var + NORM_EPS)))

# ---------- hc_split_sinkhorn / hc_pre / hc_post (from hf_bos_ref.py, verbatim) ----------
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=HC_MULT, iters=HC_SINKHORN_ITERS, eps=HC_EPS):
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])
    comb = mixes[..., 2*hc:].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[2*hc:].reshape(hc, hc)
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb

def hc_pre(x, hc_fn, hc_scale, hc_base):
    shape = x.shape
    xf = x.flatten(2).to(torch.float32)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y, post, comb

def hc_post(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y

# ---------- sparse attention (from hf_bos_ref.py, verbatim) ----------
def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    b, s, h, d = q.shape
    qf = q.to(torch.float32); kvf = kv.to(torch.float32); sinkf = attn_sink.to(torch.float32)
    out = torch.zeros(b, s, h, d, dtype=torch.float32)
    for bi in range(b):
        for si in range(s):
            idxs = topk_idxs[bi, si]
            valid = idxs[idxs >= 0]
            if valid.numel() == 0:
                continue
            kk = kvf[bi][valid]
            scores = torch.einsum("hd,td->ht", qf[bi, si], kk) * scale
            scores_max = scores.amax(dim=-1, keepdim=True)
            exp_s = torch.exp(scores - scores_max)
            sum_exp = exp_s.sum(dim=-1, keepdim=True) + torch.exp(sinkf.unsqueeze(-1) - scores_max)
            acc_o = torch.einsum("ht,td->hd", exp_s, kk)
            out[bi, si] = acc_o / sum_exp
    return out

# ---------- RoPE (ported from model.py) ----------
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)
    def linear_ramp_factor(mn, mx, dim):
        if mn == mx: mx += 0.001
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn), 0, 1)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)   # [seqlen, dim//2] complex

def apply_rotary_emb(x, freqs_cis, inverse=False):
    """x: [..., rope_dim(64)]. freqs_cis: [seqlen, 32] complex. Returns rotated [...,64]."""
    xc = torch.view_as_complex(x.float().contiguous().unflatten(-1, (-1, 2)))
    fc = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:
        fc = fc.view(1, xc.size(1), xc.size(-1))
    else:
        fc = fc.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * fc).flatten(-2)

# ---------- topk index helpers (ported verbatim from model.py) ----------
def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()

def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()

# ---------- KV Compressor (ported faithfully from model.py, rotate=False) ----------
class Compressor:
    def __init__(self, L, ratio):
        self.L = L; self.ratio = ratio; self.head_dim = HEAD_DIM
        self.rope_head_dim = ROPE_HEAD_DIM; self.nope_head_dim = NOPE_HEAD_DIM
        self.overlap = (ratio == 4)
        self.coff = 1 + self.overlap            # 2 for ratio==4
        p = f"layers.{L}.attn.compressor."
        self.ape = get_tensor(p+"ape").to(torch.float32)               # [ratio, coff*head_dim]
        self.wkv = get_tensor(p+"wkv.weight").to(torch.float32)        # [coff*head_dim, dim] (bf16->fp32)
        self.wgate = get_tensor(p+"wgate.weight").to(torch.float32)    # [coff*head_dim, dim]
        self.norm_w = get_tensor(p+"norm.weight").to(torch.float32)     # [head_dim]
        self.kv_cache = None     # [max_seq//ratio, head_dim] assigned by Attention
        self.freqs_cis = None    # layer freqs_cis
        self.kv_state = torch.zeros(1, self.coff*ratio, self.coff*self.head_dim, dtype=torch.float32)
        self.score_state = torch.full((1, self.coff*ratio, self.coff*self.head_dim), float("-inf"), dtype=torch.float32)

    def overlap_transform(self, tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.ratio, self.head_dim
        new = tensor.new_full((b, s, 2*ratio, d), value)
        new[:, :, ratio:] = tensor[:, :, :, d:]
        new[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new

    def _norm(self, x):
        var = x.square().mean(-1, keepdim=True)
        return self.norm_w * (x * torch.rsqrt(var + NORM_EPS))

    def forward(self, x, start_pos):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.ratio, self.overlap, self.head_dim, self.rope_head_dim
        x = x.float()
        kv = F.linear(x, self.wkv)        # [b,s,coff*head_dim]
        score = F.linear(x, self.wgate)  # [b,s,coff*head_dim]
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio:cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio:cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset:offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset:offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        kv = self._norm(kv)
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        kv = kv.contiguous().clone()
        rope_dim = kv[..., -rd:].contiguous()
        kv[..., -rd:] = apply_rotary_emb(rope_dim, freqs_cis)
        # rotate=False (attention's compressor): act_quant the non-rope dims, block 64
        kv[..., :-rd] = act_quant_rt(kv[..., :-rd], 64)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv

# ---------- per-layer attention state ----------
class LayerAttn:
    """Holds dequantized weights + KV cache (window ring + compressed) + compressor for one layer."""
    def __init__(self, L):
        self.L = L
        self.ratio = COMPRESS_RATIOS[L]
        self.win = WINDOW
        p = f"layers.{L}.attn."
        self.wq_a_w = get_tensor(p+"wq_a.weight"); self.wq_a_s = get_tensor(p+"wq_a.scale")
        self.wq_b_w = get_tensor(p+"wq_b.weight"); self.wq_b_s = get_tensor(p+"wq_b.scale")
        self.wkv_w  = get_tensor(p+"wkv.weight");  self.wkv_s  = get_tensor(p+"wkv.scale")
        self.wo_a_w = get_tensor(p+"wo_a.weight"); self.wo_a_s = get_tensor(p+"wo_a.scale")
        self.wo_b_w = get_tensor(p+"wo_b.weight"); self.wo_b_s = get_tensor(p+"wo_b.scale")
        self.q_norm_w = get_tensor(p+"q_norm.weight")
        self.kv_norm_w = get_tensor(p+"kv_norm.weight")
        self.attn_sink = get_tensor(p+"attn_sink")
        self.attn_norm_w = get_tensor(f"layers.{L}.attn_norm.weight")
        # Single 3D KV cache [1, win + max//ratio, 512] (matches model.py Attention.kv_cache).
        # window in [0:win], compressed in [win:win+max//ratio]. compressor.kv_cache = cache[:, win:].
        comp_slots = (MAX_SEQ // self.ratio) if self.ratio else 0
        self.kv_cache = torch.zeros(1, self.win + comp_slots, HEAD_DIM, dtype=torch.float32)
        if self.ratio:
            self.compressor = Compressor(L, self.ratio)
            self.compressor.kv_cache = self.kv_cache[:, self.win:]   # [1, max//ratio, 512]
        else:
            self.compressor = None
        # freqs_cis for this layer (compressed -> YaRN base=160000; L0/L1 -> base=10000 no YaRN)
        if self.ratio:
            orig, base = ORIG_SEQ_LEN, COMPRESS_ROPE_THETA
        else:
            orig, base = 0, ROPE_THETA
        self.freqs_cis = precompute_freqs_cis(ROPE_HEAD_DIM, MAX_SEQ, orig, base, ROPE_FACTOR, BETA_FAST, BETA_SLOW)
        if self.compressor is not None:
            self.compressor.freqs_cis = self.freqs_cis

    def forward(self, x, start_pos):
        """x: [1, s, 4096] fp32 (attn-normed). Returns [1, s, 4096] fp32."""
        bsz, seqlen, _ = x.size()
        rd = ROPE_HEAD_DIM
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.ratio
        # q
        qr = rmsnorm(fp8_linear(x, self.wq_a_w, self.wq_a_s), self.q_norm_w)   # [1,s,1024]
        q = fp8_linear(qr, self.wq_b_w, self.wq_b_s).view(bsz, seqlen, N_HEADS, HEAD_DIM)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + NORM_EPS)
        q = q.contiguous().clone()
        q[..., -rd:] = apply_rotary_emb(q[..., -rd:].contiguous(), freqs_cis)
        # kv
        kv = rmsnorm(fp8_linear(x, self.wkv_w, self.wkv_s), self.kv_norm_w)     # [1,s,512]
        kv = kv.contiguous().clone()
        if start_pos == 0 and self.L == 0 and os.environ.get("HF_TRACE_L0"):
            qr.detach().float().reshape(seqlen, -1).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_qc.f32")
            kv.detach().float().reshape(seqlen, -1).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_kvc.f32")
            print(f"[hf-trace] dumped qc[q,1024]->/tmp/hf_L00_qc.f32 kvc[s,512]->/tmp/hf_L00_kvc.f32 (post-norm, pre-rope)", flush=True)
        kv[..., -rd:] = apply_rotary_emb(kv[..., -rd:].contiguous(), freqs_cis)
        if start_pos == 0 and self.L == 0 and os.environ.get("HF_TRACE_L0"):
            q.detach().float().reshape(seqlen*N_HEADS, HEAD_DIM).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_q.f32")
            kv.detach().float().reshape(seqlen, -1).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_kv.f32")
            print(f"[hf-trace] dumped q[s*64,512]->/tmp/hf_L00_q.f32 kv[s,512]->/tmp/hf_L00_kv.f32 (pre-act_quant)", flush=True)
        kv[..., :-rd] = act_quant_rt(kv[..., :-rd], 64)
        if start_pos == 0 and self.L == 0 and os.environ.get("HF_TRACE_L0"):
            kv.detach().float().reshape(seqlen, -1).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_kv_aq.f32")
            print(f"[hf-trace] dumped kv post-act_quant -> /tmp/hf_L00_kv_aq.f32", flush=True)
        # topk
        topk_idxs = get_window_topk_idxs(self.win, bsz, seqlen, start_pos)     # [1,s,win_or_s]
        if ratio and USE_COMPRESSOR:
            offset = kv.size(1) if start_pos == 0 else self.win
            compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        if start_pos == 0:
            # store window kv into cache [0:seqlen]; sparse_attn uses LOCAL kv (+compress)
            if seqlen <= self.win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % self.win
                self.kv_cache[:bsz, cutoff:self.win], self.kv_cache[:bsz, :cutoff] = kv[:, -self.win:].split([self.win-cutoff, cutoff], dim=1)
            if ratio and USE_COMPRESSOR:
                kv_compress = self.compressor.forward(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, SOFTMAX_SCALE)
        else:
            # store window kv into ring slot; compressor writes cache[:, win + start_pos//ratio]
            self.kv_cache[:bsz, start_pos % self.win] = kv.squeeze(1)
            if ratio and USE_COMPRESSOR:
                self.compressor.forward(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, SOFTMAX_SCALE)
        # inverse rope on output
        o_post_attn = o.contiguous().clone()                                   # [1,s,512] after sparse_attn
        o = o_post_attn.clone()
        o[..., -rd:] = apply_rotary_emb(o[..., -rd:].contiguous(), freqs_cis, inverse=True)
        o_post_invrope = o                                                      # [1,s,512]
        if start_pos == 0 and self.L == 0 and os.environ.get("HF_TRACE_L0"):
            o_post_attn.detach().float().reshape(seqlen*N_HEADS, HEAD_DIM).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_svpre.f32")
            o_post_invrope.detach().float().reshape(seqlen*N_HEADS, HEAD_DIM).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_svpost.f32")
            print(f"[hf-trace] dumped svpre/svpost [s*64,512] -> /tmp/hf_L00_svpre.f32 /tmp/hf_L00_svpost.f32", flush=True)
        # o projection
        o = o_post_invrope.view(bsz, seqlen, O_GROUPS, -1)                      # [1,s,8,4096]
        wo_a = fp8_dequant(self.wo_a_w, self.wo_a_s).view(O_GROUPS, O_LORA, -1)  # [8,1024,4096] block-scaled (matches fst_converter/convert.py)
        o_g = torch.einsum("bsgd,grd->bsgr", o, wo_a)                          # [1,s,8,1024]
        if start_pos == 0 and self.L == 0 and os.environ.get("HF_TRACE_L0"):
            o_g.detach().float().reshape(seqlen, -1).cpu().numpy().astype("float32").tofile("/tmp/hf_L00_low.f32")
            print(f"[hf-trace] dumped o_g(post wo_a) [s,8192]->/tmp/hf_L00_low.f32", flush=True)
        o_flat = o_g.flatten(2)                                                  # [1,s,8192]
        x_out = fp8_linear(o_flat, self.wo_b_w, self.wo_b_s)                   # [1,s,4096]
        if start_pos == 0 and self.L == 0 and os.environ.get("HF_TRACE_L0"):
            def _mx(t, n):
                tt = t.detach().float().reshape(t.shape[1], -1)   # position dim = shape[1]
                m = tt.abs().max(dim=-1).values
                print(f"[attn-trace] {n:14s}: " + " ".join(f"{m[i].item():.3f}" for i in range(min(len(m),6))), flush=True)
            _mx(q, "q"); _mx(kv, "kv")
            _mx(o_post_attn, "o_sparse_attn"); _mx(o_post_invrope, "o_invrope")
            _mx(o_g, "o_wo_a"); _mx(o_flat, "o_wo_a_flat"); _mx(x_out, "x_out_wo_b")
        return x_out

# ---------- MoE FFN (generalized to M>1, group-by-expert like model.py) ----------
def _expert_linear(x, w, s):
    if w.dtype == torch.int8:
        return fp4_linear(x, w, s)
    return fp8_linear(x, w, s)

def expert_forward(x, w1_w, w1_s, w2_w, w2_s, w3_w, w3_s, weight=None):
    gate = _expert_linear(x, w1_w, w1_s)
    up = _expert_linear(x, w3_w, w3_s)
    if SWIGLU_LIMIT > 0:
        up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    h = F.silu(gate) * up
    if weight is not None:
        h = weight * h
    return _expert_linear(h, w2_w, w2_s)

def _load_expert(L, idx):
    ep = f"layers.{L}.ffn.experts.{idx}."
    return (get_tensor(ep+"w1.weight"), get_tensor(ep+"w1.scale"),
            get_tensor(ep+"w2.weight"), get_tensor(ep+"w2.scale"),
            get_tensor(ep+"w3.weight"), get_tensor(ep+"w3.scale"))

def ffn_forward(x, L, input_ids):
    """x: [1,s,4096] fp32. input_ids: [1,s] long. Returns [1,s,4096] fp32."""
    bsz, s, d = x.shape
    p = f"layers.{L}.ffn."
    gate_w = get_tensor(p+"gate.weight").to(torch.float32)     # [256,4096]
    xf = x.view(s, d).float()
    scores = (xf @ gate_w.T)                                    # [s,256]
    scores = F.softplus(scores).sqrt()                         # sqrtsoftplus
    original_scores = scores
    if L < N_HASH_LAYERS:
        tid2eid = get_tensor(p+"gate.tid2eid")                 # [vocab,6] int32
        indices = tid2eid[input_ids.flatten()].long()          # [s,6]
    else:
        bias = get_tensor(p+"gate.bias").to(torch.float32)      # [256]
        indices = (scores + bias).topk(N_ACTIVATED, dim=-1)[1].long()  # [s,6]
    weights = original_scores.gather(1, indices)                # [s,6]
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE
    # group tokens by expert (matches model.py MoE.forward)
    y = torch.zeros(s, d, dtype=torch.float32)
    unique_experts = indices.unique().tolist()
    expert_cache = {e: _load_expert(L, e) for e in unique_experts}
    for i in unique_experts:
        pos, slot = (indices == i).nonzero(as_tuple=True)
        if pos.numel() == 0:
            continue
        w = weights[pos, slot].unsqueeze(-1)                    # [n,1]
        out = expert_forward(xf[pos], *expert_cache[i])         # [n,d]
        y[pos] += w * out
    # shared expert (weight 1.0, no route_scale)
    sp = p + "shared_experts."
    sw = (get_tensor(sp+"w1.weight"), get_tensor(sp+"w1.scale"),
          get_tensor(sp+"w2.weight"), get_tensor(sp+"w2.scale"),
          get_tensor(sp+"w3.weight"), get_tensor(sp+"w3.scale"))
    shared_out = expert_forward(xf, *sw)
    if L == 0 and bool(os.environ.get("HF_LAYER_DUMP")) and s >= 14:
        routed_mx = y.abs().reshape(s, -1).max(dim=-1).values
        shared_mx = shared_out.abs().reshape(s, -1).max(dim=-1).values
        tot_mx = (y + shared_out).abs().reshape(s, -1).max(dim=-1).values
        print("[hf-split] L0: per-pos |routed|mx=" + " ".join(f"{routed_mx[i].item():.3f}" for i in range(s)), flush=True)
        print("[hf-split] L0: per-pos |shared|mx=" + " ".join(f"{shared_mx[i].item():.3f}" for i in range(s)), flush=True)
        print("[hf-split] L0: per-pos |ffn|mx="    + " ".join(f"{tot_mx[i].item():.3f}" for i in range(s)), flush=True)
        print("[hf-split] L0: indices[13]=" + " ".join(str(int(t)) for t in indices[13].tolist())
              + " weights[13]=" + " ".join(f"{t:.3f}" for t in weights[13].tolist()), flush=True)
        _np.save(f"/tmp/hf_L00_ffn.npy", (y + shared_out)[-1].detach().cpu().numpy().astype("float32"))
        _np.save(f"/tmp/hf_L00_routed.npy", y[-1].detach().cpu().numpy().astype("float32"))
        _np.save(f"/tmp/hf_L00_shared.npy", shared_out[-1].detach().cpu().numpy().astype("float32"))
        # per-expert down_out for m13 (unweighted), to compare with engine /tmp/eng_L0_e*_m13.f32
        def _expert_linear_noaq(x, w, s):
            if w.dtype == torch.int8:
                return fp4_dequant(w, s).to(torch.float32)
            return fp8_dequant(w, s).to(torch.float32)
        def expert_forward_raw(x, w1w, w1s, w2w, w2s, w3w, w3s):
            Wg = _expert_linear_noaq(x, w1w, w1s); Wu = _expert_linear_noaq(x, w3w, w3s)
            gate = x.to(torch.float32) @ Wg.T; up = x.to(torch.float32) @ Wu.T
            if SWIGLU_LIMIT > 0:
                up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
                gate = torch.clamp(gate, max=SWIGLU_LIMIT)
            h = F.silu(gate) * up
            Wd = _expert_linear_noaq(h, w2w, w2s)
            return (h.to(torch.float32) @ Wd.T)
        for i in indices[13].tolist():
            out_i = expert_forward(xf[13:14], *expert_cache[i])  # [1,d]  WITH act_quant
            _np.save(f"/tmp/hf_L0_e{i}_m13.npy", out_i[0].detach().cpu().numpy().astype("float32"))
            out_raw = expert_forward_raw(xf[13:14], *expert_cache[i])  # WITHOUT act_quant
            _np.save(f"/tmp/hfnoaq_L0_e{i}_m13.npy", out_raw[0].detach().cpu().numpy().astype("float32"))
        # dump HF fp32 FFN input (xf = x.view().float() AFTER moe_norm) for m13
        xf[13].detach().cpu().numpy().astype("float32").tofile("/tmp/hf_L00_xf13.f32")
        print(f"[hf-split] L0: dumped xf[13] -> /tmp/hf_L00_xf13.f32 |mx={xf[13].abs().max().item():.4f}", flush=True)
    y = y + shared_out
    return y.view(bsz, s, d)

# ---------- one transformer block (HC + attn + HC + ffn) ----------
def block_forward(h, layer_attn, L, start_pos, input_ids, ffn_norm_w,
                  hc_attn_fn, hc_attn_scale, hc_attn_base,
                  hc_ffn_fn, hc_ffn_scale, hc_ffn_base):
    residual = h
    x, post, comb = hc_pre(residual, hc_attn_fn.to(torch.float32), hc_attn_scale.to(torch.float32), hc_attn_base.to(torch.float32))
    x_attn_norm = rmsnorm(x, layer_attn.attn_norm_w)
    attn_out = layer_attn.forward(x_attn_norm, start_pos)
    if L == 0 and start_pos == 0 and bool(os.environ.get("HF_LAYER_DUMP")):
        attn_out[0, -1].detach().cpu().numpy().astype("float32").tofile("/tmp/hf_L00_attnout.f32")
        x_attn_norm[0, -1].detach().cpu().numpy().astype("float32").tofile("/tmp/hf_L00_attnnorm.f32")
        print(f"[hf-split] L0: dumped attn_out->/tmp/hf_L00_attnout.f32 |mx={attn_out[0,-1].abs().max().item():.4f}"
              f" attn_norm_in->/tmp/hf_L00_attnnorm.f32 |mx={x_attn_norm[0,-1].abs().max().item():.4f}", flush=True)
    h = hc_post(attn_out, residual, post, comb)
    if start_pos == 0 and bool(os.environ.get("HF_LAYER_DUMP")):
        _np.save(f"/tmp/hf_L{L:02d}_attn.npy", h[0, -1].detach().cpu().numpy().astype("float32"))
    residual2 = h
    x, post, comb = hc_pre(residual2, hc_ffn_fn.to(torch.float32), hc_ffn_scale.to(torch.float32), hc_ffn_base.to(torch.float32))
    x_ffn_norm = rmsnorm(x, ffn_norm_w)
    if L == 0 and start_pos == 0 and bool(os.environ.get("HF_LAYER_DUMP")):
        # dump pre-rmsnorm FFN input (x = hc_ffn_pre output) for m13, to compare
        # with engine /tmp/eng_L00_cur.f32 and isolate attn vs hc_pre vs moe_norm.
        x[0, -1].detach().cpu().numpy().astype("float32").tofile("/tmp/hf_L00_cur.f32")
        residual2[0, -1].flatten().detach().cpu().numpy().astype("float32").tofile("/tmp/hf_L00_resmid.f32")
        print(f"[hf-split] L0: dumped cur(pre-norm)->/tmp/hf_L00_cur.f32 |mx={x[0,-1].abs().max().item():.4f}"
              f" resmid->/tmp/hf_L00_resmid.f32 |mx={residual2[0,-1].abs().max().item():.4f}", flush=True)
    ffn_out = ffn_forward(x_ffn_norm, L, input_ids)
    h = hc_post(ffn_out, residual2, post, comb)
    if L == 0 and start_pos == 0 and os.environ.get("HF_TRACE_L0"):
        def mx(t, name):
            tt = t.detach().float()
            tt = tt.reshape(tt.shape[1], -1) if tt.dim() >= 3 else tt.reshape(tt.shape[0], -1)
            m = tt.abs().max(dim=-1).values
            print(f"[L0-trace] {name:14s}: per-pos mx=" + " ".join(f"{m[i].item():.3f}" for i in range(min(len(m),14))), flush=True)
        mx(residual,   "res_in");        mx(x_attn_norm, "attn_norm_in")
        mx(attn_out,   "attn_out");      mx(h,           "L0_out")
        mx(residual2,  "res_mid");       mx(x_ffn_norm,  "ffn_norm_in")
        mx(ffn_out,    "ffn_out")
    return h   # [1,s,4,4096] fp32

# ---------- head: hc_head -> norm -> lm_head (matches model.py ParallelHead: x[:,-1]) ----------
def head_logits(h):
    # h: [1,s,4,4096]
    hc_head_fn = get_tensor("hc_head_fn").to(torch.float32)
    hc_head_base = get_tensor("hc_head_base").to(torch.float32)
    hc_head_scale = get_tensor("hc_head_scale").to(torch.float32)
    norm_w = get_tensor("norm.weight")
    head_w = get_tensor("head.weight").to(torch.float32)        # [vocab,4096]
    xf = h.flatten(2).to(torch.float32)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(xf, hc_head_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_head_scale + hc_head_base) + HC_EPS
    hh = torch.sum(pre.unsqueeze(-1) * h, dim=2)                # [1,s,4096]  (pre-norm)
    hh_pre = hh.clone()
    hh = rmsnorm(hh, norm_w)                                    # [1,s,4096]  (post-norm)
    last = hh[:, -1]                                            # [1,4096]
    logits = last @ head_w.T                                   # [1,vocab]
    # Dump prefill (first call) pre-norm / post-norm hidden + full logits for cosine compare
    if not getattr(head_logits, "_dumped", False):
        head_logits._dumped = True
        import numpy as _np
        _np.save("/tmp/hf_prefill_plain.npy", hh_pre[0, -1].detach().cpu().numpy().astype("float32"))  # last pos pre-norm
        _np.save("/tmp/hf_prefill_hs.npy",     last.detach().cpu().numpy().astype("float32"))           # last pos post-norm
        _np.save("/tmp/hf_prefill_logits.npy", logits[0].detach().cpu().numpy().astype("float32"))      # full logits
        print(f"[hf_gen_ref] dumped prefill: plain[D={hh_pre.shape[-1]}] + hs + logits[V={logits.shape[-1]}] -> /tmp/hf_prefill_*.npy", flush=True)
    return logits

# ---------- sampling ----------
def sample_token(logits, temperature, rng):
    if temperature == 0:
        return int(logits.argmax(dim=-1).item())
    lg = logits / max(temperature, 1e-5)
    probs = torch.softmax(lg, dim=-1, dtype=torch.float32)
    # Gumbel-max (matches model.py sample): argmax(probs / Exp(1)) == argmax(log probs + Gumbel)
    e = torch.empty_like(probs).exponential_(1.0, generator=rng)
    return int((probs / e).argmax(dim=-1).item())

def top5(logits):
    vals, ids = torch.topk(logits.flatten(), 5)
    return [(int(ids[i].item()), float(vals[i].item())) for i in range(5)]

# ---------- tokenizer (tokenizers lib, same bridge as engine) ----------
def tokenize(prompt):
    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    chat = "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜>" + "</think>"
    ids = tok.encode(chat).ids
    return ids

def detokenize(ids):
    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    return tok.decode(ids)

# ---------- main generate loop ----------
def main():
    print(f"[hf_gen_ref] prompt={PROMPT!r} n_gen={N_GEN} temp={TEMPERATURE} "
          f"compressor={USE_COMPRESSOR} greedy={GREEDY}", flush=True)
    input_ids = tokenize(PROMPT)
    print(f"[hf_gen_ref] prefill tokens ({len(input_ids)}): {input_ids}", flush=True)
    S = len(input_ids)

    # build per-layer attn state
    print(f"[hf_gen_ref] building per-layer attention state for {N_LAYERS} layers...", flush=True)
    layer_attns = [LayerAttn(L) for L in range(N_LAYERS)]
    # preload all hc params + norms (small)
    hc_params = []
    for L in range(N_LAYERS):
        hc_params.append((
            get_tensor(f"layers.{L}.hc_attn_fn"), get_tensor(f"layers.{L}.hc_attn_scale"), get_tensor(f"layers.{L}.hc_attn_base"),
            get_tensor(f"layers.{L}.hc_ffn_fn"), get_tensor(f"layers.{L}.hc_ffn_scale"), get_tensor(f"layers.{L}.hc_ffn_base"),
            get_tensor(f"layers.{L}.ffn_norm.weight"),
        ))
    embed = get_tensor("embed.weight").to(torch.float32)     # [vocab,4096]

    rng = torch.Generator().manual_seed(SAMPLE_SEED)

    def run_model(token_ids, start_pos):
        ids_t = torch.tensor([token_ids], dtype=torch.long)   # [1,s]
        h = embed[ids_t].unsqueeze(2).repeat(1,1,HC_MULT,1).to(torch.float32)  # [1,s,4,4096]
        dump_layers = (start_pos == 0 and bool(os.environ.get("HF_LAYER_DUMP")))
        if dump_layers:
            # baseline: embedding (last prefill pos m13) before L0
            _np.save(f"/tmp/hf_L00_in.npy", h[0, -1].detach().cpu().numpy().astype("float32"))
        for L in range(N_LAYERS):
            (afn,asc,abas,ffn,fsc,fbas,fnorm) = hc_params[L]
            h = block_forward(h, layer_attns[L], L, start_pos, ids_t, fnorm,
                              afn,asc,abas, ffn,fsc,fbas)
            if dump_layers:
                # 4-stream residual [4,4096] of last prefill pos AFTER layer L
                _np.save(f"/tmp/hf_L{L:02d}_h.npy", h[0, -1].detach().cpu().numpy().astype("float32"))
            if start_pos == 0 and L in (0, 13, 26, N_LAYERS - 1):
                # per-position residual magnitude at L42 OUT (prefill)
                norms = h[0].float().reshape(h.shape[1], -1).norm(dim=-1).tolist()
                mxs   = h[0].float().reshape(h.shape[1], -1).abs().max(dim=-1).values.tolist()
                print(f"[hf_gen_ref] L42 OUT per-pos |residual|: " +
                      " ".join(f"m{i}={norms[i]:.1f}(mx{mxs[i]:.0f})" for i in range(h.shape[1])), flush=True)
        return head_logits(h)   # [1,vocab] for LAST position

    generated = []
    # PREFILL (start_pos=0, all S tokens)
    logits = run_model(input_ids, 0)
    t5 = top5(logits[0])
    prefill_top5 = t5
    print(f"[hf_gen_ref] PREFILL last-pos (pos {S-1}) top5 raw logits: " +
          " ".join(f"{tid}({v:.5f})" for tid,v in t5), flush=True)
    if GREEDY:
        nxt = int(logits.argmax(dim=-1).item())
    else:
        nxt = sample_token(logits, TEMPERATURE, rng)
    generated.append(nxt)
    print(f"[hf_gen_ref] -> token[0] = {nxt}  ({detokenize([nxt])!r})", flush=True)

    # DECODE (M=1)
    for step in range(1, N_GEN):
        sp = S - 1 + step      # position of the just-sampled token
        logits = run_model([nxt], sp)
        t5 = top5(logits[0])
        label = "FIRST-DECODE" if step == 1 else f"decode{step}"
        print(f"[hf_gen_ref] {label} (pos {sp}) top5 raw logits: " +
              " ".join(f"{tid}({v:.5f})" for tid,v in t5), flush=True)
        if GREEDY:
            nxt = int(logits.argmax(dim=-1).item())
        else:
            nxt = sample_token(logits, TEMPERATURE, rng)
        generated.append(nxt)
        print(f"[hf_gen_ref] -> token[{step}] = {nxt}  ({detokenize([nxt])!r})", flush=True)

    text = detokenize(generated)
    print("\n==== HF GENERATION RESULT ====", flush=True)
    print(f"prompt    : {PROMPT}", flush=True)
    print(f"mode      : {'greedy' if GREEDY else f'sampled(temp={TEMPERATURE},seed={SAMPLE_SEED})'}  compressor={USE_COMPRESSOR}", flush=True)
    print(f"tokens    : {generated}", flush=True)
    print(f"generated : {text!r}", flush=True)

    result = {
        "prompt": PROMPT, "n_gen": N_GEN, "temperature": TEMPERATURE,
        "compressor": USE_COMPRESSOR, "greedy": GREEDY,
        "prefill_tokens": input_ids, "generated_tokens": generated, "generated_text": text,
        "prefill_last_top5": prefill_top5,
        "script": "hf_gen_ref.py",
    }
    with open("/home/raffaele/Progetti/FaStar/hf_gen_ref_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("result written to hf_gen_ref_result.json", flush=True)

if __name__ == "__main__":
    main()