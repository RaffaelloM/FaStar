#!/usr/bin/env python3
"""FaStar (FST) model converter.

Produces a .fst file: a zero-copy, page-aligned container for hybrid
MoE inference (iGPU attention/router + NPU expert dequant).

Layout of a .fst file
---------------------
    [ page 0 ]      4096-byte main header (FSTHeader)
    [ pages 1..K ]  shared-bank directory (SharedTensorEntry[]) then
                    Q8_0 / BF16 dense tensors (attention, router, norms),
                    every tensor start 4KB-aligned
    [ expert bank ] num_layers * num_experts fixed 16,250,880-byte expert
                    blocks, each block start 4KB-aligned (stride 16,252,928)

Expert block layout (dense DS4 format, validated on XDNA2 NPU)
-------------------------------------------------------------
    Per expert: 3 projections × (out × in/32 × 17) bytes = 13,369,344 bytes.
    Layout: gate [4096×128×17], up [4096×128×17], down [2048×64×17].
    Each 17-byte block: [1 e8m0 scale, 16 packed FP4 nibbles] for 32 elements.
    No padding. No zero-bias. No tile grid. Pure row-major interleaved blocks.
"""

import struct
import math
import sys
import json
import time
import gc
from pathlib import Path

import numpy as np

try:
    import tqdm as _tqdm_mod
    def _tqdm(it, desc=""):
        return _tqdm_mod.tqdm(it, desc=desc, ncols=100)
except ImportError:
    def _tqdm(it, desc=""):
        return it

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


PAGE = 4096

FST_MAGIC = b"FST\x00"
FST_VERSION = 2

MXFP4_CHUNK_BYTES = 2560
MXFP4_TILE_ROWS = 32
MXFP4_TILE_COLS = 128
MXFP4_NIBBLES = MXFP4_TILE_ROWS * MXFP4_TILE_COLS          # 4096
MXFP4_SCALE_BYTES = 128
MXFP4_BIAS_BYTES = 64
MXFP4_PAD_BYTES = 320
MXFP4_NIBBLE_BYTES = 2048

# DeepSeek V4 Flash expert geometry (hidden=4096, intermediate=2048).
# Dense block format: 32 FP4 elements per block = 1 scale byte + 16 nibble bytes = 17 bytes.
# w1 (gate) / w3 (up): [out=2048, in=4096], blocks_per_row = 4096/32 = 128
# w2 (down):           [out=4096, in=2048], blocks_per_row = 2048/32 = 64
# Stored sequentially: gate, up, down. No tiling, no padding.
# Native MXFP4 dense block: 32 FP4 elements = 1 E8M0 scale byte + 16 nibble bytes = 17 bytes.
# Emitted as B[N, K] row-major (out-contiguous) — matches the transposed-B expert GEMM.
DENSE_BLOCK_ELEMS = 32
DENSE_BLOCK_BYTES = 17

# Q4_0 format: 32 elements per block, 18 bytes (fp16 scale + 16 nibble bytes)
Q4_0_BLOCK_ELEMS = 32
Q4_0_BLOCK_BYTES = 18

# Expert projection sizes (after FP4 dequant, before transpose):
# w1 (gate): packed [out=2048, in/2=2048] → dequant [out=2048, in=4096]
# w3 (up):   same
# w2 (down): packed [out=4096, in/2=1024] → dequant [out=4096, in=2048]
EXPERT_GATE_OUT = 2048
EXPERT_GATE_IN = 4096
EXPERT_DOWN_OUT = 4096
EXPERT_DOWN_IN = 2048

# Native MXFP4 dense block sizes per projection (32 elements per block, 17 bytes)
MXFP4_PROJ_BYTES_GATE = (EXPERT_GATE_OUT * EXPERT_GATE_IN // DENSE_BLOCK_ELEMS) * DENSE_BLOCK_BYTES
MXFP4_PROJ_BYTES_UP   = (EXPERT_GATE_OUT * EXPERT_GATE_IN // DENSE_BLOCK_ELEMS) * DENSE_BLOCK_BYTES
MXFP4_PROJ_BYTES_DOWN = (EXPERT_DOWN_OUT * EXPERT_DOWN_IN // DENSE_BLOCK_ELEMS) * DENSE_BLOCK_BYTES
EXPERT_BLOCK_BYTES = MXFP4_PROJ_BYTES_GATE + MXFP4_PROJ_BYTES_UP + MXFP4_PROJ_BYTES_DOWN  # 13,369,344
EXPERT_BLOCK_STRIDE = ((EXPERT_BLOCK_BYTES + PAGE - 1) // PAGE) * PAGE                  # 4096-aligned

# NPU GEMM kernel K dimensions (matches dequant output after transpose, no padding):
#   gate/up: [in=4096, out=2048] → B[K=4096, N=2048] (K matches exactly)
#   down:    [in=2048, out=4096] → B[K=2048, N=4096] (K matches exactly)
GATE_K_PAD = 4096
DOWN_K_PAD = 2048
# Dequant output size per projection (padded, BF16):
DEQUANT_GATE_BYTES = GATE_K_PAD * EXPERT_GATE_OUT * 2   # 4096*2048*2 = 16 MB
DEQUANT_DOWN_BYTES = DOWN_K_PAD * EXPERT_DOWN_OUT * 2   # 2048*4096*2 = 16 MB

# Legacy MXFP4 tile grid (unused for Q4_K, kept for reference)
V4_HIDDEN = 4096
V4_INTERMEDIATE = 2048
V4_W2_ROW_BLOCKS = V4_INTERMEDIATE // MXFP4_TILE_ROWS  # 2048/32 = 64
V4_W2_COL_BLOCKS = V4_HIDDEN // MXFP4_TILE_COLS        # 4096/128 = 32

Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 2 + Q8_0_BLOCK          # fp16 scale + 32 int8

QTYPE_Q8_0 = 1
QTYPE_BF16 = 2
QTYPE_F32 = 3

TID_EMBED = 0
TID_OUTPUT_NORM = 1
TID_LM_HEAD = 2
TID_INPUT_NORM = 3
TID_POST_ATTN_NORM = 4
TID_Q_PROJ = 5
TID_K_PROJ = 6
TID_V_PROJ = 7
TID_O_PROJ = 8
TID_ROUTER = 9
TID_ROUTER_BIAS = 10
TID_SHARED_GATE = 11
TID_SHARED_UP = 12
TID_SHARED_DOWN = 13
TID_Q_NORM = 14
TID_KV_NORM = 15
TID_MARKOV_W1 = 16       # [vocab, markov_rank] BF16 — Markov embedding lookup
TID_MARKOV_W2 = 17       # [vocab, markov_rank] BF16 — Markov head projection
TID_CONFIDENCE_PROJ = 18 # [1, dim+markov_rank] BF16 — confidence score layer
TID_MAIN_PROJ = 19       # [dim, 3*dim] FP8 — projects main hidden to draft input
TID_MAIN_NORM = 20        # [dim] BF16 — RMSNorm for main projection
TID_DRAFT_NORM = 21       # [dim] BF16 — final norm before head (mtp last stage)
TID_ATTN_SINK = 22        # [n_heads] F32 — attention sink parameter
# ── Hybrid Connection (HC) 4-stream residual — per-layer + global ──────────
# hc_*_fn   : [hc_dim=4*hidden, hc_mix_dim=24] BF16  (matvec control matrix)
# hc_*_scale: [3] F32 (pre/post/comb scales)   hc_*_base: [24] F32
# hc_head_* : global final HC collapse -> plain embedding before output norm
TID_HC_ATTN_FN = 23       # [4*hidden, 24] BF16 — attn HC fn (per layer)
TID_HC_ATTN_SCALE = 24    # [3] F32
TID_HC_ATTN_BASE = 25     # [24] F32
TID_HC_FFN_FN = 26        # [4*hidden, 24] BF16 — ffn HC fn (per layer)
TID_HC_FFN_SCALE = 27     # [3] F32
TID_HC_FFN_BASE = 28      # [24] F32
TID_HC_HEAD_FN = 29       # [4*hidden, 4] BF16 — global output HC collapse fn
TID_HC_HEAD_SCALE = 30    # [1] F32
TID_HC_HEAD_BASE = 31     # [4] F32
# ── V4 attention KV compressor (per-layer, ratio != 0) ─────────────────────
# DeepSeek V4 Flash compresses KV windows: ratio=4 (even L2..L40, two-lane
# comp_width=1024) and ratio=128 (odd L3..L39, single-lane comp_width=512).
# HF keys: layers.N.attn.compressor.{ape,wkv,wgate,norm}.  ape is F32
# [ratio, comp_width] (no ".weight"); wkv/wgate BF16 [comp_width, 4096]
# (bcol-native [N,K], no transpose); norm BF16 [512].
TID_ATTN_COMPRESSOR_APE  = 32   # [ratio, comp_width] F32
TID_ATTN_COMPRESSOR_WKV  = 33   # [comp_width, 4096] BF16
TID_ATTN_COMPRESSOR_GATE = 34   # [comp_width, 4096] BF16
TID_ATTN_COMPRESSOR_NORM = 35   # [512] BF16
# ── Global config arrays ────────────────────────────────────────────────
TID_COMPRESS_RATIOS = 40       # [n_layers] F32 — per-layer compression ratio

GLOBAL_LAYER = 0xFFFF

FP4_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32,
)

HEADER_FMT = "<4sIIIIIIIIIIIQffQQQQQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SHARED_ENTRY_FMT = "<IHHHHIQQQQQQ"
SHARED_ENTRY_SIZE = struct.calcsize(SHARED_ENTRY_FMT)


def align_up(x, a=PAGE):
    return (x + a - 1) & ~(a - 1)


def _to_f32_np(t):
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        return t.float().cpu().numpy()
    return np.ascontiguousarray(t, dtype=np.float32)


def f32_to_e8m0(x):
    """Vectorized e8m0 encoding.  Accepts scalar or np.ndarray."""
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
        if x <= 0.0 or math.isnan(x) or math.isinf(x):
            return 0
        e = int(math.ceil(math.log2(x)))
        e = max(-127, min(127, e))
        return (e + 127) & 0xFF
    # Vectorized path for np.ndarray
    x = np.asarray(x, dtype=np.float32)
    result = np.zeros(x.shape, dtype=np.uint8)
    mask = (x > 0) & np.isfinite(x)
    safe = np.where(mask, x, 1.0)
    e = np.ceil(np.log2(safe)).astype(np.int32)
    e = np.clip(e, -127, 127)
    result[mask] = ((e[mask] + 127) & 0xFF).astype(np.uint8)
    return result


def e8m0_to_f32(e):
    if e == 0:
        return 2.0 ** (-123)
    return 2.0 ** (e - 127)


def f32_to_bf16_bits(x):
    bits = np.array([float(x)], dtype=np.float32).view(np.uint32)[0]
    return int((bits >> 16) & 0xFFFF)


def quantize_mxfp4_tile(tensor):
    """Quantize a 32x128 BF16/F32 tile into one 2560-byte MXFP4 chunk.

    Per-column e8m0 scale, per-row BF16 bias (zero here; the FFN per-row
    bias is folded in by the caller via pack_expert_interleaved when the
    model ships expert biases). Nibble = nearest signed FP4 level to
    value/scale, packed low nibble first. Matches the XDNA2 dequant.
    """
    arr = _to_f32_np(tensor).reshape(MXFP4_TILE_ROWS, MXFP4_TILE_COLS)
    col_abs_max = np.abs(arr).max(axis=0)                       # [128]
    scale_e8m0 = np.array([f32_to_e8m0(v / 6.0) for v in col_abs_max], dtype=np.uint8)
    scale_f = np.array([e8m0_to_f32(int(s)) for s in scale_e8m0], dtype=np.float32)
    scale_f[scale_f < 1e-30] = 1.0                               # avoid /0 for dead cols

    target = arr / scale_f                                        # [32,128], range ~[-6,6]
    abs_t = np.abs(target)
    idx_pos = np.argmin(np.abs(FP4_LEVELS[None, :] - abs_t[:, :, None]), axis=2)  # [32,128] in 0..7
    nibbles = np.where(target >= 0.0, idx_pos, idx_pos + 8).astype(np.uint8)

    # byte[row, k] = nibble[row, 2k] (low) | nibble[row, 2k+1] (high), k = col//2
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).reshape(-1)

    bias_bf16 = np.zeros(MXFP4_TILE_ROWS, dtype=np.uint16)        # zero per-row bias

    chunk = bytearray(MXFP4_CHUNK_BYTES)
    chunk[0:MXFP4_SCALE_BYTES] = scale_e8m0.tobytes()
    chunk[MXFP4_SCALE_BYTES:MXFP4_SCALE_BYTES + MXFP4_BIAS_BYTES] = bias_bf16.tobytes()
    chunk[MXFP4_SCALE_BYTES + MXFP4_BIAS_BYTES:MXFP4_SCALE_BYTES + MXFP4_BIAS_BYTES + MXFP4_PAD_BYTES] = b"\x00" * MXFP4_PAD_BYTES
    chunk[512:512 + MXFP4_NIBBLE_BYTES] = packed.tobytes()
    return bytes(chunk)


def _tile_projection_in_out(in_out_np, row_blocks, col_blocks):
    """Tile a [in, out] float32 matrix into a [row_blocks, col_blocks, 2560] uint8 grid."""
    arr = np.ascontiguousarray(in_out_np, dtype=np.float32)
    grid = np.zeros((row_blocks, col_blocks, MXFP4_CHUNK_BYTES), dtype=np.uint8)
    for rb in range(row_blocks):
        r0 = rb * MXFP4_TILE_ROWS
        if r0 >= arr.shape[0]:
            continue
        for cb in range(col_blocks):
            c0 = cb * MXFP4_TILE_COLS
            if c0 >= arr.shape[1]:
                break
            tile = arr[r0:r0 + MXFP4_TILE_ROWS, c0:c0 + MXFP4_TILE_COLS]
            if tile.shape[0] < MXFP4_TILE_ROWS or tile.shape[1] < MXFP4_TILE_COLS:
                pad = np.zeros((MXFP4_TILE_ROWS, MXFP4_TILE_COLS), dtype=np.float32)
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            grid[rb, cb] = np.frombuffer(quantize_mxfp4_tile(tile), dtype=np.uint8)
    return grid


def _weight_to_in_out(weight):
    """HF stores gate/up/down as [out, in]; the NPU/Q4NX layout is [in, out]."""
    arr = _to_f32_np(weight)
    if arr.ndim != 2:
        raise ValueError(f"expert weight must be 2D, got shape {arr.shape}")
    return np.ascontiguousarray(arr.T)


def _quantize_dense_block(group_f32):
    """Quantize a float32 vector of 32 elements into a 17-byte dense DS4 block.

    Block layout: [1 e8m0 scale byte, 16 packed FP4 nibble bytes].
    """
    abs_max = np.abs(group_f32).max()
    scale_e8m0 = f32_to_e8m0(abs_max / 6.0) if abs_max > 0 else 0
    scale_f = e8m0_to_f32(scale_e8m0) if scale_e8m0 > 0 else 1.0
    normalized = group_f32 / scale_f
    idx = np.argmin(np.abs(FP4_LEVELS[:8] - np.abs(normalized[:, None])), axis=1)
    nibbles = np.where(normalized >= 0, idx, idx + 8).astype(np.uint8)
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)
    out = np.empty(17, dtype=np.uint8)
    out[0] = scale_e8m0
    out[1:] = packed
    return out.tobytes()


def _float32_to_dense_blocks(arr_f32):
    """Convert a float32 [out, in] matrix to dense DS4 blocks.
    Fully vectorized — no Python loops over elements."""
    out_dim, in_dim = arr_f32.shape
    num_groups = in_dim // DENSE_BLOCK_ELEMS

    # Reshape to [out_dim, num_groups, 32] and sanitize
    blocks = np.nan_to_num(arr_f32.reshape(out_dim, num_groups, DENSE_BLOCK_ELEMS),
                            nan=0.0, posinf=0.0, neginf=0.0)

    # Vectorized e8m0 scale computation
    abs_max = np.abs(blocks).max(axis=2)  # [out_dim, num_groups]
    scale_target = abs_max / 4.0
    safe_target = np.where(scale_target > 0, scale_target, 1.0).astype(np.float32)
    e_raw = np.ceil(np.log2(safe_target)).astype(np.int32)
    e_raw = np.clip(e_raw, -127, 127)
    scale_e8m0 = np.where(abs_max > 0, ((e_raw + 127) & 0xFF).astype(np.uint8), 0)

    # Convert e8m0 to float for normalization
    scale_f = np.power(2.0, (scale_e8m0.astype(np.int32) - 127).astype(np.float32))
    scale_f = np.where(scale_e8m0 > 0, scale_f, np.float32(1e-38))

    # Normalize and quantize to FP4
    normalized = blocks / scale_f[:, :, None]  # [out_dim, num_groups, 32]
    flat = normalized.reshape(-1, DENSE_BLOCK_ELEMS)
    abs_flat = np.abs(flat)

    # FP4 quantization via np.select (fastest vectorized approach)
    flat = normalized.reshape(-1, DENSE_BLOCK_ELEMS)
    abs_flat = np.abs(flat)
    conditions = [abs_flat >= 5.0, abs_flat >= 3.5, abs_flat >= 2.5, abs_flat >= 1.75,
                 abs_flat >= 1.25, abs_flat >= 0.75, abs_flat >= 0.25]
    choices = [7, 6, 5, 4, 3, 2, 1]
    idx = np.select(conditions, choices, default=0).astype(np.uint8)
    nibbles = np.where(flat >= 0, idx, idx + 8).astype(np.uint8)

    # Pack 32 nibbles -> 16 bytes
    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)

    # Concatenate scale + packed into [N, 17] and return as bytes
    scale_col = scale_e8m0.reshape(-1, 1)
    result = np.empty((scale_col.shape[0], 17), dtype=np.uint8)
    result[:, 0:1] = scale_col
    result[:, 1:] = packed
    return result.tobytes()


# ── Q4_0 quantization (simplified, fully vectorized) ─────────────────
# 32 elements per block, 18 bytes: fp16 scale (2B) + 16 packed 4-bit nibbles
# Dequant: val = (q - 8) * scale, where q is 0-15 and scale is fp16

def _f32_to_f16_vec(arr):
    """Vectorized float32 → float16 bits."""
    h = np.clip(arr, -65504.0, 65504.0)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return h.astype(np.float16).view(np.uint16)

def _quantize_q4_0_row(arr_f32):
    """Quantize a 2D float32 [out, in] array to Q4_0 blocks. Fully vectorized.
    Returns bytes of length (out * in / 32 * 18).
    """
    out_dim, in_dim = arr_f32.shape
    n_blocks = out_dim * (in_dim // Q4_0_BLOCK_ELEMS)
    # Reshape to [n_blocks, 32]
    blocks = np.nan_to_num(arr_f32.reshape(out_dim, in_dim // Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_ELEMS),
                           nan=0.0, posinf=0.0, neginf=0.0)
    blocks = blocks.reshape(-1, Q4_0_BLOCK_ELEMS)  # [n_blocks, 32]

    # Scale: abs_max / 7.0 (symmetric, levels 0-15, center at 8)
    abs_max = np.abs(blocks).max(axis=1)  # [n_blocks]
    scale = abs_max / 7.0
    scale = np.where(scale > 1e-20, scale, 1.0).astype(np.float32)

    # Quantize: q = round(x / scale) + 8, clipped to [0, 15]
    normalized = blocks / scale[:, None]
    q = np.round(normalized).astype(np.int32) + 8
    q = np.clip(q, 0, 15).astype(np.uint8)  # [n_blocks, 32]

    # Pack 32 4-bit values into 16 bytes
    lo = q[:, 0::2]  # [n_blocks, 16]
    hi = q[:, 1::2]  # [n_blocks, 16]
    packed = (lo | (hi << 4)).astype(np.uint8)  # [n_blocks, 16]

    # Scale as fp16
    scale_f16 = _f32_to_f16_vec(scale)  # [n_blocks]

    # Build output: [n_blocks, 18] = [scale(2), packed(16)]
    result = np.empty((n_blocks, Q4_0_BLOCK_BYTES), dtype=np.uint8)
    result[:, 0:2] = scale_f16.reshape(-1, 1).view(np.uint8)
    result[:, 2:] = packed

    return result.tobytes()


def _hf_fp4_to_mxfp4_tiles(w_raw, w_shape, s_raw, s_shape, row_blocks, col_blocks):
    """Convert HF FP4 packed bytes to MXFP4 tile bytes without float dequant.

    HF stores FP4 expert weights as:
      weight: int8 [out, in/2]  — packed nibbles, low nibble first
      scale:  uint8 [out, in/32] — one e8m0 per 32-element input group per output row

    The NPU MXFP4 tile layout (2560 bytes per 32x128 tile):
      [0:128)   128 e8m0 scales, one per output column
      [128:192) 32 BF16 biases (zero)
      [192:512) padding (zero)
      [512:2560) 2048 packed nibbles: byte[row*64 + col//2], lo if col even

    Scale mapping: tile(rb,cb).scale[c] = HF_scale[cb*128+c, rb]
    Nibble mapping: transpose HF [out,in] -> tile [32,128], repack along col axis.
    """
    out_dim, in_half = w_shape
    in_dim = in_half * 2

    w = np.frombuffer(w_raw, dtype=np.uint8).reshape(out_dim, in_half)
    lo = w & 0x0F
    hi = (w >> 4) & 0x0F
    nibbles = np.empty((out_dim, in_dim), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi

    scales = np.frombuffer(s_raw, dtype=np.uint8).reshape(out_dim, in_dim // 32)

    result = bytearray(row_blocks * col_blocks * MXFP4_CHUNK_BYTES)

    for rb in range(row_blocks):
        r0 = rb * MXFP4_TILE_ROWS
        if r0 >= in_dim:
            continue
        for cb in range(col_blocks):
            c0 = cb * MXFP4_TILE_COLS
            if c0 >= out_dim:
                break

            tile_off = (rb * col_blocks + cb) * MXFP4_CHUNK_BYTES

            block = nibbles[c0:c0 + MXFP4_TILE_COLS,
                            r0:r0 + MXFP4_TILE_ROWS].T

            tile_scales = scales[c0:c0 + MXFP4_TILE_COLS, rb]

            result[tile_off:tile_off + MXFP4_SCALE_BYTES] = tile_scales.tobytes()

            packed = (block[:, 0::2] | (block[:, 1::2] << 4)).astype(np.uint8)
            result[tile_off + 512:tile_off + 512 + MXFP4_NIBBLE_BYTES] = \
                packed.tobytes()

    return bytes(result)


def _hf_fp4_to_dense_blocks(w_raw, w_shape, s_raw, s_shape):
    """Convert HF FP4 to DS4 dense block format without float dequant.

    DS4 dense block layout (17 bytes per 32 elements):
      byte 0:  e8m0 scale for this 32-element group
      bytes 1..16: 16 packed FP4 nibbles (low nibble first, 2 per byte)

    Vectorized: operates on all output rows and groups simultaneously.
    """
    out_dim, in_half = w_shape
    in_dim = in_half * 2
    num_groups = in_dim // DENSE_BLOCK_ELEMS

    w = np.frombuffer(w_raw, dtype=np.uint8).reshape(out_dim, in_half)
    lo = w & 0x0F
    hi = (w >> 4) & 0x0F
    nibbles = np.empty((out_dim, in_dim), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi

    scales = np.frombuffer(s_raw, dtype=np.uint8).reshape(out_dim, num_groups)

    # Reshape nibbles to [out_dim, num_groups, 32]
    nib_groups = nibbles.reshape(out_dim, num_groups, DENSE_BLOCK_ELEMS)

    # Pack 32 nibbles -> 16 bytes
    lo_pack = nib_groups[:, :, 0::2]
    hi_pack = nib_groups[:, :, 1::2]
    packed = (lo_pack | (hi_pack << 4)).astype(np.uint8)  # [out_dim, num_groups, 16]

    # Concatenate scale byte + 16 packed bytes -> [out_dim, num_groups, 17]
    scale_col = scales.reshape(out_dim, num_groups, 1)
    result = np.concatenate([scale_col, packed], axis=2)
    return result.tobytes()


def pack_expert_raw_interleaved(gate, up, down):
    """Pack raw HF FP4 expert data into one EXPERT_BLOCK_BYTES block.

    Each argument is a tuple: (w_raw, w_shape, s_raw, s_shape).
    Dense block format: [1 scale_byte, 16 nibble_bytes] per 32 elements.
    No float32 dequantization — pure byte-level interleaving.
    """
    g = _hf_fp4_to_dense_blocks(*gate)
    u = _hf_fp4_to_dense_blocks(*up)
    d = _hf_fp4_to_dense_blocks(*down)
    return g + u + d


def pack_expert_mxfp4_from_float(gate, up, down):
    """Pack float32 [out, in] gate/up/down into one MXFP4 dense expert block.

    Debug/legacy path only — goes through float32. Production uses
    pack_expert_raw_interleaved (pure byte, no float roundtrip). Emits B[N, K]
    row-major (no transpose) to match the transposed-B expert GEMM.
    """
    g = _float32_to_dense_blocks(_to_f32_np(gate))
    u = _float32_to_dense_blocks(_to_f32_np(up))
    d = _float32_to_dense_blocks(_to_f32_np(down))
    return g + u + d


def quantize_q8_0(tensor):
    """GGUF Q8_0: 32-element blocks of (fp16 scale, 32 int8). Symmetric."""
    arr = _to_f32_np(tensor).ravel()
    n = arr.size
    n_blocks = (n + Q8_0_BLOCK - 1) // Q8_0_BLOCK
    pad = np.zeros(n_blocks * Q8_0_BLOCK, dtype=np.float32)
    pad[:n] = arr
    blocks = pad.reshape(n_blocks, Q8_0_BLOCK)
    abs_max = np.abs(blocks).max(axis=1)
    scale = abs_max / 127.0
    scale[scale < 1e-12] = 0.0
    safe = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
    q = np.clip(np.round(blocks / safe[:, None]), -127, 127).astype(np.int8)
    scale_fp16 = scale.astype(np.float16)
    out = bytearray()
    out.extend(scale_fp16.tobytes())
    out.extend(q.tobytes())
    return bytes(out)


def _bf16_bytes(tensor):
    arr = _to_f32_np(tensor)
    bits = arr.astype(np.float32).view(np.uint32)
    bf16 = (bits >> 16).astype(np.uint16)
    return bf16.tobytes()


def _quant_shared(tensor, qtype):
    if qtype == QTYPE_Q8_0:
        return quantize_q8_0(tensor)
    if qtype == QTYPE_BF16:
        return _bf16_bytes(tensor)
    if qtype == QTYPE_F32:
        return _to_f32_np(tensor).tobytes()
    raise ValueError(f"bad qtype {qtype}")


def _parse_state_dict(state_dict):
    """Split a HF state_dict into config, shared entries, and expert groups.

    Supports both legacy (model.layers.N.mlp.experts.E.gate_proj) and
    DeepSeek V4 (layers.N.ffn.experts.E.w1) tensor naming.
    Handles Q8_0 INT8 weights with separate .scale companions.

    Returns (config, shared_list, expert_groups) where
      shared_list  = list of (tensor_id, layer_id, sub_id, qtype, tensor)
      expert_groups= { layer: { expert_id: {'gate_proj':..,'up_proj':..,'down_proj':..} } }
    """
    num_layers = 0
    num_experts = 0
    hidden = None
    inter = None
    vocab = None
    shared = []
    experts = {}

    def _maybe_dequant_q8(t, base_key):
        """If base_key.replace('.weight','.scale') exists, dequant: fp8/int8 * scale -> f32.
        Handles per-channel (1D scale) and per-block (2D scale) layouts.
        """
        scale_key = base_key.replace(".weight", ".scale")
        if scale_key not in state_dict:
            return _to_f32_np(t)
        scale_t = state_dict[scale_key]
        scale = _to_f32_np(scale_t)
        w_f32 = _to_f32_np(t)

        if w_f32.ndim == 2 and scale.ndim == 2 and w_f32.shape[0] % scale.shape[0] == 0:
            bh = w_f32.shape[0] // scale.shape[0]
            bw = w_f32.shape[1] // scale.shape[1]
            s = scale[:, np.newaxis, :, np.newaxis]
            deq = w_f32.reshape(scale.shape[0], bh, scale.shape[1], bw) * s
            deq = deq.reshape(w_f32.shape)
        else:
            deq = w_f32 * scale
        return deq.astype(np.float32)

    for key, t in state_dict.items():
        k = key
        if k in ("embed.weight", "embed_tokens.weight", "model.embed_tokens.weight"):
            vocab = int(_shape_of(t)[0])
            shared.append((TID_EMBED, GLOBAL_LAYER, 0, QTYPE_BF16, t))
            continue
        if k in ("norm.weight", "model.norm.weight", "model.final_norm.weight"):
            shared.append((TID_OUTPUT_NORM, GLOBAL_LAYER, 0, QTYPE_BF16, t))
            continue
        if k in ("head.weight", "lm_head.weight", "output.weight"):
            shared.append((TID_LM_HEAD, GLOBAL_LAYER, 0, QTYPE_BF16, t))
            continue

        # Parse layer ID from either "model.layers.N." or "layers.N." prefix
        prefix = None
        if k.startswith("model.layers."):
            prefix = "model.layers."
        elif k.startswith("layers."):
            prefix = "layers."
        else:
            continue
        rest = k[len(prefix):]
        dot = rest.find(".")
        layer_id = int(rest[:dot])
        suffix = rest[dot + 1:]
        num_layers = max(num_layers, layer_id + 1)

        if suffix in ("attn_norm.weight", "input_layernorm.weight"):
            if hidden is None:
                hidden = int(t.shape[0])
            shared.append((TID_INPUT_NORM, layer_id, 0, QTYPE_BF16, t))
        elif suffix in ("ffn_norm.weight", "post_attention_layernorm.weight"):
            shared.append((TID_POST_ATTN_NORM, layer_id, 0, QTYPE_BF16, t))
        elif suffix in ("attn.q_norm.weight", "attention.q_norm.weight"):
            shared.append((TID_Q_NORM, layer_id, 0, QTYPE_BF16, t))
        elif suffix in ("attn.kv_norm.weight", "attention.kv_norm.weight"):
            shared.append((TID_KV_NORM, layer_id, 0, QTYPE_BF16, t))
        elif suffix in ("ffn.gate.weight", "mlp.gate.weight"):
            shared.append((TID_ROUTER, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("ffn.gate.e_score_correction_bias", "mlp.gate.e_score_correction_bias", "mlp.gate.bias"):
            shared.append((TID_ROUTER_BIAS, layer_id, 0, QTYPE_BF16, t))
        elif suffix in ("ffn.shared_experts.w1.weight", "mlp.shared_experts.gate_proj.weight"):
            shared.append((TID_SHARED_GATE, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("ffn.shared_experts.w3.weight", "mlp.shared_experts.up_proj.weight"):
            shared.append((TID_SHARED_UP, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("ffn.shared_experts.w2.weight", "mlp.shared_experts.down_proj.weight"):
            shared.append((TID_SHARED_DOWN, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("attn.wq_a.weight", "attention.wq_a.weight"):
            shared.append((TID_Q_PROJ, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("attn.wq_b.weight", "attention.wq_b.weight"):
            shared.append((TID_Q_PROJ, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("attn.wkv.weight", "attention.wkv.weight"):
            shared.append((TID_K_PROJ, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("attn.wo_a.weight", "attention.wo_a.weight"):
            shared.append((TID_O_PROJ, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix in ("attn.wo_b.weight", "attention.wo_b.weight"):
            shared.append((TID_O_PROJ, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
        elif suffix.startswith("ffn.shared_experts.") or suffix.startswith("mlp.shared_experts."):
            parts = suffix.split(".")
            if len(parts) >= 3 and parts[-1] == "weight":
                proj_raw = parts[-2]
                proj = {"0": "shared_gate", "1": "shared_up", "2": "shared_down",
                        "w1": "shared_gate", "w3": "shared_up", "w2": "shared_down",
                        "gate_proj": "shared_gate", "up_proj": "shared_up", "down_proj": "shared_down"}.get(proj_raw, None)
                if proj == "shared_gate":
                    shared.append((TID_SHARED_GATE, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
                elif proj == "shared_up":
                    shared.append((TID_SHARED_UP, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
                elif proj == "shared_down":
                    shared.append((TID_SHARED_DOWN, layer_id, 0, QTYPE_BF16, _maybe_dequant_q8(t, k)))
                else:
                    shared.append((TID_SHARED_GATE, layer_id, int(parts[-2]) if parts[-2].isdigit() else 0, QTYPE_Q8_0, t))
        elif suffix.startswith("ffn.experts.") or suffix.startswith("mlp.experts."):
            parts = suffix.split(".")
            eid = int(parts[2])
            proj_raw = parts[3]
            if proj_raw in ("weight", "scale"):
                continue
            proj = {"gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj",
                    "w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[proj_raw]
            num_experts = max(num_experts, eid + 1)
            experts.setdefault(layer_id, {}).setdefault(eid, {})[proj] = _maybe_dequant_q8(t, k)
            if inter is None and proj == "gate_proj":
                inter = int(t.shape[0])
        else:
            shared.append((TID_INPUT_NORM, layer_id, 1, QTYPE_BF16, t))

    config = {
        "hidden_dim": hidden or V4_HIDDEN,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_inter_dim": inter or V4_INTERMEDIATE,
        "vocab_size": vocab or 0,
    }
    return config, shared, experts


def _pack_header(cfg, shared_dir_offset, shared_dir_count,
                 expert_bank_offset, expert_count_total):
    dspark_block_size = int(cfg.get("dspark_block_size", 0))
    dspark_markov_rank = int(cfg.get("dspark_markov_rank", 0))
    r1 = (dspark_block_size & 0xFFFF) | ((dspark_markov_rank & 0xFFFF) << 16)
    r2 = int(cfg.get("dspark_noise_token_id", 0))
    return struct.pack(
        HEADER_FMT,
        FST_MAGIC,
        FST_VERSION,
        int(cfg.get("hidden_dim", 0)),
        int(cfg.get("num_layers", 0)),
        int(cfg.get("num_experts", 0)),
        int(cfg.get("top_k", 8)),
        int(cfg.get("num_q_heads", cfg.get("hidden_dim", 2048) // 128)),
        int(cfg.get("num_kv_heads", 4)),
        int(cfg.get("head_dim", 128)),
        int(cfg.get("expert_inter_dim", 0)),
        int(cfg.get("num_shared_experts", 1)),
        0,
        int(cfg.get("vocab_size", 0)),
        float(cfg.get("rms_eps", 1e-6)),
        float(cfg.get("rope_freq_base", 10000.0)),
        int(shared_dir_offset),
        int(shared_dir_count),
        int(expert_bank_offset),
        int(EXPERT_BLOCK_BYTES),
        int(EXPERT_BLOCK_STRIDE),
        int(expert_count_total),
        r1, r2,
    )


def _shape_of(t):
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        return tuple(int(x) for x in t.shape)
    return tuple(int(x) for x in np.asarray(t).shape)


def convert_to_fst(state_dict, output_path, config=None):
    """Write a HuggingFace state_dict to a .fst file at output_path."""
    cfg, shared, experts = _parse_state_dict(state_dict)
    if config:
        cfg.update(config)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    shared_dir_offset = align_up(HEADER_SIZE)
    shared_dir_count = len(shared)
    dir_size = shared_dir_count * SHARED_ENTRY_SIZE
    data_offset = align_up(shared_dir_offset + dir_size)

    with open(out, "wb") as f:
        f.write(b"\x00" * data_offset)        # reserve header + dir + alignment

        dir_entries = []
        for (tid, layer_id, sub_id, qtype, t) in shared:
            shape = _shape_of(t)
            ndim = len(shape)
            shape3 = (shape + (1, 1, 1))[:3]
            data = _quant_shared(t, qtype)
            data_off = align_up(f.tell())
            f.seek(data_off)
            f.write(data)
            dir_entries.append(struct.pack(
                SHARED_ENTRY_FMT,
                tid, layer_id, sub_id, qtype, ndim, 0,
                int(shape3[0]), int(shape3[1]), int(shape3[2]),
                int(data_off), int(len(data)), 0,
            ))

        after_shared_data = align_up(f.tell())

        dir_blob = b"".join(dir_entries)
        f.seek(shared_dir_offset)
        f.write(dir_blob)

        expert_bank_offset = after_shared_data
        f.seek(expert_bank_offset)

        n_layers = cfg["num_layers"]
        n_experts = cfg["num_experts"]
        for layer_id in range(n_layers):
            layer_experts = experts.get(layer_id, {})
            for eid in range(n_experts):
                grp = layer_experts.get(eid, {})
                gate = grp.get("gate_proj")
                up = grp.get("up_proj")
                down = grp.get("down_proj")
                if gate is None or up is None or down is None:
                    block = b"\x00" * EXPERT_BLOCK_BYTES
                elif isinstance(gate, tuple):
                    block = pack_expert_raw_interleaved(gate, up, down)
                else:
                    block = pack_expert_mxfp4_from_float(gate, up, down)
                assert len(block) == EXPERT_BLOCK_BYTES
                pos = expert_bank_offset + (layer_id * n_experts + eid) * EXPERT_BLOCK_STRIDE
                f.seek(pos)
                f.write(block)

        f.seek(0, 2)
        end = align_up(f.tell())
        f.truncate(end)

        f.seek(0)
        f.write(_pack_header(
            cfg, shared_dir_offset, shared_dir_count,
            expert_bank_offset, n_layers * n_experts,
        ))

    return str(out)


def _f8_e4m3_to_f32(x):
    """Convert F8_E4M3 uint8 array to float32. OCP standard: 1 sign, 4 exp, 3 mant, bias=7."""
    bits = np.asarray(x, dtype=np.uint8)
    sign = ((bits >> 7) & 1).astype(np.float32)
    exp = ((bits >> 3) & 0xF).astype(np.int32)
    mant = (bits & 0x7).astype(np.float32)
    sign_f = 1.0 - 2.0 * sign
    result = np.zeros(bits.shape, dtype=np.float32)
    normal = (exp >= 1) & (exp <= 14)
    if np.any(normal):
        e = exp[normal].astype(np.float32) - 7.0
        m = 1.0 + mant[normal] / 8.0
        result[normal] = sign_f[normal] * m * np.power(2.0, e)
    denormal = exp == 0
    if np.any(denormal):
        m = mant[denormal] / 8.0
        result[denormal] = sign_f[denormal] * m * np.power(2.0, -6.0)
    return result


def _e8m0_scale_to_f32(scale_u8):
    """Convert F8_E8M0 uint8 array to float32. bias=127."""
    e = scale_u8.astype(np.int32)
    result = np.ones(scale_u8.shape, dtype=np.float32)
    valid = (e > 0) & (e < 255)
    result[valid] = np.power(2.0, (e[valid] - 127).astype(np.float32))
    result[e == 0] = np.finfo(np.float32).tiny
    return result


def _broadcast_scale_to_weight(scale_f32, weight_shape):
    """Expand a 2D e8m0 scale grid to match a 2D weight shape.

    scale_f32 has shape [a, b]. weight has shape [a * block_r, b * block_c].
    Returns array of shape weight_shape.
    """
    a, b = scale_f32.shape
    block_r = weight_shape[0] // a
    block_c = weight_shape[1] // b
    if block_r > 1:
        scale_f32 = np.repeat(scale_f32, block_r, axis=0)
    if block_c > 1:
        scale_f32 = np.repeat(scale_f32, block_c, axis=1)
    return scale_f32


def dequant_v4_fp4(weight_i8, scale_e8m0):
    """Dequantize DeepSeek V4 FP4 weights to float32.

    Args:
        weight_i8: int8 array [out, in/2] -- two fp4 nibbles per byte (low first).
        scale_e8m0: uint8 array [out, in/32] -- one e8m0 exponent per 32-element group.

    Returns:
        float32 array [out, in] of dequantized weights.
    """
    w = weight_i8.astype(np.uint8)
    lo = w & 0x0F
    hi = (w >> 4) & 0x0F
    fp4 = np.empty((w.shape[0], w.shape[1] * 2), dtype=np.uint8)
    fp4[:, 0::2] = lo
    fp4[:, 1::2] = hi

    scales = _e8m0_scale_to_f32(scale_e8m0)
    scales_exp = _broadcast_scale_to_weight(scales, fp4.shape)

    return FP4_TABLE[fp4] * scales_exp


def dequant_mxint8(weight_i8, scale_e8m0):
    """Dequantize MXINT8 weights (int8 values with e8m0 per-block scales).

    Args:
        weight_i8: int8 array [out, in] -- one quantized int8 value per element.
        scale_e8m0: uint8 array [out, in/32] -- one e8m0 exponent per 32-element group.

    Returns:
        float32 array [out, in] of dequantized weights.
    """
    scales = _e8m0_scale_to_f32(scale_e8m0)
    scales_exp = _broadcast_scale_to_weight(scales, weight_i8.shape)
    return weight_i8.astype(np.float32) * scales_exp


def dequant_f8_e4m3(weight_u8, scale_e8m0=None):
    """Dequantize F8_E4M3 weight with optional F8_E8M0 per-block scale.

    Args:
        weight_u8: uint8 array [out, in] -- raw F8_E4M3 bytes.
        scale_e8m0: uint8 array [a, b] -- F8_E8M0 scale grid (optional).

    Returns:
        float32 array [out, in] of dequantized weights.
    """
    w_f32 = _f8_e4m3_to_f32(weight_u8)
    if scale_e8m0 is not None and scale_e8m0.size > 0:
        scale_f32 = _e8m0_scale_to_f32(scale_e8m0)
        scale_exp = _broadcast_scale_to_weight(scale_f32, w_f32.shape)
        w_f32 = w_f32 * scale_exp
    return w_f32


# ─── Safetensors low-level helpers ─────────────────────────────────────


def _decode_safetensor_raw(raw, dtype, shape):
    """Decode raw bytes from safetensors into a float32 numpy array."""
    if dtype == "BF16":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        f32 = np.zeros(arr.shape, dtype=np.float32)
        f32.view(np.uint32)[:] = np.uint32(arr) << 16
        return f32
    elif dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    elif dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    elif dtype == "I64":
        return np.frombuffer(raw, dtype=np.int64).reshape(shape).copy()
    elif dtype == "I8":
        return np.frombuffer(raw, dtype=np.int8).reshape(shape).copy()
    elif dtype == "F8_E4M3":
        return np.frombuffer(raw, dtype=np.int8).reshape(shape).copy()
    elif dtype == "F8_E8M0":
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    else:
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def _read_safetensors_header(shard_path):
    """Read safetensors header without loading data.

    Returns (header_dict, data_start_offset).
    """
    p = Path(shard_path)
    with open(p, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        data_start = 8 + hlen
    return hdr, data_start


def _load_single_tensor_from_shard(shard_path, hdr, data_start, key):
    """Load one tensor from a safetensors shard given pre-read header."""
    if key not in hdr:
        return None
    entry = hdr[key]
    off0, off1 = entry["data_offsets"]
    dtype = entry["dtype"]
    shape = tuple(entry["shape"])

    p = Path(shard_path)
    with open(p, "rb") as f:
        f.seek(data_start + off0)
        raw = f.read(off1 - off0)

    return _decode_safetensor_raw(raw, dtype, shape)


def _dequant_fp4_from_shard(shard_path, hdr, data_start, weight_key):
    """Load FP4/F8_E4M3 weight + its scale companion from the same shard, dequantize."""
    weight_entry = hdr.get(weight_key)
    if weight_entry is None:
        return None

    scale_key = weight_key.replace(".weight", ".scale")
    scale_entry = hdr.get(scale_key)

    w_off0, w_off1 = weight_entry["data_offsets"]
    p = Path(shard_path)
    with open(p, "rb") as f:
        f.seek(data_start + w_off0)
        w_raw = f.read(w_off1 - w_off0)

    dtype = weight_entry["dtype"]
    w_shape = tuple(weight_entry["shape"])

    if scale_entry is not None:
        s_off0, s_off1 = scale_entry["data_offsets"]
        with open(p, "rb") as f:
            f.seek(data_start + s_off0)
            s_raw = f.read(s_off1 - s_off0)
        scale = np.frombuffer(s_raw, dtype=np.uint8).reshape(tuple(scale_entry["shape"]))
    else:
        scale = None

    if dtype == "F8_E4M3":
        weight = np.frombuffer(w_raw, dtype=np.uint8).reshape(w_shape)
        return dequant_f8_e4m3(weight, scale)
    else:
        # INT8 with e8m0 block scales — could be MXINT8 or FP4 (nibbles).
        # MXINT8: w_shape[1] == scale_shape[1] * block_size (one int8 per element)
        #   where block_size = w_shape[1] / scale_shape[1] (e.g. 32).
        # FP4:    w_shape[1] == scale_shape[1] * block_size / 2 (two nibbles per byte)
        #   where block_size = 32 (32 nibbles per scale, 16 packed bytes per scale).
        # For FP4: w_shape[1] / scale_shape[1] = 16 (16 bytes = 32 nibbles).
        # For MXINT8: w_shape[1] / scale_shape[1] = 32 (32 int8 values).
        weight = np.frombuffer(w_raw, dtype=np.int8).reshape(w_shape)
        if scale is None:
            return None
        scale_shape = tuple(scale_entry["shape"])
        scale_w = scale_shape[1] if len(scale_shape) >= 2 else 1
        packed_per_scale = w_shape[1] // scale_w if scale_w > 0 else 32
        if packed_per_scale == 16:
            # FP4: 16 packed bytes per scale → 32 nibbles per scale
            return dequant_v4_fp4(weight, scale)
        else:
            # MXINT8: each int8 is a full element
            return dequant_mxint8(weight, scale)


def load_v4_safetensors(shard_path, layer_id, num_experts=256):
    """Load one layer's tensors from a DeepSeek V4 safetensors shard.

    Returns a state_dict-like dict mapping HF-compatible names to numpy arrays.
    Expert weights are dequantized from V4 FP4 to BF16 (float32 numpy).

    Handles both ``layers.N.`` and ``model.layers.N.`` key prefixes.
    Dequantizes w1, w2, and w3 (not just w1).
    """
    p = Path(shard_path)
    hdr, data_start = _read_safetensors_header(p)

    prefix1 = f"layers.{layer_id}."
    prefix2 = f"model.layers.{layer_id}."
    sd = {}
    for k, v in hdr.items():
        if not (k.startswith(prefix1) or k.startswith(prefix2)):
            continue

        off0, off1 = v["data_offsets"]
        dtype = v["dtype"]
        shape = tuple(v["shape"])

        with open(p, "rb") as f:
            f.seek(data_start + off0)
            raw = f.read(off1 - off0)

        if dtype == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            f32 = np.zeros(arr.shape, dtype=np.float32)
            f32.view(np.uint32)[:] = np.uint32(arr) << 16
            sd[k] = f32
        elif dtype == "F16":
            sd[k] = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        elif dtype == "F32":
            sd[k] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        elif dtype == "I64":
            sd[k] = np.frombuffer(raw, dtype=np.int64).reshape(shape)
        elif dtype == "I8":
            if ".weight" in k:
                deq = _dequant_fp4_from_shard(p, hdr, data_start, k)
                if deq is not None:
                    sd[k] = deq
                    continue
            sd[k] = np.frombuffer(raw, dtype=np.int8).reshape(shape)
        elif dtype == "F8_E4M3":
            sd[k] = np.frombuffer(raw, dtype=np.int8).reshape(shape)
        elif dtype == "F8_E8M0":
            sd[k] = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        else:
            sd[k] = np.frombuffer(raw, dtype=np.uint8).reshape(shape)

    return sd


def convert_v4_layer(model_dir, layer_id, output_path, num_experts=256):
    """Convert one DeepSeek V4 layer from safetensors to a .fst file.

    Loads the shard containing the requested layer, dequants FP4 experts to
    BF16, and packs them into the .fst MXFP4 expert format.
    """
    import time

    model_dir = Path(model_dir)
    snapshot = next((model_dir / "snapshots").iterdir())

    index_path = snapshot / "model.safetensors.index.json"
    with open(index_path, "rb") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    layer_prefix = f"layers.{layer_id}."
    layer_prefix2 = f"model.layers.{layer_id}."
    shard_names = set(
        weight_map[k] for k in weight_map
        if k.startswith(layer_prefix) or k.startswith(layer_prefix2)
    )
    if not shard_names:
        raise FileNotFoundError(f"No tensors for layer {layer_id} in index")

    t0 = time.time()

    sd = {}
    for shard_name in sorted(shard_names):
        shard_path = snapshot / shard_name
        layer_sd = load_v4_safetensors(shard_path, layer_id, num_experts)
        sd.update(layer_sd)

    load_time = time.time() - t0
    print(f"  loaded {len(sd)} tensors from {len(shard_names)} shard(s) in {load_time:.1f}s")

    n_experts_actual = max(
        (int(k.split(".")[4]) for k in sd
         if (".ffn.experts." in k or ".mlp.experts." in k) and ".w1.weight" in k),
        default=0,
    ) + 1

    config = {
        "hidden_dim": V4_HIDDEN,
        "num_layers": 1,
        "num_experts": n_experts_actual,
        "expert_inter_dim": V4_INTERMEDIATE,
        "top_k": 6,
        "num_q_heads": 64,
        "num_kv_heads": 1,
        "head_dim": 512,
        "vocab_size": 129280,
    }

    t1 = time.time()
    path = convert_to_fst(sd, output_path, config=config)
    convert_time = time.time() - t1
    total = time.time() - t0

    out_size = Path(path).stat().st_size
    print(f"  converted {n_experts_actual} experts in {convert_time:.1f}s")
    print(f"  wrote {path} ({out_size / 1e9:.2f} GB)")
    print(f"  total: {total:.1f}s (load {load_time:.1f}s + convert {convert_time:.1f}s)")
    print(f"  expert block: {EXPERT_BLOCK_BYTES} bytes, stride {EXPERT_BLOCK_STRIDE}")
    return path


# ─── Full-model streaming converter ────────────────────────────────────


_LAYER_SUFFIX_TIDS = {
    "attn_norm.weight":           (TID_INPUT_NORM,     0, QTYPE_BF16, False),
    "input_layernorm.weight":     (TID_INPUT_NORM,     0, QTYPE_BF16, False),
    "ffn_norm.weight":            (TID_POST_ATTN_NORM, 0, QTYPE_BF16, False),
    "post_attention_layernorm.weight": (TID_POST_ATTN_NORM, 0, QTYPE_BF16, False),
    "attn.q_norm.weight":         (TID_Q_NORM,  0, QTYPE_BF16, False),
    "attention.q_norm.weight":    (TID_Q_NORM,  0, QTYPE_BF16, False),
    "attn.kv_norm.weight":        (TID_KV_NORM, 0, QTYPE_BF16, False),
    "attention.kv_norm.weight":   (TID_KV_NORM, 0, QTYPE_BF16, False),
    "attn.attn_sink":             (TID_ATTN_SINK, 0, QTYPE_F32, False),
    "ffn.gate.weight":            (TID_ROUTER, 0, QTYPE_BF16, False),
    "mlp.gate.weight":            (TID_ROUTER, 0, QTYPE_BF16, False),
    "ffn.gate.e_score_correction_bias": (TID_ROUTER_BIAS, 0, QTYPE_BF16, False),
    "mlp.gate.e_score_correction_bias": (TID_ROUTER_BIAS, 0, QTYPE_BF16, False),
    "mlp.gate.bias":              (TID_ROUTER_BIAS, 0, QTYPE_F32, False),
    "ffn.gate.bias":              (TID_ROUTER_BIAS, 0, QTYPE_F32, False),
    "ffn.shared_experts.w1.weight":  (TID_SHARED_GATE, 0, QTYPE_BF16, True),
    "mlp.shared_experts.gate_proj.weight": (TID_SHARED_GATE, 0, QTYPE_BF16, True),
    "ffn.shared_experts.w3.weight":  (TID_SHARED_UP, 0, QTYPE_BF16, True),
    "mlp.shared_experts.up_proj.weight": (TID_SHARED_UP, 0, QTYPE_BF16, True),
    "ffn.shared_experts.w2.weight":  (TID_SHARED_DOWN, 0, QTYPE_BF16, True),
    "mlp.shared_experts.down_proj.weight": (TID_SHARED_DOWN, 0, QTYPE_BF16, True),
    "attn.wq_a.weight":           (TID_Q_PROJ, 0, QTYPE_BF16, True),
    "attention.wq_a.weight":      (TID_Q_PROJ, 0, QTYPE_BF16, True),
    "attn.wq_b.weight":           (TID_Q_PROJ, 1, QTYPE_BF16, True),
    "attention.wq_b.weight":      (TID_Q_PROJ, 1, QTYPE_BF16, True),
    "attn.wkv.weight":            (TID_K_PROJ, 0, QTYPE_BF16, True),
    "attention.wkv.weight":       (TID_K_PROJ, 0, QTYPE_BF16, True),
    "attn.wo_a.weight":           (TID_O_PROJ, 0, QTYPE_BF16, True),
    "attention.wo_a.weight":      (TID_O_PROJ, 0, QTYPE_BF16, True),
    "attn.wo_b.weight":           (TID_O_PROJ, 1, QTYPE_BF16, True),
    "attention.wo_b.weight":      (TID_O_PROJ, 1, QTYPE_BF16, True),
    # ── Hybrid Connection (HC) 4-stream residual (DeepSeek V4) ──────────
    # NOTE: HF keys have NO ".weight" suffix (layers.N.hc_attn_fn).  fn is stored
    # [24, 4*hidden] F32 in the checkpoint; the engine transposes to [4*hidden,24]
    # and casts to BF16 for the NPU matvec.  scale/base are tiny F32.
    "hc_attn_fn":                 (TID_HC_ATTN_FN,    0, QTYPE_BF16, False),
    "hc_attn_scale":              (TID_HC_ATTN_SCALE, 0, QTYPE_F32,  False),
    "hc_attn_base":               (TID_HC_ATTN_BASE,  0, QTYPE_F32,  False),
    "hc_ffn_fn":                  (TID_HC_FFN_FN,     0, QTYPE_BF16, False),
    "hc_ffn_scale":               (TID_HC_FFN_SCALE,  0, QTYPE_F32,  False),
    "hc_ffn_base":                (TID_HC_FFN_BASE,   0, QTYPE_F32,  False),
    # ── V4 attention KV compressor (DeepSeek V4 Flash) ─────────────────
    # layers.N.attn.compressor.ape (no ".weight"): F32 [ratio, comp_width].
    # wkv/wgate BF16 [comp_width, 4096] bcol-native [N,K] — no dequant.
    # norm BF16 [512].  Present only for layers with compress_ratio != 0.
    "attn.compressor.ape":         (TID_ATTN_COMPRESSOR_APE,  0, QTYPE_F32,  False),
    "attn.compressor.wkv.weight":  (TID_ATTN_COMPRESSOR_WKV,  0, QTYPE_BF16, False),
    "attn.compressor.wgate.weight":(TID_ATTN_COMPRESSOR_GATE, 0, QTYPE_BF16, False),
    "attn.compressor.norm.weight": (TID_ATTN_COMPRESSOR_NORM, 0, QTYPE_BF16, False),
    # ── DSpark draft-specific ──────────────────────────────────────────
    "main_proj.weight":           (TID_MAIN_PROJ,   0, QTYPE_Q8_0, True),
    "main_norm.weight":           (TID_MAIN_NORM,   0, QTYPE_BF16, False),
    "norm.weight":                (TID_DRAFT_NORM,  0, QTYPE_BF16, False),
    "markov_head.markov_w1.weight": (TID_MARKOV_W1, 0, QTYPE_BF16, False),
    "markov_head.markov_w2.weight": (TID_MARKOV_W2, 0, QTYPE_BF16, False),
    "confidence_head.proj.weight": (TID_CONFIDENCE_PROJ, 0, QTYPE_BF16, False),
}

_EXPERT_PROJ_MAP = {
    "gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj",
    "w1": "gate_proj", "w3": "up_proj", "w2": "down_proj",
}


def _strip_layer_prefix(key):
    """Return (layer_id, suffix) if key has a layer prefix, else (None, None)."""
    for prefix in ("model.layers.", "layers."):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            dot = rest.find(".")
            if dot >= 0:
                return int(rest[:dot]), rest[dot + 1:]
    return None, None


def _strip_mtp_prefix(key):
    """Return (stage_id, suffix) if key has an mtp prefix, else (None, None)."""
    if key.startswith("mtp."):
        rest = key[4:]
        dot = rest.find(".")
        if dot >= 0:
            return int(rest[:dot]), rest[dot + 1:]
    return None, None


def _is_expert_suffix(suffix):
    return suffix.startswith("ffn.experts.") or suffix.startswith("mlp.experts.")


def _is_scale_suffix(suffix):
    return suffix.endswith(".scale")


def _parse_expert_suffix(suffix):
    """Return (expert_id, proj_name) or (None, None)."""
    parts = suffix.split(".")
    if len(parts) < 4:
        return None, None
    try:
        eid = int(parts[2])
    except ValueError:
        return None, None
    proj_raw = parts[3]
    proj = _EXPERT_PROJ_MAP.get(proj_raw)
    return eid, proj


def convert_full_model_v4(model_dir, output_path, verbose=False, max_layers=None):
    """Convert full DeepSeek V4 Flash model to .fst with memory-efficient streaming.

    Memory strategy
    ---------------
    * Global tensors (embed ~1 GB, norm ~8 KB, lm_head ~1 GB): loaded once.
    * Per-layer shared tensors (norms, attention, router, shared experts):
      accumulated in a list; typically ~3-5 GB total.
    * Expert weights: loaded one layer at a time from the safetensors shard,
      dequantized from FP4, repacked to MXFP4, written to disk, then freed.
      Peak additional memory ~3 GB per layer.

    The .fst file is written in a single pass:
      header -> shared directory -> shared data -> expert bank -> final header
    """
    t_start = time.time()
    model_dir = Path(model_dir)

    if (model_dir / "snapshots").exists():
        snapshot = next((model_dir / "snapshots").iterdir())
    else:
        snapshot = model_dir

    # ── Read index & config ──────────────────────────────────────────
    index_path = snapshot / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    config_path = snapshot / "config.json"
    hf_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            hf_cfg = json.load(f)

    cfg = {
        "hidden_dim": hf_cfg.get("hidden_size", V4_HIDDEN),
        "num_layers": hf_cfg.get("num_hidden_layers", 43),
        "num_experts": hf_cfg.get("n_routed_experts", 256),
        "expert_inter_dim": hf_cfg.get("intermediate_size", V4_INTERMEDIATE),
        "top_k": hf_cfg.get("num_experts_per_tok", 6),
        "num_q_heads": hf_cfg.get("num_attention_heads", 64),
        "num_kv_heads": hf_cfg.get("num_key_value_heads", 1),
        "head_dim": hf_cfg.get("head_dim", 128),
        "vocab_size": hf_cfg.get("vocab_size", 129280),
        "rms_eps": hf_cfg.get("rms_norm_eps", 1e-6),
        "rope_freq_base": hf_cfg.get("rope_theta", 10000.0),
    }
    n_layers = cfg["num_layers"]
    n_experts = cfg["num_experts"]
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)
        cfg["num_layers"] = n_layers
    print(f"Model: {n_layers} layers, {n_experts} experts, "
          f"hidden={cfg['hidden_dim']}, vocab={cfg['vocab_size']}")

    # ── Pre-group weight_map keys by layer ───────────────────────────
    layer_keys = {}     # layer_id -> [(suffix, full_key), ...]
    global_keys = []    # keys without layer prefix

    for key in weight_map:
        lid, suf = _strip_layer_prefix(key)
        if lid is not None:
            layer_keys.setdefault(lid, []).append((suf, key))
        else:
            global_keys.append(key)

    # ── Shard I/O helpers ────────────────────────────────────────────
    _shard_cache = {}   # shard_name -> (hdr, data_start)

    def _get_shard(shard_name):
        if shard_name not in _shard_cache:
            p = snapshot / shard_name
            _shard_cache[shard_name] = _read_safetensors_header(p)
        return _shard_cache[shard_name]

    def _load_tensor(key):
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        return _load_single_tensor_from_shard(snapshot / sn, hdr, ds, key)

    def _load_tensor_raw(key):
        """Return (raw_bytes, dtype, shape) without decoding."""
        sn = weight_map.get(key)
        if sn is None:
            return None, None, None
        hdr, ds = _get_shard(sn)
        entry = hdr.get(key)
        if entry is None:
            return None, None, None
        off0, off1 = entry["data_offsets"]
        p = snapshot / sn
        with open(p, "rb") as f:
            f.seek(ds + off0)
            raw = f.read(off1 - off0)
        return raw, entry["dtype"], tuple(entry["shape"])

    def _load_expert_weight(key):
        """Load an FP4 expert weight, dequantized to float32."""
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        return _dequant_fp4_from_shard(snapshot / sn, hdr, ds, key)

    def _load_expert_raw(key):
        """Load raw FP4 weight + e8m0 scale bytes from shard (no dequant).

        Returns (w_raw, w_shape, s_raw, s_shape) or None.
        """
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        entry = hdr.get(key)
        if entry is None:
            return None
        scale_key = key.replace(".weight", ".scale")
        scale_entry = hdr.get(scale_key)
        if scale_entry is None:
            return None
        w_off0, w_off1 = entry["data_offsets"]
        s_off0, s_off1 = scale_entry["data_offsets"]
        p = snapshot / sn
        with open(p, "rb") as f:
            f.seek(ds + w_off0)
            w_raw = f.read(w_off1 - w_off0)
            f.seek(ds + s_off0)
            s_raw = f.read(s_off1 - s_off0)
        return (w_raw, tuple(entry["shape"]),
                s_raw, tuple(scale_entry["shape"]))

    # ── Phase 1: Load global tensors ─────────────────────────────────
    print("\nPhase 1: Loading global tensors (embed, norm, lm_head)...")
    shared_entries = []  # (tid, layer_id, sub_id, qtype, tensor)

    # DeepSeek V4 Flash uses: embed.weight, norm.weight, head.weight
    # (NOT model.embed_tokens.weight / model.norm.weight / lm_head.weight)
    global_tids = [
        ("embed.weight",  TID_EMBED,      QTYPE_BF16),
        ("norm.weight",   TID_OUTPUT_NORM, QTYPE_BF16),
        ("head.weight",   TID_LM_HEAD,     QTYPE_BF16),
        # Hybrid Connection global output-collapse (final 4-stream -> plain).
        # HF keys have no ".weight" suffix; fn is [4, 4*hidden] F32 (engine transposes).
        ("hc_head_fn",    TID_HC_HEAD_FN,    QTYPE_BF16),
        ("hc_head_scale", TID_HC_HEAD_SCALE, QTYPE_F32),
        ("hc_head_base",  TID_HC_HEAD_BASE,  QTYPE_F32),
    ]

    lm_head_tensor = None
    for key, tid, qtype in global_tids:
        t = _load_tensor(key)
        if t is not None:
            shared_entries.append((tid, GLOBAL_LAYER, 0, qtype, t))
            print(f"  {key}: shape={list(t.shape)}, qtype={qtype}")
            if tid == TID_LM_HEAD:
                lm_head_tensor = t
        else:
            print(f"  {key}: NOT FOUND (may be tied to embedding)")

    if lm_head_tensor is None:
        embed_tensor = next((t for tid, _, _, _, t in shared_entries if tid == TID_EMBED), None)
        if embed_tensor is not None:
            print("  lm_head tied to embedding (reusing embed.weight)")
            shared_entries.append((TID_LM_HEAD, GLOBAL_LAYER, 0, QTYPE_BF16, embed_tensor))
        else:
            print("  WARNING: neither lm_head nor embed found")

    # ── Global config: per-layer compression ratios ────────────────────
    # DeepSeek V4 Flash compress_ratios: [0,0,4,128,4,128,...,4,0,0,0] (43).
    # Stored as a global F32 shared tensor (TID_COMPRESS_RATIOS) so the engine
    # can read it without a header-format change (mirrors the HC sidecar idea).
    ratios = hf_cfg.get("compress_ratios")
    if ratios is None:
        ratios = []
        for il in range(n_layers):
            ratios.append(0 if il < 2 else (4 if (il % 2 == 0) else 128))
    ratios = np.asarray(ratios[:n_layers], dtype=np.float32)
    shared_entries.append((TID_COMPRESS_RATIOS, GLOBAL_LAYER, 0, QTYPE_F32, ratios))
    print(f"  compress_ratios: n={len(ratios)} vals={list(map(int, ratios))}")

    _shard_cache.clear()
    gc.collect()

    # ── Phase 2: Load per-layer shared tensors ───────────────────────
    print(f"\nPhase 2: Loading shared tensors for {n_layers} layers...")

    for layer_id in _tqdm(range(n_layers), desc="Shared tensors"):
        for suffix, full_key in layer_keys.get(layer_id, []):
            if _is_expert_suffix(suffix) or _is_scale_suffix(suffix):
                continue

            info = _LAYER_SUFFIX_TIDS.get(suffix)
            if info is None:
                continue
            tid, sub_id, qtype, needs_dequant = info

            if needs_dequant:
                t = _load_expert_weight(full_key)
            else:
                t = _load_tensor(full_key)

            if t is not None:
                shared_entries.append((tid, layer_id, sub_id, qtype, t))

        if layer_id % 10 == 9:
            _shard_cache.clear()

    _shard_cache.clear()
    gc.collect()

    print(f"  Total shared entries: {len(shared_entries)}")

    # ── Phase 3: Write .fst file ─────────────────────────────────────
    print(f"\nPhase 3: Writing {output_path}...")
    t_write = time.time()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    shared_dir_offset = align_up(HEADER_SIZE)
    shared_dir_count = len(shared_entries)
    dir_size = shared_dir_count * SHARED_ENTRY_SIZE
    data_offset = align_up(shared_dir_offset + dir_size)

    with open(out, "wb") as f:
        # Reserve header + directory space
        f.write(b"\x00" * data_offset)

        # Write shared tensor data
        dir_entries = []
        for i, (tid, lid, sub_id, qtype, t) in enumerate(
                _tqdm(shared_entries, desc="Shared data")):
            shape = _shape_of(t)
            ndim = len(shape)
            shape3 = (shape + (1, 1, 1))[:3]
            data = _quant_shared(t, qtype)
            data_off = align_up(f.tell())
            f.seek(data_off)
            f.write(data)
            dir_entries.append(struct.pack(
                SHARED_ENTRY_FMT,
                tid, lid, sub_id, qtype, ndim, 0,
                int(shape3[0]), int(shape3[1]), int(shape3[2]),
                int(data_off), int(len(data)), 0,
            ))
            shared_entries[i] = None  # free reference

        shared_entries = None
        gc.collect()

        after_shared_data = align_up(f.tell())

        # Write directory at reserved offset (does NOT overlap with data)
        dir_blob = b"".join(dir_entries)
        f.seek(shared_dir_offset)
        f.write(dir_blob)
        del dir_entries, dir_blob

        # ── Phase 4: Write expert bank (streaming, one layer at a time) ──
        expert_bank_offset = after_shared_data
        print(f"\nPhase 4: Writing expert bank ({n_layers} x {n_experts} = "
              f"{n_layers * n_experts} blocks)...")

        experts_written = 0
        # Pre-build a map: eid -> {proj: full_key} for this layer
        for layer_id in _tqdm(range(n_layers), desc="Expert bank"):
            # Build expert key map for this layer
            exp_keys = {}  # eid -> {proj: full_key}
            for suffix, full_key in layer_keys.get(layer_id, []):
                if not _is_expert_suffix(suffix) or _is_scale_suffix(suffix):
                    continue
                eid, proj = _parse_expert_suffix(suffix)
                if eid is not None and proj is not None:
                    exp_keys.setdefault(eid, {})[proj] = full_key

            for eid in range(n_experts):
                keys = exp_keys.get(eid, {})
                gate_key = keys.get("gate_proj")
                up_key = keys.get("up_proj")
                down_key = keys.get("down_proj")

                if gate_key and up_key and down_key:
                    # Native MXFP4: pack raw HF FP4 bytes directly (no float roundtrip).
                    gr = _load_expert_raw(gate_key)
                    ur = _load_expert_raw(up_key)
                    dr = _load_expert_raw(down_key)
                    if gr and ur and dr:
                        block = pack_expert_raw_interleaved(gr, ur, dr)
                    else:
                        block = b"\x00" * EXPERT_BLOCK_BYTES
                else:
                    block = b"\x00" * EXPERT_BLOCK_BYTES

                assert len(block) == EXPERT_BLOCK_BYTES
                pos = expert_bank_offset + \
                    (layer_id * n_experts + eid) * EXPERT_BLOCK_STRIDE
                f.seek(pos)
                f.write(block)
                experts_written += 1
                del block

            del exp_keys
            _shard_cache.clear()
            gc.collect()

        # Pad file to page alignment
        f.seek(0, 2)
        end = align_up(f.tell())
        f.truncate(end)

        # Write final header
        f.seek(0)
        f.write(_pack_header(
            cfg, shared_dir_offset, shared_dir_count,
            expert_bank_offset, n_layers * n_experts,
        ))

    total_time = time.time() - t_start
    fsize = Path(output_path).stat().st_size
    print(f"\nConversion complete: {output_path}")
    print(f"  File size:        {fsize / 1e9:.2f} GB ({fsize:,} bytes)")
    print(f"  Shared entries:   {shared_dir_count}")
    print(f"  Expert blocks:    {experts_written} "
          f"({n_layers} layers x {n_experts} experts)")
    print(f"  Expert bank off:  {expert_bank_offset:#x}")
    print(f"  Wall time:        {total_time:.1f}s")

    return str(out)


# ─── HC sidecar (Hybrid Connection only) ───────────────────────────────


def convert_hc_only(model_dir, output_path, verbose=False, max_layers=None):
    """Write a tiny .fst.hc sidecar containing ONLY the Hybrid Connection
    tensors (per-layer TIDs 23-28 + global TIDs 29-31), ~70 MB.

    The main 150 GB .fst is left untouched: the engine auto-discovers a
    `<model>.fst.hc` next to the main .fst and layers the HC weights on top
    of the shared/expert weights it already loads.  This avoids the
    disk-critical full re-conversion (the HC weights are <0.05% of the model)
    while still shipping the 4-stream residual that fixes the garbage output.
    Layout is identical to a normal .fst (header -> shared dir -> shared data),
    with an empty expert bank (expert_count_total = 0).
    """
    t_start = time.time()
    model_dir = Path(model_dir)
    snapshot = next((model_dir / "snapshots").iterdir()) \
        if (model_dir / "snapshots").exists() else model_dir

    index_path = snapshot / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    config_path = snapshot / "config.json"
    hf_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            hf_cfg = json.load(f)
    n_layers = int(hf_cfg.get("num_hidden_layers", 43))
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)
    cfg = {
        "hidden_dim": hf_cfg.get("hidden_size", V4_HIDDEN),
        "num_layers": n_layers,
        "num_experts": hf_cfg.get("n_routed_experts", 256),
        "top_k": hf_cfg.get("num_experts_per_tok", 6),
        "vocab_size": hf_cfg.get("vocab_size", 129280),
    }
    print(f"HC sidecar: {n_layers} layers, hidden={cfg['hidden_dim']}")

    layer_keys = {}
    global_keys = []
    for key in weight_map:
        lid, suf = _strip_layer_prefix(key)
        if lid is not None:
            layer_keys.setdefault(lid, []).append((suf, key))
        else:
            global_keys.append(key)

    _shard_cache = {}
    def _get_shard(sn):
        if sn not in _shard_cache:
            _shard_cache[sn] = _read_safetensors_header(snapshot / sn)
        return _shard_cache[sn]
    def _load_tensor(key):
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        return _load_single_tensor_from_shard(snapshot / sn, hdr, ds, key)

    # HC suffixes we keep (must match _LAYER_SUFFIX_TIDS).
    HC_LAYER_SUFFIXES = {
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    }
    shared_entries = []  # (tid, lid, sub_id, qtype, tensor)

    # Global HC tensors (no ".weight" suffix in HF).
    for key, tid, qtype in [
        ("hc_head_fn",    TID_HC_HEAD_FN,    QTYPE_BF16),
        ("hc_head_scale", TID_HC_HEAD_SCALE, QTYPE_F32),
        ("hc_head_base",  TID_HC_HEAD_BASE,  QTYPE_F32),
    ]:
        t = _load_tensor(key)
        if t is not None:
            shared_entries.append((tid, GLOBAL_LAYER, 0, qtype, t))
            print(f"  {key}: shape={list(_shape_of(t))}")

    # Per-layer HC tensors.
    for layer_id in range(n_layers):
        for suffix, full_key in layer_keys.get(layer_id, []):
            if suffix not in HC_LAYER_SUFFIXES:
                continue
            info = _LAYER_SUFFIX_TIDS.get(suffix)
            if info is None:
                continue
            tid, sub_id, qtype, _needs_dequant = info
            t = _load_tensor(full_key)
            if t is not None:
                shared_entries.append((tid, layer_id, sub_id, qtype, t))
        if layer_id % 16 == 15:
            _shard_cache.clear()
    _shard_cache.clear()
    gc.collect()
    print(f"  Total HC entries: {len(shared_entries)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shared_dir_offset = align_up(HEADER_SIZE)
    shared_dir_count = len(shared_entries)
    data_offset = align_up(shared_dir_offset + shared_dir_count * SHARED_ENTRY_SIZE)

    with open(out, "wb") as f:
        f.write(b"\x00" * data_offset)
        dir_entries = []
        for i, (tid, lid, sub_id, qtype, t) in enumerate(
                _tqdm(shared_entries, desc="HC data")):
            shape = _shape_of(t)
            ndim = len(shape)
            shape3 = (shape + (1, 1, 1))[:3]
            data = _quant_shared(t, qtype)
            data_off = align_up(f.tell())
            f.seek(data_off)
            f.write(data)
            dir_entries.append(struct.pack(
                SHARED_ENTRY_FMT,
                tid, lid, sub_id, qtype, ndim, 0,
                int(shape3[0]), int(shape3[1]), int(shape3[2]),
                int(data_off), int(len(data)), 0,
            ))
            shared_entries[i] = None
        after_shared = align_up(f.tell())
        f.seek(shared_dir_offset)
        f.write(b"".join(dir_entries))
        f.seek(0, 2)
        f.truncate(align_up(f.tell()))
        f.seek(0)
        f.write(_pack_header(cfg, shared_dir_offset, shared_dir_count,
                             after_shared, 0))

    fsize = out.stat().st_size
    print(f"HC sidecar written: {output_path} ({fsize / 1e6:.1f} MB, "
          f"{shared_dir_count} entries, {time.time() - t_start:.1f}s)")
    return str(out)


# ─── Norm sidecar (correct RMSNorm weights + attn sinks) ───────────────


def convert_norm_only(model_dir, output_path, verbose=False, max_layers=None):
    """Write a tiny .fst.norm sidecar holding ONLY the RMSNorm weights and
    attention sinks (per-layer TIDs 3,4,14,15,22 + global TID 1), ~3 MB.

    The main 150 GB .fst was produced by an earlier converter that wrote
    incorrect (≈40x too small) norm values, which starves the FFN sublayer
    (moe_norm ~0.03 instead of ~1.6 → ffn_out ≈ 0 → garbage).  The engine
    auto-discovers <model>.fst.norm next to the main .fst and overrides the
    bad norms by TID (also sidestepping the dim-key collision that swapped
    attn_norm/moe_norm).  Other shared tensors (router, attention, experts)
    are correct in the main .fst and are NOT duplicated here.
    """
    t_start = time.time()
    model_dir = Path(model_dir)
    snapshot = next((model_dir / "snapshots").iterdir()) \
        if (model_dir / "snapshots").exists() else model_dir
    index_path = snapshot / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    config_path = snapshot / "config.json"
    hf_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            hf_cfg = json.load(f)
    n_layers = int(hf_cfg.get("num_hidden_layers", 43))
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)
    cfg = {
        "hidden_dim": hf_cfg.get("hidden_size", V4_HIDDEN),
        "num_layers": n_layers,
        "num_experts": hf_cfg.get("n_routed_experts", 256),
        "top_k": hf_cfg.get("num_experts_per_tok", 6),
        "vocab_size": hf_cfg.get("vocab_size", 129280),
    }
    print(f"Norm sidecar: {n_layers} layers")

    layer_keys = {}
    for key in weight_map:
        lid, suf = _strip_layer_prefix(key)
        if lid is not None:
            layer_keys.setdefault(lid, []).append((suf, key))

    _shard_cache = {}
    def _get_shard(sn):
        if sn not in _shard_cache:
            _shard_cache[sn] = _read_safetensors_header(snapshot / sn)
        return _shard_cache[sn]
    def _load_tensor(key):
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        return _load_single_tensor_from_shard(snapshot / sn, hdr, ds, key)

    NORM_SUFFIXES = {
        "attn_norm.weight", "input_layernorm.weight",
        "ffn_norm.weight", "post_attention_layernorm.weight",
        "attn.q_norm.weight", "attention.q_norm.weight",
        "attn.kv_norm.weight", "attention.kv_norm.weight",
        "attn.attn_sink",
    }
    shared_entries = []
    # Global final norm (TID 1).
    for key, tid, qtype in [("norm.weight", TID_OUTPUT_NORM, QTYPE_BF16)]:
        t = _load_tensor(key)
        if t is not None:
            shared_entries.append((tid, GLOBAL_LAYER, 0, qtype, t))
            print(f"  {key}: mean_abs={float(np.abs(_to_f32_np(t)).mean()):.4f}")

    for layer_id in range(n_layers):
        for suffix, full_key in layer_keys.get(layer_id, []):
            if suffix not in NORM_SUFFIXES:
                continue
            info = _LAYER_SUFFIX_TIDS.get(suffix)
            if info is None:
                continue
            tid, sub_id, qtype, _nd = info
            t = _load_tensor(full_key)
            if t is not None:
                shared_entries.append((tid, layer_id, sub_id, qtype, t))
        if layer_id % 16 == 15:
            _shard_cache.clear()
    _shard_cache.clear()
    gc.collect()
    print(f"  Total norm entries: {len(shared_entries)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shared_dir_offset = align_up(HEADER_SIZE)
    shared_dir_count = len(shared_entries)
    data_offset = align_up(shared_dir_offset + shared_dir_count * SHARED_ENTRY_SIZE)
    with open(out, "wb") as f:
        f.write(b"\x00" * data_offset)
        dir_entries = []
        for i, (tid, lid, sub_id, qtype, t) in enumerate(
                _tqdm(shared_entries, desc="Norm data")):
            shape = _shape_of(t)
            ndim = len(shape)
            shape3 = (shape + (1, 1, 1))[:3]
            data = _quant_shared(t, qtype)
            data_off = align_up(f.tell())
            f.seek(data_off)
            f.write(data)
            dir_entries.append(struct.pack(
                SHARED_ENTRY_FMT,
                tid, lid, sub_id, qtype, ndim, 0,
                int(shape3[0]), int(shape3[1]), int(shape3[2]),
                int(data_off), int(len(data)), 0,
            ))
            shared_entries[i] = None
        after_shared = align_up(f.tell())
        f.seek(shared_dir_offset)
        f.write(b"".join(dir_entries))
        f.seek(0, 2)
        f.truncate(align_up(f.tell()))
        f.seek(0)
        f.write(_pack_header(cfg, shared_dir_offset, shared_dir_count,
                             after_shared, 0))
    fsize = out.stat().st_size
    print(f"Norm sidecar written: {output_path} ({fsize / 1e6:.1f} MB, "
          f"{shared_dir_count} entries, {time.time() - t_start:.1f}s)")
    return str(out)


# ─── Verification ──────────────────────────────────────────────────────


def verify_fst(path):
    """Read back the .fst header and assert structural invariants.

    Checks:
      - Magic and version
      - num_experts_written == num_layers * num_experts
      - TID 0 (EMBED) is present in shared directory
      - TID 1 (OUTPUT_NORM) is present
      - TID 2 (LM_HEAD) is present
      - Expert bank is large enough for all declared expert blocks

    Raises AssertionError on failure, returns True on success.
    """
    p = Path(path)
    fsize = p.stat().st_size

    if fsize < HEADER_SIZE:
        raise AssertionError(f"File too small for header: {fsize} < {HEADER_SIZE}")

    with open(p, "rb") as fh:
        hdr_data = fh.read(HEADER_SIZE)
        hdr = struct.unpack_from(HEADER_FMT, hdr_data, 0)

        magic = hdr[0]
        version = hdr[1]

        assert magic == FST_MAGIC, f"Bad magic: {magic!r}"
        assert version == FST_VERSION, f"Bad version: {version}"

        hidden_dim    = hdr[2]
        num_layers    = hdr[3]
        num_experts   = hdr[4]
        top_k         = hdr[5]
        vocab_size    = hdr[11]
        shared_dir_off = hdr[15]
        shared_dir_cnt = hdr[16]
        expert_bank_off = hdr[17]
        expert_block_bytes = hdr[18]
        expert_block_stride = hdr[19]
        expert_count_total = hdr[20]

        expected_experts = num_layers * num_experts
        assert expert_count_total == expected_experts, (
            f"expert_count_total={expert_count_total} != "
            f"num_layers({num_layers}) * num_experts({num_experts}) = {expected_experts}")

        print(f"  Header OK: {num_layers} layers, {num_experts} experts, "
              f"hidden={hidden_dim}, vocab={vocab_size}")

        # Scan shared directory for TIDs (stream from file)
        tid_set = set()
        tid_counts = {}
        if shared_dir_cnt > 0:
            dir_end = shared_dir_off + shared_dir_cnt * SHARED_ENTRY_SIZE
            assert dir_end <= fsize, (
                f"Directory extends past EOF: {dir_end} > {fsize}")

            fh.seek(shared_dir_off)
            dir_data = fh.read(shared_dir_cnt * SHARED_ENTRY_SIZE)
            for i in range(shared_dir_cnt):
                off = i * SHARED_ENTRY_SIZE
                entry = struct.unpack_from(SHARED_ENTRY_FMT, dir_data, off)
                tid = entry[0]
                tid_set.add(tid)
                tid_counts[tid] = tid_counts.get(tid, 0) + 1

    tid_names = {
        0: "EMBED", 1: "OUTPUT_NORM", 2: "LM_HEAD",
        3: "INPUT_NORM", 4: "POST_ATTN_NORM",
        9: "ROUTER", 11: "SHARED_GATE", 12: "SHARED_UP", 13: "SHARED_DOWN",
        16: "MARKOV_W1", 17: "MARKOV_W2", 18: "CONFIDENCE_PROJ",
        19: "MAIN_PROJ", 20: "MAIN_NORM", 21: "DRAFT_NORM", 22: "ATTN_SINK",
        23: "HC_ATTN_FN", 24: "HC_ATTN_SCALE", 25: "HC_ATTN_BASE",
        26: "HC_FFN_FN", 27: "HC_FFN_SCALE", 28: "HC_FFN_BASE",
        29: "HC_HEAD_FN", 30: "HC_HEAD_SCALE", 31: "HC_HEAD_BASE",
        32: "CMP_APE", 33: "CMP_WKV", 34: "CMP_GATE", 35: "CMP_NORM",
        40: "COMPRESS_RATIOS",
    }

    print(f"  Shared directory: {shared_dir_cnt} entries")
    for tid in sorted(tid_counts):
        name = tid_names.get(tid, f"TID_{tid}")
        print(f"    TID {tid:2d} ({name:16s}): {tid_counts[tid]:5d} tensors")

    assert 0 in tid_set, "TID 0 (EMBED) MISSING from shared directory"
    assert 2 in tid_set, "TID 2 (LM_HEAD) MISSING from shared directory"

    # Expert bank size check
    if expert_bank_off > 0 and expert_bank_off < fsize:
        bank_size = fsize - expert_bank_off
        expected_bank = expert_count_total * expert_block_stride
        actual_blocks = bank_size // expert_block_stride
        print(f"  Expert bank: offset={expert_bank_off:#x}, "
              f"size={bank_size / 1e9:.2f} GB, "
              f"blocks={actual_blocks}/{expert_count_total}")
        assert actual_blocks >= expert_count_total, (
            f"Expert bank too small: room for {actual_blocks} blocks, "
            f"need {expert_count_total}")

    print("\n  VERIFICATION PASSED")
    return True


# ─── DSpark Draft Model Converter ──────────────────────────────────────


def convert_dspark_draft(model_dir, output_path, verbose=False, max_stages=None):
    """Convert the DSpark draft model (mtp.0, mtp.1, mtp.2) to a .fst file.

    The draft model is a 3-stage transformer with the same expert geometry
    as the main model (256 experts, 20-byte aligned DS4 dense packing).
    Stage 2 also contains the Markov head and confidence head for
    confidence-scheduled speculative decoding.

    Layout: identical to main .fst — header, shared directory, shared data,
    expert bank — but n_layers=3.
    """
    t_start = time.time()
    model_dir = Path(model_dir)

    if (model_dir / "snapshots").exists():
        snapshot = next((model_dir / "snapshots").iterdir())
    else:
        snapshot = model_dir

    index_path = snapshot / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    config_path = snapshot / "config.json"
    hf_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            hf_cfg = json.load(f)

    n_mtp_layers = hf_cfg.get("num_nextn_predict_layers",
                               hf_cfg.get("n_mtp_layers", 3))
    n_experts = hf_cfg.get("n_routed_experts", 256)
    top_k = hf_cfg.get("num_experts_per_tok", 6)

    # Auto-detect MTP stages from weight map if config is wrong
    mtp_stages_found = set()
    for key in weight_map:
        if key.startswith("mtp."):
            parts = key.split(".")
            if len(parts) > 1:
                try:
                    mtp_stages_found.add(int(parts[1]))
                except ValueError:
                    pass
    if mtp_stages_found:
        n_mtp_layers = max(n_mtp_layers, max(mtp_stages_found) + 1)
    if max_stages is not None:
        n_mtp_layers = min(n_mtp_layers, max_stages)

    cfg = {
        "hidden_dim": hf_cfg.get("hidden_size", V4_HIDDEN),
        "num_layers": n_mtp_layers,
        "num_experts": n_experts,
        "expert_inter_dim": hf_cfg.get("moe_intermediate_size", V4_INTERMEDIATE),
        "top_k": top_k,
        "num_q_heads": hf_cfg.get("num_attention_heads", 64),
        "num_kv_heads": hf_cfg.get("num_key_value_heads", 1),
        "head_dim": hf_cfg.get("head_dim", 512),
        "vocab_size": hf_cfg.get("vocab_size", 129280),
        "rms_eps": hf_cfg.get("rms_norm_eps", 1e-6),
        "rope_freq_base": hf_cfg.get("rope_theta", 10000.0),
        "dspark_block_size": hf_cfg.get("dspark_block_size", 5),
        "dspark_markov_rank": hf_cfg.get("dspark_markov_rank", 256),
        "dspark_noise_token_id": hf_cfg.get("dspark_noise_token_id", 128799),
    }

    print(f"DSpark Draft: {n_mtp_layers} stages, {n_experts} experts/stage, "
          f"hidden={cfg['hidden_dim']}, vocab={cfg['vocab_size']}")

    # ── Pre-group MTP weight_map keys by stage ────────────────────────
    mtp_keys = {}
    for key in weight_map:
        sid, suf = _strip_mtp_prefix(key)
        if sid is not None:
            mtp_keys.setdefault(sid, []).append((suf, key))

    # ── Shard I/O helpers ─────────────────────────────────────────────
    _shard_cache = {}

    def _get_shard(shard_name):
        if shard_name not in _shard_cache:
            _shard_cache[shard_name] = _read_safetensors_header(snapshot / shard_name)
        return _shard_cache[shard_name]

    def _load_tensor(key):
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        return _load_single_tensor_from_shard(snapshot / sn, hdr, ds, key)

    def _load_expert_weight(key):
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        return _dequant_fp4_from_shard(snapshot / sn, hdr, ds, key)

    def _load_expert_raw(key):
        sn = weight_map.get(key)
        if sn is None:
            return None
        hdr, ds = _get_shard(sn)
        entry = hdr.get(key)
        if entry is None:
            return None
        scale_key = key.replace(".weight", ".scale")
        scale_entry = hdr.get(scale_key)
        if scale_entry is None:
            return None
        w_off0, w_off1 = entry["data_offsets"]
        s_off0, s_off1 = scale_entry["data_offsets"]
        p = snapshot / sn
        with open(p, "rb") as f:
            f.seek(ds + w_off0)
            w_raw = f.read(w_off1 - w_off0)
            f.seek(ds + s_off0)
            s_raw = f.read(s_off1 - s_off0)
        return (w_raw, tuple(entry["shape"]),
                s_raw, tuple(scale_entry["shape"]))

    # ── Phase 1: Load global tensors (embed, norm, head shared with main) ─
    print("\nDraft Phase 1: Loading global tensors...")
    shared_entries = []

    # NOTE: top-level "norm.weight" is the MAIN model's final norm (shard 45),
    # NOT the draft's own final norm.  The draft's final norm is the LAST MTP
    # stage's "norm.weight" (mtp.2.norm.weight, shard 48), emitted per-stage as
    # TID_DRAFT_NORM at lid=last by the Phase-2 loop below.  Loading the main
    # norm here as a GLOBAL TID_DRAFT_NORM @ lid=65535 would shadow the draft's
    # own norm (the engine's pop prefers lid==DRAFT_GLOBAL_LAYER) and apply the
    # wrong norm to draft hidden states -- so it is intentionally NOT loaded.
    global_tids = [
        ("embed.weight",   TID_EMBED,       QTYPE_BF16),
        ("head.weight",     TID_LM_HEAD,     QTYPE_BF16),
    ]
    for key, tid, qtype in global_tids:
        t = _load_tensor(key)
        if t is not None:
            shared_entries.append((tid, GLOBAL_LAYER, 0, qtype, t))
            print(f"  {key}: shape={list(t.shape)}, qtype={qtype}")
        else:
            print(f"  {key}: NOT FOUND")

    # ── Phase 2: Load per-stage shared tensors ────────────────────────
    print(f"\nDraft Phase 2: Loading shared tensors for {n_mtp_layers} stages...")
    for stage_id in _tqdm(range(n_mtp_layers), desc="Draft shared"):
        for suffix, full_key in mtp_keys.get(stage_id, []):
            if _is_expert_suffix(suffix) or _is_scale_suffix(suffix):
                continue

            info = _LAYER_SUFFIX_TIDS.get(suffix)
            if info is None:
                continue
            tid, sub_id, qtype, needs_dequant = info

            if needs_dequant:
                t = _load_expert_weight(full_key)
            else:
                t = _load_tensor(full_key)

            if t is not None:
                shared_entries.append((tid, stage_id, sub_id, qtype, t))

        _shard_cache.clear()
        gc.collect()

    print(f"  Total shared entries: {len(shared_entries)}")

    # ── Phase 3: Write .fst file ─────────────────────────────────────
    print(f"\nDraft Phase 3: Writing {output_path}...")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    shared_dir_offset = align_up(HEADER_SIZE)
    shared_dir_count = len(shared_entries)
    dir_size = shared_dir_count * SHARED_ENTRY_SIZE
    data_offset = align_up(shared_dir_offset + dir_size)

    with open(out, "wb") as f:
        f.write(b"\x00" * data_offset)

        dir_entries = []
        for i, (tid, lid, sub_id, qtype, t) in enumerate(
                _tqdm(shared_entries, desc="Draft shared data")):
            shape = _shape_of(t)
            ndim = len(shape)
            shape3 = (shape + (1, 1, 1))[:3]
            data = _quant_shared(t, qtype)
            data_off = align_up(f.tell())
            f.seek(data_off)
            f.write(data)
            dir_entries.append(struct.pack(
                SHARED_ENTRY_FMT,
                tid, lid, sub_id, qtype, ndim, 0,
                int(shape3[0]), int(shape3[1]), int(shape3[2]),
                int(data_off), int(len(data)), 0,
            ))
            shared_entries[i] = None

        shared_entries = None
        gc.collect()

        after_shared_data = align_up(f.tell())
        dir_blob = b"".join(dir_entries)
        f.seek(shared_dir_offset)
        f.write(dir_blob)
        del dir_entries, dir_blob

        # ── Phase 4: Write expert bank ───────────────────────────────
        expert_bank_offset = after_shared_data
        print(f"\nDraft Phase 4: Writing expert bank ({n_mtp_layers} x {n_experts} = "
              f"{n_mtp_layers * n_experts} blocks)...")

        experts_written = 0
        for stage_id in _tqdm(range(n_mtp_layers), desc="Draft expert bank"):
            # Build expert key map for this stage
            exp_keys = {}
            for suffix, full_key in mtp_keys.get(stage_id, []):
                if not _is_expert_suffix(suffix) or _is_scale_suffix(suffix):
                    continue
                eid, proj = _parse_expert_suffix(suffix)
                if eid is not None and proj is not None:
                    exp_keys.setdefault(eid, {})[proj] = full_key

            for eid in range(n_experts):
                keys = exp_keys.get(eid, {})
                gate_key = keys.get("gate_proj")
                up_key = keys.get("up_proj")
                down_key = keys.get("down_proj")

                if gate_key and up_key and down_key:
                    # Native MXFP4: pack raw HF FP4 bytes directly (no float roundtrip).
                    gr = _load_expert_raw(gate_key)
                    ur = _load_expert_raw(up_key)
                    dr = _load_expert_raw(down_key)
                    if gr and ur and dr:
                        block = pack_expert_raw_interleaved(gr, ur, dr)
                    else:
                        block = b"\x00" * EXPERT_BLOCK_BYTES
                else:
                    block = b"\x00" * EXPERT_BLOCK_BYTES

                assert len(block) == EXPERT_BLOCK_BYTES
                pos = expert_bank_offset + \
                    (stage_id * n_experts + eid) * EXPERT_BLOCK_STRIDE
                f.seek(pos)
                f.write(block)
                experts_written += 1
                del block

            del exp_keys
            _shard_cache.clear()
            gc.collect()

        f.seek(0, 2)
        end = align_up(f.tell())
        f.truncate(end)

        f.seek(0)
        f.write(_pack_header(
            cfg, shared_dir_offset, shared_dir_count,
            expert_bank_offset, n_mtp_layers * n_experts,
        ))

    total_time = time.time() - t_start
    fsize = Path(output_path).stat().st_size
    print(f"\nDraft conversion complete: {output_path}")
    print(f"  File size:        {fsize / 1e9:.2f} GB ({fsize:,} bytes)")
    print(f"  Shared entries:   {shared_dir_count}")
    print(f"  Expert blocks:    {experts_written} "
          f"({n_mtp_layers} stages x {n_experts} experts)")
    print(f"  Wall time:        {total_time:.1f}s")

    return str(out)


# ─── Legacy / demo ─────────────────────────────────────────────────────


def _demo():
    print("V4 FP4 dequant self-test:")
    w = np.zeros((1, 16), dtype=np.int8)
    w[0, 0] = 0x12
    w[0, 1] = 0x34
    s = np.array([[120]], dtype=np.uint8)
    d = dequant_v4_fp4(w, s)
    print(f"  weight[0,:4]={w[0,:4].tolist()} scale={s[0].tolist()} -> {d[0,:4].tolist()}")
    assert d.shape == (1, 32), f"expected (1,32), got {d.shape}"
    print("  OK")


if __name__ == "__main__":
    import sys
    usage = (
        "Usage:\n"
        "  python3 fst_converter.py                       # run FP4 dequant self-test\n"
        "  python3 fst_converter.py v4 <model_dir> [layer] [output]  # single layer\n"
        "  python3 fst_converter.py full <model_dir> [output]        # full model\n"
        "  python3 fst_converter.py dspark <model_dir> [main_out] [draft_out]  # DSpark\n"
        "  python3 fst_converter.py draft <model_dir> [output]       # draft only\n"
        "  python3 fst_converter.py verify <fst_file>               # verify .fst\n"
    )

    if len(sys.argv) < 2:
        _demo()
        print(usage)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "v4":
        model_dir = sys.argv[2] if len(sys.argv) > 2 else \
            "Source/models--deepseek-ai--DeepSeek-V4-Flash"
        layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        out = sys.argv[4] if len(sys.argv) > 4 else f"deepseek_layer{layer}.fst"
        print(f"=== Converting DeepSeek V4 Flash layer {layer} -> {out} ===")
        convert_v4_layer(model_dir, layer, out)

    elif cmd == "full":
        model_dir = sys.argv[2] if len(sys.argv) > 2 else \
            "Source/models--deepseek-ai--DeepSeek-V4-Flash"
        out = sys.argv[3] if len(sys.argv) > 3 else "deepseek_v4_full.fst"
        max_layers = int(sys.argv[4]) if len(sys.argv) > 4 else None
        print(f"=== Converting full DeepSeek V4 Flash -> {out} ===")
        convert_full_model_v4(model_dir, out, verbose=True, max_layers=max_layers)
        print(f"\n=== Verifying {out} ===")
        verify_fst(out)

    elif cmd == "dspark":
        model_dir = sys.argv[2] if len(sys.argv) > 2 else \
            "Source/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
        main_out = sys.argv[3] if len(sys.argv) > 3 else "deepseek_v4_dspark.fst"
        draft_out = sys.argv[4] if len(sys.argv) > 4 else "dspark_draft.fst"
        if not Path(main_out).exists():
            print(f"=== Converting DSpark Main Model -> {main_out} ===")
            convert_full_model_v4(model_dir, main_out, verbose=True)
        else:
            print(f"=== {main_out} already exists, skipping main conversion ===")
        print(f"\n=== Converting DSpark Draft Model -> {draft_out} ===")
        convert_dspark_draft(model_dir, draft_out)

    elif cmd == "draft":
        model_dir = sys.argv[2] if len(sys.argv) > 2 else \
            "Source/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
        draft_out = sys.argv[3] if len(sys.argv) > 3 else "dspark_draft.fst"
        max_stages = int(sys.argv[4]) if len(sys.argv) > 4 else None
        print(f"=== Converting DSpark Draft Model -> {draft_out} ===")
        convert_dspark_draft(model_dir, draft_out, max_stages=max_stages)

    elif cmd == "hc":
        # Emit ONLY the Hybrid Connection sidecar (~70 MB) next to an existing
        # main .fst.  No expert/shared rewrite, no disk risk.  The engine
        # auto-loads <main>.fst.hc.  Usage: hc <model_dir> <main.fst> [max_layers]
        model_dir = sys.argv[2] if len(sys.argv) > 2 else \
            "Source/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
        main_fst = sys.argv[3] if len(sys.argv) > 3 else "deepseek_v4_dspark.fst"
        max_layers = int(sys.argv[4]) if len(sys.argv) > 4 else None
        out = main_fst + ".hc"
        print(f"=== Writing HC sidecar -> {out} (main .fst untouched) ===")
        convert_hc_only(model_dir, out, verbose=True, max_layers=max_layers)

    elif cmd == "norm":
        # Emit ONLY the RMSNorm + attn-sink sidecar (~3 MB) with CORRECT
        # values, overriding the bad norms baked into a stale main .fst.
        # Usage: norm <model_dir> <main.fst> [max_layers]
        model_dir = sys.argv[2] if len(sys.argv) > 2 else \
            "Source/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
        main_fst = sys.argv[3] if len(sys.argv) > 3 else "deepseek_v4_dspark.fst"
        max_layers = int(sys.argv[4]) if len(sys.argv) > 4 else None
        out = main_fst + ".norm"
        print(f"=== Writing norm sidecar -> {out} (main .fst untouched) ===")
        convert_norm_only(model_dir, out, verbose=True, max_layers=max_layers)

    elif cmd == "verify":
        path = sys.argv[2] if len(sys.argv) > 2 else "deepseek_v4_full.fst"
        print(f"=== Verifying {path} ===")
        verify_fst(path)

    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
