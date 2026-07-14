# HY3_PLAN — C++ engine changes to run Tencent Hunyuan-3.0 (HY3) on FaStar

> **Status (2026-07-09):** Architecture extracted from the live GGUF metadata (step 1 ✅).
> Python converter extended with a `hy_v3` arch path and unit-tested (step 2 ✅, code-complete,
> conversion queued behind the 145 GB download — step 3 ⏳). **This document is the step-4
> deliverable: the concrete C++ engine-change plan.** No engine code has been changed yet.

## 0. TL;DR

HY3 (`hy_v3`, ~298.8 B params / ~21 B active) is **GQA + sigmoid router + NextN MTP**, the
opposite attention family from DeepSeek V4 Flash's MLA. The plan rewrites the per-layer path in
`fst_engine.cpp` behind an arch branch and **adds zero mandatory NPU kernels**: HY3 reuses the
proven `expert_gemm_vec` / `expert_gemm_down` / `ew_unified` / `router_gemm` / `lm_head` xclbins
unchanged (expert geometry is byte-identical to the existing MXFP4 layout, just a different block
size read from the header). HY3 **drops `mla_unified`** (GQA does not need latent compression) →
the engine falls from ~5–6 hw_contexts to ~4, **freeing ≥3 of the 9 hw_context cap** — exactly the
headroom the deferred fused-FFN xclbin (the only proven >1 tok/s lever) needs.

## 1. Validated HY3 architecture (from the GGUF, not config.json)

Parsed offline from `hy3-1M-MTP-Q3_K_M.gguf` metadata (`scripts/hy3_parse_header.py`). Values
used by the converter are read straight from GGUF KV; the constants below are the inspected
defaults.

| Hyperparameter | HY3 value | DeepSeek V4 Flash (current) |
|---|---|---|
| `general.architecture` | `hy_v3` | `deepseek_v3` |
| `block_count` (`num_hidden_layers` + MTP) | **81** (blk.0..80) | 43 |
| `embedding_length` (hidden) | 4096 | 4096 |
| `attention.head_count` (Q heads) | **64** | MLA (Q latent 1024) |
| `attention.head_count_kv` (KV heads) | **8** (GQA 8:1) | MLA (KV latent 512) |
| `attention.key_length` (head_dim) | **128** | — |
| Attention | **GQA + per-head Q/K RMSNorm** (`qk_norm: true`) | MLA latent compression |
| `expert_count` | **192** | 1376 |
| `expert_used_count` (top-k) | **8** | 64 routed |
| shared experts | **1, always-on, ungated** | shared experts |
| `expert_feed_forward_length` (expert inter) | **1536** | MXFP4 17-byte blocks |
| dense FFN inter (layer 0, `feed_forward_length`) | **13312** | — |
| `first_k_dense_replace` | **1** (L0 dense, L1..79 MoE) | — |
| Router | **sigmoid + per-expert bias** (`expert_gating_func=2`), bias added **only to top-k selection**; weight = unbiased sigmoid; **route_norm ÷Σ then × `expert_weights_scale` 2.826** | sqrtsoftplus + bias + hash routing L0–L2 |
| Activation / FFN | SiLU, parallel gate‖up → down | SiLU |
| Norm | RMSNorm, `layer_norm_rms_epsilon` 1e-5, everywhere | RMSNorm |
| RoPE | NeoX rotate-half, `rope.freq_base` 11158840, **YaRN** (`rope.scaling.factor` 4.0, orig 262144) → 1 M ctx | RoPE on latents |
| `vocab_size` (`token_embd.weight` shape) | **120832** | 128000 |
| MTP | `blk.80` NextN, `nextn_predict_layers=1` (shares trunk embd + lm_head) | DSpark separate draft model |
| Source quant | **Q3_K_M** (mixed K-quants) | MXFP4 / FP8 |

**Expert geometry (validated, byte-exact):** gate/up `[out=1536, in=4096]` and down
`[out=4096, in=1536]`. Each projection = `(1536×4096/32)×17 = 3,342,336 B`. One expert block =
gate+up+down = **10,027,008 B**, and `10,027,008 = 2448 × 4096` → exactly page-aligned, so
`expert_block_stride == expert_block_bytes` (no padding). This is the **same 17-byte MXFP4 dense
layout** the existing dequant+GEMM kernel consumes — only the block size differs, and that comes
from the header.

## 2. `.fst` layout the converter produces

`convert_hy3` (scripts/fst_converter.py:1642) writes a standard `.fst`: 128-byte `FSTH` header →
64-byte shared-bank directory entries → BF16/F32 shared tensors → fixed-stride MXFP4 expert bank.

**Header fields (`HEADER_FMT="<4sIIIIIIIIIIIQffQQQQQQQQ"`, 128 B):**

| field | HY3 value | note |
|---|---|---|
| `m`/`v` | `"FST\0"` / 2 | unchanged container |
| `hd` | 4096 | hidden |
| `nl` | 81 | block_count (incl. MTP blk.80) |
| `ne` | 192 | experts per MoE layer |
| `tk` | 8 | top-k |
| `qh` | 64 | Q heads |
| `kh` | 8 | KV heads |
| `hd2` | 128 | head_dim |
| `id` | 1536 | expert inter dim |
| `ns` | 1 | shared experts |
| `r0` | 0 | reserved |
| `vs` | 120832 | vocab |
| `re` | 1e-5 | rms_eps |
| `rf` | 11158840.0 | rope freq base |
| `eb` | 10,027,008 | expert_block_bytes (header-driven, read at fst_engine.cpp:1512) |
| `es` | 10,027,008 | expert_block_stride (page-aligned → == eb) |
| `et` | 15552 | expert_count_total = `nl × ne` = 81×192 |
| `r1` | **0x20001** | low16 `first_k_dense_replace=1`, high16 `expert_gating_func=2` |
| `r2` | **0xFA00002B1F10** | high32 `int(yarn_factor×1000)=4000`, low32 `int(ew_scale×1e6)=2826000` |

HY3-specific constants not in the standard header are packed into the reserved `r1`/`r2` u64 slots
(no struct change): `first_k_dense_replace`, `expert_gating_func`, YaRN `rope.scaling.factor`,
`expert_weights_scale`. The engine HY3 path unpacks these.

**Tensor TIDs** (shared bank; all matrices BF16 `[out,in]`, router + bias F32):
- Reused unchanged: `TID_EMBED`, `TID_OUTPUT_NORM`, `TID_LM_HEAD`, `TID_INPUT_NORM` (attn_norm),
  `TID_Q_PROJ`/`TID_K_PROJ`/`TID_V_PROJ`/`TID_O_PROJ`, `TID_POST_ATTN_NORM` (ffn_norm),
  `TID_ROUTER` (ffn_gate_inp, F32), `TID_ROUTER_BIAS` (exp_probs_b, F32), `TID_SHARED_GATE`/`UP`/`DOWN` (shexp).
- New HY3 TIDs (converter:171–179): `TID_HY3_Q_NORM=50`, `TID_HY3_K_NORM=51` (per-head `[128]`),
  `TID_HY3_DENSE_GATE=52`/`DENSE_UP=53`/`DENSE_DOWN=54` (L0 dense FFN, inter 13312),
  `TID_HY3_NEXTN_EH_PROJ=55` (`[2*hidden, hidden]`), `TID_HY3_NEXTN_ENORM=56`/`HNORM=57`/`SHARED_HEAD_N=58`.

**Expert bank:** layers 1..80 × 192 experts each at stride 10,027,008 B
(`pos = eo + (L*192 + e)*es`). Layer 0 is dense → its 192 bank slots are left sparse (never read).
Each block is the existing gate+up+down MXFP4 triple — the engine fetches it via `get_expert_bo`
and runs the existing kernel.

## 3. Engine changes (`src/fst_engine.cpp` + `include/fst_engine.h`)

### 3.1 `ModelConfig` + arch enum (`fst_engine.h:127`)
Add an `enum Arch { ARCH_DS4, ARCH_HY3 }` and HY3 fields:
`num_q_heads, num_kv_heads, head_dim, expert_inter_dim, first_k_dense_replace, expert_gating_func,
expert_weights_scale, rope_scaling_factor, yarn_orig_ctx`. The loader sets `arch=ARCH_HY3` when it
detects `qh=64 && kh=8 && hd2=128` (or a magic in `r1`), and unpacks `r1`/`r2` into the new fields.
The DS4 path is untouched (gated on `arch`).

### 3.2 `Hy3LayerWeights` + arch-branched loader
Add a per-layer struct holding the GQA tensors at their **actual stored dims** — q `[8192,4096]`,
k/v `[1024,4096]`, o `[8192,8192]`, q_norm/k_norm `[128]`, router `[192,4096]` (F32), bias `[192]`
(F32), shared gate/up `[1536,4096]` + down `[4096,1536]`, dense gate/up `[13312,4096]` + down
`[4096,13312]` (L0 only), nextn `[4096,8192]` + 3 norms (L80 only). **Critical: do not hardcode the
MLA dims** (the current loader hardcodes `load_bf16_vec(14,…,1024)` at fst_engine.cpp:1917). Read
dims from each directory entry's `shape` fields — the converter already stores them. The expert
loader already reads `eb`/`es` from the header (fst_engine.cpp:1512,1296), so paging needs no
change beyond using the HY3 block size.

### 3.3 GQA attention (replaces `process_mla`, fst_engine.cpp:3320)
New `process_gqa(layer, h)`:
1. RMSNorm(h, input_norm) — reuse `ew_unified`.
2. q = h·Wq, k = h·Wk, v = h·Wv — reuse `expert_gemm_vec` (M=seq, K=4096, N=8192/1024). k/v produce
   the GQA KV row (8 heads × 128).
3. **Per-head Q/K RMSNorm** — reshape q→`[seq,64,128]`, k→`[seq,8,128]`, RMSNorm each head over 128
   with its own `[128]` weight — reuse `ew_unified` (RMSNorm op already there; per-head is a
   reshape). V has no norm.
4. YaRN RoPE on q/k (64/8 heads × 128, base 11158840, factor 4.0, orig 262144) — **new host+EW
   code** (rotate-half; YaRN curve, not plain RoPE). Optional: fold into `ew_unified`.
5. Append k/v to the **uncompressed** GQA KV cache (`max_seq × 8 × 128 × 2 (K+V) × bf16` per layer).
6. Scaled-dot-product attention (8 KV heads broadcast to 64 Q heads) → softmax — reuse `ew_unified`.
7. o = attn_out·Wo — reuse `expert_gemm_vec` (N=8192). Add to residual.
**0 new xclbins.** The MLA latent split, `wq_a` compression, KV-compressor, and `mla_unified` are
all skipped on the HY3 path.

### 3.4 Sigmoid+bias router (replaces `npu_router`, fst_engine.cpp:936)
New `npu_router_hy3(layer, h)`:
1. logits = h·Wrouter — reuse `router_gemm` (M=seq, K=4096, N=192). (Same GEMM as DS4; the
   sqrtsoftplus post-processing is what changes.)
2. `s = sigmoid(logits)` per expert.
3. **Top-8 selection on `s + exp_probs_b`** (bias added to the *selection score only*, not to the
   weight — HY3 gotcha #1).
4. Selected weights = **unbiased** `s` (without bias) for the 8 winners; `route_norm` = weight ÷ Σ;
   then × `expert_weights_scale` 2.826 (HY3 gotcha #2).
5. **No hash routing** (DS4's L0–L2 hash trick is DS4-specific; drop it on the HY3 path).
Optional: a dedicated sigmoid xclbin, or do the sigmoid/topk on CPU (router is 192-wide, small).

### 3.5 Dense L0 + always-on shared expert
- **Dense L0 switch:** in `process_layer` (fst_engine.cpp:2584), if `layer < first_k_dense_replace`
  (i.e. `lid==0`), run a dense SwiGLU FFN with `TID_HY3_DENSE_GATE/UP/DOWN` (inter 13312) instead of
  the MoE block — reuse `expert_gemm_vec`/`expert_gemm_down` on the BF16 dense weights, SiLU via
  `ew_unified`. No router, no expert dispatch.
- **Always-on shared expert:** for MoE layers (lid≥1), after the 8 routed experts, run the shared
  expert (`TID_SHARED_*`, inter 1536, ungated) via `process_shared_expert` (fst_engine.cpp:4373,
  reused) and add: `ffn_out = Σ wi·expert_i(h) + shared_expert(h)`. HY3 gotcha #4.

### 3.6 NextN MTP head (replaces DSpark, fst_engine.cpp:4534/5493)
Replace `load_draft_model`/`forward_draft` with a NextN head that **shares the trunk's embedding +
lm_head** (no second `.fst`, no second pager):
- Inputs: last token id → shared `token_embd`; previous trunk post-`output_norm` hidden `h`.
- `enorm(e)`, `hnorm(h)` → `concat(e_norm, h_norm)` → `eh_proj {2*hidden → hidden}` (TID 55/56/57).
- One full GQA+MoE decoder block (blk.80) → `shared_head_norm` (TID 58) → shared `lm_head` → draft
  token. **Return post-`shared_head_norm` hidden as the next step's `h`** (post-norm chaining — HY3
  gotcha #6; chaining raw residuals gives wrong drafts).
- **Iterative single-token** drafting up to `n_max=3` with `p_min` cutoff, matching the llama.cpp
  `hy3-mtp` fork's `common_speculative_state_mtp::draft()`. The MTP layer keeps its own KV
  (separate from the trunk's 80-layer KV). Structurally simpler than DSpark (no second router/KV
  model); the trunk `lm_head` is the verify head.

### 3.7 KV cache
Uncompressed GQA KV: `max_seq × 8 × 128 × 2 × bf16` per layer (vs DS4's compressed 512 latent). The
per-token decode KV update is one row either way, so the memory delta matters at prefill, not
decode.

## 4. hw_context budget (the real prize)

MAX_HW_CONTEXTS=9 (fst_aiebu_cache.hpp:120). DS4 today uses ~5–6: `expert_gemm_vec`,
`expert_gemm_down`, `dequant_v4`/`ffn_unified`, `mla_unified`, `ew_unified` (+ `router_gemm`).
HY3 **drops `mla_unified`** (GQA reuses `expert_gemm_vec` + `ew_unified`) and reuses everything
else → **~4 contexts, ≥3 freed**. That headroom is exactly what the deferred fused-FFN xclbin needs
(see memory: fusion was blocked by the 9-cap under MLA). HY3 is the better vehicle to reach >1 tok/s
because it removes the MLA context pressure that currently blocks adding a fused kernel — without
retiring any working kernel.

## 5. Reused unchanged / new kernels

| Kernel | HY3 use | Status |
|---|---|---|
| `expert_gemm_vec` / `expert_gemm_down` (or `ffn_unified`) | expert + dense + shared FFN GEMMs, q/k/v/o proj | **REUSE** (recompile shapes; design identical) |
| `ew_unified` | RMSNorm, SiLU, per-head Q/K norm, softmax | **REUSE** |
| `router_gemm` | router GEMM (192-wide) | **REUSE** (post-processing changes, host-side) |
| `lm_head` | final + MTP shared head | **REUSE** (recompile for vocab 120832) |
| MXFP4 dequant | expert dequant | **REUSE** (block size from header) |
| `mla_unified` | — | **DROPPED** on HY3 path |
| GQA unified / sigmoid router / YaRN RoPE / fused-FFN | optional | **NEW, optional** (fold into `ew_unified` or new xclbin; not required for first run) |

## 6. Validation

1. `verify_fst.py hy3.fst` + per-layer expert round-trip (dequant one expert block, requant,
   cos vs the GGUF-dequantized BF16 → expect ≥0.99 given Q3_K_M→BF16→MXFP4 double-quant).
2. Per-layer cos vs an HF `hy_v3` reference (pure-PyTorch, real weights) — expect cos>0.99 for
   attention/FFN; residual will have HY3's intrinsic HC recurrence (cf. DS4 ±36 floor — do not chase
   exact-zero residual).
3. End-to-end greedy (`--no-sd --tokens 1`) → coherent first token + argmax matches HF HY3.
4. With MTP: draft acceptance rate (fork reports 85.8 % on RTX 5090; NPU target TBD).
5. `pager_->hit_rate()` ≥ 0.9 (192-expert pool is far smaller than DS4's 1376 → expect high).

## 7. Risks

- **`expert_weights_norm` formula** — Σ-normalize-then-×2.826 vs softmax; verify exactly vs HF
  `hy_v3` before trusting expert selection.
- **YaRN RoPE** — exact curve (factor 4.0, orig 262144); plain RoPE will silently diverge.
- **Per-head Q/K RMSNorm shape `[128]`** broadcast over heads — easy to mis-reshape.
- **MTP** — `eh_proj` concat ordering (`[e_norm; h_norm]`), post-norm chaining (gotcha #6),
  iterative acceptance logic; validate against the fork.
- **Double quantization** (Q3_K_M→BF16→MXFP4) — fine for a speed eval, not a faithfulness claim;
  use BF16 safetensors source if scratch disk allows (see §9).
- **Disk**: GGUF 145 GB + `.fst` ~145 GB peak ≈ 290 GB; at download completion free drops to ~204 GB
  and conversion to ~43 GB headroom — tight, monitor; stream-convert + delete GGUF if needed.
- **0.04–0.06 tok/s floor** is dispatch-COUNT-bound and is **not broken by HY3 alone** (80 vs 43
  layers ≈ 1.86× more attention/norm dispatches). The win is the freed context budget (§4), not raw
  tok/s.

## 8. Implementation order

1. Converter run (step 3, queued on download) → `hy3.fst` + `verify_fst`.
2. `ModelConfig` + `Arch` enum + `Hy3LayerWeights` + arch-branched loader (read stored dims).
3. GQA attention (`process_gqa`) reusing `expert_gemm_vec` + `ew_unified`; per-head Q/K norm;
   YaRN RoPE; uncompressed KV cache.
4. Sigmoid+bias router (`npu_router_hy3`) + dense L0 switch + always-on shared expert.
5. NextN MTP head (iterative, post-norm chaining, shared embd/head).
6. End-to-end greedy `--no-sd --tokens 1` → coherent + argmax match.
7. (Optional, the real lever) fused-FFN xclbin in the freed hw_context slots → toward >1 tok/s.

## 9. Honest speed prognosis

On FaStar's **current** dispatch-COUNT-bound engine (~0.05 tok/s; dispatch-packing was proven *not*
to move tok/s because a packed N-copy blob does N× the NPU compute — the only proven lever to >1
tok/s is IRON-level fusion, which needs free hw_contexts):

- **Slower factors:** 80 vs 43 layers → ~1.86× more attention/norm/router dispatches/token (the
  dominant term on a dispatch-count-bound engine). More KV to manage at prefill.
- **Faster factors:** ~4.4× fewer expert FFN dispatches/token (8×79=632 vs 64×43=2752); smaller
  experts (~9.5 MB vs 16 MB → less DMA); 192-expert pool vs 1376 → higher LRU hit rate.
- **Bottom line:** expect HY3 to land in the same ~0.05 tok/s band, modestly slower per token out
  of the box. **HY3's value is architectural, not raw rate:** GQA collapses MLA's 5 latent kernels
  into ~1 unified context, freeing ≥3 hw_contexts — the headroom fusion needs. HY3 is the better
  *platform to eventually break 1 tok/s*; if the goal is raw tok/s today, stay on DeepSeek V4 Flash.