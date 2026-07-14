# Qwopus3.6-27B Pivot — Investigation, Converter & Tiered-Memory Post-Mortem

**Date:** 2026-07-11 · **Repo target:** `Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF`
· **Branch:** `hy3-pivot-openmp-simd`

## a) Model Architecture — exact specs from the GGUF metadata

Parsed directly from the GGUF header (Q4_K_M file, 16 810 711 616 B / 15.66 GB;
GGUF v3, 866 tensors, 57 KV metadata fields). No model-card claims trusted.

| field | value | note |
|---|---|---|
| `general.architecture` | **`qwen35`** | **NOT** "Qwen 3.6" — it is Qwen3.5-Next |
| `qwen35.block_count` | **65** | 64 trunk + 1 NextN/MTP |
| `qwen35.context_length` | **262 144** | **256 k native, NOT 1 M** |
| `qwen35.embedding_length` | 5120 | hidden_dim |
| `qwen35.feed_forward_length` | 17 408 | dense FFN inter |
| `qwen35.attention.head_count` | 24 | Q heads (GQA) |
| `qwen35.attention.head_count_kv` | 4 | KV heads → 6:1 GQA |
| `qwen35.attention.key_length` | 256 | head_dim |
| `qwen35.attention.value_length` | 256 | head_dim |
| `qwen35.rope.freq_base` | 10 000 000 | long-range RoPE |
| `qwen35.rope.dimension_sections` | [11, 11, 10, 0] | partial rotary |
| `qwen35.rope.dimension_count` | 64 | |
| `qwen35.attention.layer_norm_rms_epsilon` | 1e-6 | |
| `qwen35.full_attention_interval` | **4** | every 4th layer is full attention |
| `qwen35.ssm.conv_kernel` | 4 | Mamba2 conv |
| `qwen35.ssm.state_size` | 128 | SSM state |
| `qwen35.ssm.group_count` | 16 | SSM groups |
| `qwen35.ssm.time_step_rank` | 48 | dt rank |
| `qwen35.ssm.inner_size` | 6144 | SSM inner |
| `qwen35.nextn_predict_layers` | **1** | MTP/NextN head confirmed |
| vocab (from `token_embd.weight` [5120, 248320]) | **248 320** | not weight-tied (`output.weight` is separate) |
| tokenizer | `gpt2` / pre `qwen35`, 248 320 tokens | |

**Quantization = Q4_K_M** (mixed, confirmed per-tensor): 439× Q4_K, 67× Q6_K, 360× F32.
Variants shipped: Q3_K_M (13.5 GB), Q4_K_M (16.8 GB), Q5_K_M (19.5 GB), Q6_K (22.4 GB), Q8_0 (29 GB).

**Hybrid layer structure (authoritative, from tensor inventory):**
- **48 SSM (Mamba2) layers** — `blk.{0,1,2,4,5,6,…}` (every layer except `L%4==3` and the MTP layer). Tensors: `attn_qkv` (5120→10240), `attn_gate` (5120→6144), `ssm_conv1d` (4×10240), `ssm_a` (48), `ssm_alpha`/`ssm_beta` (5120→48), `ssm_dt.bias` (48), `ssm_norm` (128), `ssm_out` (6144→5120). **No attention KV.**
- **16 full GQA attention layers** — `blk.{3,7,11,…,63}` (`L % 4 == 3`). Tensors: `attn_q` (5120→12288), `attn_k`/`attn_v` (5120→1024), `attn_output` (6144→5120), `attn_q_norm`/`attn_k_norm` (256). These are the only layers with a token-growing KV cache.
- **1 MTP/NextN layer** — `blk.64`: full-attention set + `nextn.eh_proj` (10240×5120), `nextn.enorm`/`hnorm`/`shared_head_norm` (5120).
- Every block also has `attn_norm`, `post_attention_norm`, and a dense `ffn_gate`/`ffn_up`/`ffn_down` (5120↔17408).

**Discrepancies vs. the task brief (flagged honestly):**
1. **Not "standard Qwen 3.6 GQA".** It is a hybrid Mamba2-SSM + GQA model. The "reuse the HY3 GQA path" assumption holds for only **17/65 layers**; the other 48 are SSM and need a Mamba2 scan kernel FaStar does not have.
2. **Native context is 256 k, not 1 M.** 1 M is an extrapolation target (YaRN + the SSD/palace tiers), not a trained/trusted length.
3. **Multimodal.** The repo also ships `mmproj-F32.gguf` (a CLIP vision projector; HF tags include `vision`/`multimodal`). The main `…-MTP-Q4_K_M.gguf` is text-only (`general.architecture=qwen35`); vision is out of scope for this pivot.
4. **Q4_K_M is 15.66 GB, not ~15 GB** (close enough; fits in 64 GB RAM alongside the ~2.2 GB hot KV).

## b) Converter Status — `qwen3`/`qwen35` path

**Status: READY (code complete + validated against the real GGUF header; not yet run end-to-end).**

Added to `scripts/fst_converter.py`:
- `QTYPE_MXFP4 = 4` + branch in `_quant_shared()` → requantizes `[out,in]` F32 weights to the existing DS4 dense MXFP4 block layout (1 e8m0 scale + 16 FP4 nibbles / 32 elems) via the existing `_float32_to_dense_blocks()`. `in % 32 == 0` is enforced.
- Fresh Qwen3.5 TIDs 60–79 (`TID_Q35_*`: Q/K head norms, FFN gate/up/down, SSM qkv/gate/conv1d/A/alpha/beta/dt_bias/norm/out, NextN eh_proj/enorm/hnorm/head_norm, `LAYER_TYPES`, `CFG`). Reuses generic TIDs (EMBED, OUTPUT_NORM, LM_HEAD, INPUT_NORM, POST_ATTN_NORM, Q/K/V/O proj) where semantics match, mirroring the HY3 TID philosophy.
- `convert_qwen35(gguf_path, output_path)`: mirrors `convert_hy3` — `gguf.GGUFReader` mmap, dequant Q4_K/Q6_K→F32 via `_gguf_dequant`, requantize large projections + FFN + embeddings + lm_head + eh_proj to **MXFP4**, keep RMSNorms BF16 and small precision-sensitive SSM vectors (A log, dt bias, group norm, conv1d, Q/K norms) **F32**. No expert bank (dense). Hybrid/SSM hyperparams that don't fit the fixed header are packed into reserved `r1`/`r2` (full_attn_interval, nextn, rope sections) + a `TID_Q35_CFG` F32 blob + `TID_Q35_LAYER_TYPES` array the Q35 loader reads — **no header struct change**, so existing `.fst` files / HY3 are unaffected.
- CLI: `python3 fst_converter.py qwen3 <gguf> [output]` (alias `qwen35`), default output `qwopus.fst`.

**Validation (against the 32 MB GGUF header downloaded from HF):**
- `fst_converter.py` parses (syntax OK).
- Layer-kind detection: **48 SSM + 16 full-attn + 1 MTP = 65**, **zero** cadence mismatches vs `L%4==3`.
- Every tensor name the converter dereferences **resolves** (no missing tensors).
- Estimated `qwopus.fst` size: **14.51 GB** (FFN 9.23 + SSM 2.95 + attn 0.95 + embed 0.68 + lm_head 0.68 + nextn 0.03) — within the 14–16 GB target.

**Not done (deliberate):** I did **not** download the 16.8 GB GGUF or produce `qwopus.fst`. The 17 GB download + ~14.5 GB write is costly and the engine cannot execute the result yet (SSM kernels missing — see §c/§d). To run it:
```bash
huggingface-cli download Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF \
    Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf --local-dir .
python3 scripts/fst_converter.py qwen3 Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf qwopus.fst
python3 scripts/fst_converter.py verify qwopus.fst
```
Requires `pip install gguf` (the repo's Python env did not have it; PEP-668 — use `--break-system-packages` or a venv).

## c) Memory Plan — `MEMORY_PLAN.md`

**Status: COMPLETE.** `MEMORY_PLAN.md` written with the full 3-tier architecture, grounded in the engine's existing structures and in online research (llama.cpp KV-mmap PRs #21792/#18747 + issue #20697 + the `--kv-mmap-path` fork; MemGPT/Letta archival storage + recursive summary; antirez ds4 / Anemll ds4-ssd disk-KV-with-session-resume).

Key design facts:
- **Only the 17 attention layers have growing KV** (SSM is O(1), ~48 MB total). KV ≈ **68 KB/token** → 32 k = 2.18 GB (hot), 256 k = 17.4 GB, 1 M = 68 GB.
- **Tier 1 (Hot, 32 k):** per-layer `xrt::bo host_only` for the 17 attn layers + SSM state — reuses `init_kv_cache_bo()` (`fst_engine.cpp:1398`).
- **Tier 2 (Cold, SSD-mmap):** `kv_cache_ssd.bin`, `mmap(MAP_SHARED)` + `madvise`; prefetch via the existing `ExpertPager` background `pread` worker (`src/expert_pager.cpp`); `fallocate(PUNCH_HOLE)` on eviction; SHA1-keyed for session resume (à la ds4).
- **Tier 3 (Memory Palace):** recursive summaries of evicted spans in `qwopus.palace.jsonl` → HNSW vector index; `palace_recall(query,k)` re-injects into working context (MemGPT archival model).
- Integration: new `ARCH_QWEN35` path + `Q35KVCache3T` class; attn layers reuse the SIMD HY3 host-attention path; SSM layers need a new Mamba2 host/NPU kernel.

## d) Bottom line / what blocks running it

1. **Converter: ready** (validated, 14.5 GB output expected). Producing `qwopus.fst` is a `huggingface-cli download` + one command away.
2. **Tiered memory: designed** (`MEMORY_PLAN.md` complete); implementation is engine work, decoupled from the converter.
3. **Hard blocker — SSM kernels:** 48/65 layers are Mamba2 state-space. FaStar has MLA + GQA + FFN kernels only. Until a Mamba2 scan kernel (host OpenMP/SIMD or NPU) and the NextN attention head are implemented, the engine cannot run this model regardless of KV/weights. The GQA attention layers (17) and dense FFN reuse existing HY3 host-attention + fused-FFN paths.
4. **1 M context is an extrapolation target**, not the model's native 256 k — quality unverified beyond 256 k.