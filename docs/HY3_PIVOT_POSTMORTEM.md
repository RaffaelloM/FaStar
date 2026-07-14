# HY3 Pivot — Post-Mortem Report (5-step process)

> **Date:** 2026-07-09. **Objective:** pivot FaStar to support Tencent Hunyuan-3.0 (HY3, `hy_v3`)
> by converting the GGUF `hy3-1M-MTP-Q3_K_M.gguf` (~145 GB) to an MXFP4 `.fst` and planning the C++
> engine changes. This report covers the required (a) architecture specs, (b) converter status,
> (c) engine-plan completeness. The detailed engine plan is `HY3_PLAN.md`; the unrelated
> MLA-consolidation `HY3_POSTMORTEM.md` is left untouched.

## Process status

| Step | Task | Status |
|---|---|---|
| 1 | Inspect GGUF metadata + architecture | ✅ done |
| 2 | Extend `fst_converter.py` for HY3 (`hy_v3`) | ✅ code-complete + unit-tested |
| 3 | Launch conversion in background | ⏳ queued (download ~65 %; see (b)) |
| 4 | Plan C++ engine changes (`HY3_PLAN.md`) | ✅ done |
| 5 | Post-mortem report (this file) | ✅ done |

---

## (a) HY3 architecture — exact specs

Parsed **offline from the GGUF metadata** (`scripts/hy3_parse_header.py`, manual GGUF header parser;
values cross-checked against `tencent/Hy3` `config.json` and the `hy_v3` modeling code). The
converter reads these straight from GGUF KV; constants below are the inspected defaults.

| Spec | Value |
|---|---|
| `general.architecture` | `hy_v3` |
| Blocks | **81** (`blk.0..80`): L0 dense + L1..79 MoE + L80 NextN MTP |
| Hidden (`embedding_length`) | 4096 |
| Attention | **GQA**, 64 Q heads / 8 KV heads (8:1), `head_dim` 128 |
| Per-head Q/K RMSNorm | `qk_norm: true`, weight shape `[128]` (V has no norm) |
| Experts | **192**, top-**8** routed + **1 always-on ungated shared** expert |
| Expert inter (`expert_feed_forward_length`) | **1536** |
| Dense FFN inter (L0, `feed_forward_length`) | **13312** |
| `first_k_dense_replace` | **1** (L0 dense, L1+ MoE) |
| Router | **sigmoid + per-expert bias** (`expert_gating_func=2`); bias added **only to top-k selection score**; selected weight = unbiased sigmoid; **route_norm ÷Σ then × `expert_weights_scale` 2.826** |
| Activation | SiLU, parallel gate‖up → down |
| Norm | RMSNorm, `layer_norm_rms_epsilon` = 1e-5, everywhere |
| RoPE | NeoX rotate-half, `rope.freq_base` = 11158840, **YaRN** (`rope.scaling.factor` 4.0, orig 262144) → 1 M ctx |
| Vocab (`token_embd.weight` shape) | **120832** |
| MTP | `blk.80` NextN, `nextn_predict_layers=1`; **shares trunk `token_embd` + `lm_head`**; MTP-only tensors: `eh_proj` `[2*hidden,hidden]`, `enorm`, `hnorm`, `shared_head_norm`; **post-norm hidden chaining**; iterative single-token drafting |
| Source quant | **Q3_K_M** (mixed K-quants) |

**Expert geometry (validated, byte-exact):** gate/up `[out=1536, in=4096]`, down `[out=4096, in=1536]`.
Each projection = `(1536×4096/32)×17 = 3,342,336 B`. One expert block = gate+up+down =
**10,027,008 B** = `2448 × 4096` (exactly page-aligned → `stride == bytes`). This is the **same
17-byte MXFP4 dense layout** (1 e8m0 scale + 16 FP4 nibbles per 32 elements) the existing
dequant+GEMM kernel already consumes — only the block size differs, read from the header.

**Header packing (`r1`/`r2`, no struct change):** `r1 = 0x20001` (low16 `first_k_dense=1`, high16
`gating_func=2`); `r2 = 0xFA00002B1F10` (high32 `int(yarn_factor×1000)=4000`, low32
`int(ew_scale×1e6)=2826000`). `et = expert_count_total = 81×192 = 15552`.

**Key contrast with DeepSeek V4 Flash (current):** HY3 is **GQA, not MLA** — the entire MLA pipeline
(`wq_a` latent compression, KV-latent 512, nope/rope split, `fst_mla_unified.xclbin`, V4
KV-compressor) does not apply. HY3 uses a conventional q/k/v/o + per-head-norm + RoPE + full
8×128 KV-cache path, a sigmoid+bias router (not sqrtsoftplus), a dense L0, an always-on shared
expert, and a NextN MTP head (not a separate DSpark draft model).

---

## (b) Converter status

**Code-complete and launch-ready.** `scripts/fst_converter.py` has a new `hy_v3` arch path:

- **GGUF reader:** uses `gguf.GGUFReader` (mmap) per the user's rule ("use the Python `gguf` library
  to parse the GGUF offline"). Helpers: `_gguf_get` (scalar KV via `reader.fields`, handles
  STRING/BOOL/numeric), `_gguf_dequant` (`gguf.quants.dequantize` → float32 → reshape
  `tuple(reversed(shape))` → `[out,in]` / `[n_experts,out,in]`), `_hy3_expert_block_bytes`.
- **`convert_hy3` (fst_converter.py:1642):** opens the GGUF, asserts `hy_v3`, reads cfg from KV
  (with inspected-value fallbacks), packs HY3 arch constants into `r1`/`r2`, builds shared entries
  (globals + per-layer 0..80 GQA attn + per-head Q/K norm + both RMSNorms; L0 adds dense FFN;
  L1..80 add router+bias+shared expert; L80 adds the 4 NextN tensors), writes `.fst` with the
  existing MXFP4 expert bank layout (layers 1..80 × 192 experts, stride from cfg), then `verify_fst`.
- **Reuses the proven MXFP4 path unchanged:** `quantize_mxfp4_tile` / `_float32_to_dense_blocks`
  — the only piece that touches the NPU's exact dequant layout. Dequant+GEMM kernel byte-identical.
- **New HY3 TIDs 50–58** (per-head Q/K norm, dense L0 gate/up/down, NextN eh_proj/enorm/hnorm/shared_head_norm); reuses TIDs 0–13 + router/shared TIDs.
- **Dynamic expert block size** from `expert_dim=1536` (10,027,008 B), honored via cfg overrides in
  `_pack_header` (`expert_block_bytes`/`expert_block_stride`) — backward-compatible.
- **CLI:** `python3 fst_converter.py hy3 <gguf> [output]` → `convert_hy3` then `verify_fst`
  (fst_converter.py:2594–2600). `python3 -m py_compile` **passes**; all unit tests pass; the
  gguf→dequant→MXFP4 path was validated on real HY3 bytes (all finite, plausible magnitudes,
  per-expert and 192-batch pack match).

**Conversion run (step 3): queued behind the download.** `hy3-1M-MTP-Q3_K_M.gguf` is downloading
(`scripts/hy3_download.py`, PID 18344, resumable Range) at ~94 GB / 145 GB (~65 %), ~55 MB/s. The
converter needs the full file (`GGUFReader` builds tensor views across the whole file), so the run
cannot start until the download completes. A background watcher is armed to notify on completion.

**Launch command (ready to fire on download completion):**
```bash
setsid nohup python3 scripts/fst_converter.py hy3 hy3-1M-MTP-Q3_K_M.gguf hy3.fst > convert_hy3.log 2>&1 &
```
**"Verify it starts without crashing" checks** (within ~30 s of launch): log shows
`Opening GGUF (mmap): …`, the `HY3: 81 blocks … 192 experts top-8 … GQA 64/8 @ 128 … vocab=120832`
line, `expert block = 10,027,008 B (stride 10,027,008)`, `shared entries: N`, and `Expert bank: 80
MoE layers x 192 experts …` — with no `Traceback`/`KeyError` (missing-tensor) / `ValueError`
(not-hy_v3). The conversion itself (dequant+pack of 80×192 experts) runs for a while; completion is
not required to satisfy step 3, only a clean start.

**Disk headroom:** 501 GB volume, 267 GB free at last check. Peak = GGUF 145 GB + `.fst` ~145 GB
≈ 290 GB → ~43 GB headroom at conversion peak. Tight but feasible; monitor; stream-convert + delete
the GGUF if it tightens.

---

## (c) Engine-plan completeness

`HY3_PLAN.md` is written and is the step-4 deliverable. It is **complete** for a first HY3 run and
covers, with concrete file/line anchors:

1. **`ModelConfig` + `Arch` enum** (`fst_engine.h:127`): adds `num_q_heads, num_kv_heads, head_dim,
   expert_inter_dim, first_k_dense_replace, expert_gating_func, expert_weights_scale,
   rope_scaling_factor, yarn_orig_ctx`; loader detects HY3 and unpacks `r1`/`r2`. DS4 path gated on
   `arch`, untouched.
2. **`Hy3LayerWeights` + arch-branched loader:** reads **actual stored dims** from directory entries
   (explicitly avoids the MLA hardcoded `load_bf16_vec(14,…,1024)` at fst_engine.cpp:1917). Expert
   paging unchanged beyond using the header `eb`/`es`.
3. **GQA attention** (`process_gqa`, replaces `process_mla` :3320): RMSNorm → q/k/v GEMM (reuse
   `expert_gemm_vec`) → per-head Q/K RMSNorm (reuse `ew_unified`) → YaRN RoPE (new host+EW) →
   uncompressed 8×128 KV cache → softmax (reuse `ew_unified`) → o proj. **0 new xclbins.**
4. **Sigmoid+bias router** (`npu_router_hy3`, replaces `npu_router` :936): reuse `router_gemm` GEMM;
   new post-GEMM sigmoid + top-8 on `sigmoid+bias` + unbiased-weight ÷Σ × 2.826; **no hash routing**.
5. **Dense L0 switch** (`process_layer` :2584): `lid==0` runs dense SwiGLU FFN (TID 52/53/54, inter
   13312), no MoE.
6. **Always-on shared expert:** added to every MoE layer's FFN sum via reused `process_shared_expert`
   (:4373).
7. **NextN MTP head** (replaces DSpark :4534/5493): shares trunk embd+lm_head; `enorm`/`hnorm`→concat→
   `eh_proj`→full blk.80 block→`shared_head_norm`→shared `lm_head`; **post-norm hidden chaining**;
   iterative single-token drafting up to `n_max=3` with `p_min`.
8. **hw_context budget (the real prize):** HY3 drops `mla_unified` → ~4 contexts vs DS4's ~5–6,
   **freeing ≥3 of the 9-cap** — headroom for the deferred fused-FFN xclbin (the only proven >1 tok/s
   lever).
9. **Reused unchanged:** MXFP4 dequant+GEMM kernels, `ExpertPager` (header-driven `eb`/`es`), `.fst`
   container, BO pool, `lm_head`, `ew_unified`. **New mandatory kernels: none** (optional GQA-unified
   / sigmoid / YaRN-RoPE / fused-FFN xclbins listed but not required for first run).
10. **Validation:** `verify_fst` + per-expert round-trip cos; per-layer cos vs HF `hy_v3` reference
    (>0.99); end-to-end greedy coherent + argmax match; MTP acceptance; `hit_rate ≥ 0.9`.
11. **Risks:** `expert_weights_norm` formula, YaRN RoPE curve, per-head norm reshape, MTP concat
    ordering + post-norm chaining, double-quantization quality, disk headroom, the 0.04–0.06 tok/s
    floor not broken by HY3 alone.
12. **Implementation order:** converter run → ModelConfig+loader → GQA → sigmoid router+dense L0+
    shared → NextN MTP → e2e greedy → (optional) fused-FFN.

**Honest prognosis (carried into the plan):** on the current dispatch-COUNT-bound engine HY3 lands
in the same ~0.05 tok/s band (80 vs 43 layers ≈ 1.86× more attn/norm dispatches), modestly slower
per token out of the box, but with ~4.4× fewer expert dispatches (632 vs 2752) and — the actual
value — a healthier hw_context budget that unblocks the fusion path to >1 tok/s.

---

## Next action

Wait for the download-completion watcher, then fire the step-3 launch command above and verify a
clean start (the checks in (b)). Conversion completion + `hy3.fst` integrity then unblocks engine
implementation per `HY3_PLAN.md` §8.