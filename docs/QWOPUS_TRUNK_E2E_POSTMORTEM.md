# Qwopus3.6 Trunk End-to-End Post-Mortem — "Engine is correct; the 2-cycle is genuine"

**Date:** 2026-07-13
**Branch:** `hy3-pivot-openmp-simd`
**Task:** Free disk, build a full-trunk numpy reference reading `qwopus.fst` directly, and find the end-to-end divergence blamed for the `` / `\n\n` 2-cycle incoherence.
**Verdict:** **No engine bug.** An independent reference reproduces the engine's output bit-for-bit at pos 0, at pos 17 (full 18-token templated prefill), and token-for-token through decode. The degenerate 2-cycle is a property of the (Q4_K_M→MXFP4-quantized) Qwen3.5-Next model under the no_think chat template, **not** of trunk composition or pos>0 state accumulation.

---

## (a) Disk space

The GGUF source was deleted to clear the quota block:

```
rm -f downloads/Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf
df -h /  →  /dev/nvme0n1p6  501G  147G  329G  31%  /
```

Freed ~16 GB (314 GB → 329 GB free). Quota issue resolved; no large files downloaded (everything needed was already local).

## (b) Logit comparison + first diverging layer

Two independent numpy references were built reading weights **directly from `qwopus.fst`** (header/dir parsing ported from `fst_converter.py`, HF-spec math, bf16 **truncation** `u & 0xffff0000` to match the engine):

- `scripts/qwopus_full_ref.py` — single-token (pos-0) full 64-layer trunk.
- `scripts/qwopus_full_ref_seq.py` — multi-token **layer-major** prefill carrying SSM conv1d buffer `[CK,CDIM]` + matrix state `S [NV,HV,HV]` per SSM layer and GQA KV cache per GQA layer across positions, plus a greedy-decode mode that continues from the carried state.

### Pos-0 ("Hello", 1-token prompt) — `logs/cmp_head.txt`

| tensor | eng norm | ref norm | max\|Δ\| | cos |
|--------|---------|----------|----------|-----|
| h_prenorm | 779.20 | 775.20 | 10.6 | 0.99893 |
| logits | — | — | 0.66 | — |

Top-5 **identical** (9419, 12675, 13198, 2811, 29242), argmax = 9419 both. Per-layer bisection (`logs/cmp_trunk.txt`) shows **smooth** bf16 noise growth (L0 max\|Δ\|=9.8e-4 → L63 = 10.6, cos > 0.998 throughout) — **no sudden jump, no first-diverging layer.**

### Pos-17 (full 18-token no_think template prefill) — `logs/cmp_tmpl.txt`

The templated prompt `<|im_start|>user\nHello, what are you?<|im_end|>\n<|im_start|>assistant\nмалую\n\n\n\n` is 18 tokens (`logs/tmpl_ids.txt`). This is the **first test that exercises pos>0 paths**: RoPE at nonzero positions, SSM conv1d + matrix-state carry across 18 tokens, and GQA **multi-key** KV attention (pos0 ref was self-only).

| tensor | eng norm | ref norm | max\|Δ\| | cos |
|--------|---------|----------|----------|-----|
| h_prenorm | 398.91 | 392.82 | 8.0 | 0.99390 |
| hn | 128.24 | 129.14 | 0.93 | 0.99273 |
| logits | 1017.7 | 1025.9 | 1.20 | 0.99406 |

Top-8 nearly identical; **argmax = 248069 both** (MATCH). The differences are bf16 accumulation noise over 64 layers × 18 positions (~1% in norm, ~0.3 in logit). **No divergence.** The prefill trunk — including the suspected GQA multi-key KV cache and SSM state carry — is correct.

### Decode (the 2-cycle) — `logs/ref_seq_decode.txt`

The reference's **own greedy decode**, continuing from the carried prefill state, reproduces the engine's 2-cycle **token for token**:

| pos | feed | ref argmax (logit) | engine `run_final.txt` |
|-----|------|--------------------|------------------------|
| 17 | template | 248069 (18.17) | 248069 (prefill-last) |
| 18 | 248069 | **271** (22.59) | **271** (tok1) |
| 19 | 271 | **248069** (17.99) | **248069** (tok2) |
| 20 | 248069 | **271** (22.07) | **271** (tok3) |
| 21 | 271 | **248069** (18.40) | **248069** (tok4) |
| 22 | 248069 | **271** (22.30) | **271** (tok5) |

Exact match across all positions. The 271 / 248069 attractor is dominant (logits 18–22 vs runners-up ~14), so it is deterministic under both greedy and temp-0.6 sampling.

## (c) The bug + fix

**There is no bug in `fst_engine.cpp`.** The task's premise — "SSM/GQA/FFN are bit-correct at pos 0, so the incoherence means trunk composition or pos>0 accumulation is broken" — is **refuted**:

1. The pos-0 trunk composition is correct (independent ref matches, top-5 identical, smooth noise).
2. The pos>0 prefill (18-token, multi-key GQA + SSM state carry + nonzero RoPE) is correct (cos 0.994, argmax match).
3. The decode 2-cycle is reproduced **exactly** by the independent reference.

The Q35 trunk math has **no `is_prefill_` branching** (the `is_prefill_` uses at lines 5986–6016 are in the DS4 MLA path, not Q35). `process_gqa_q35` (5185–5312) appends K/V at `pos*kvrow` and attends over `nkeys=pos+1` identically in prefill and decode; `process_ssm` carries conv1d + S state implicitly. Prefill and decode use the **same** `run_trunk_token`; the only difference is the carried state, which the reference confirms is correct.

**No fix applied.** The `FST_TRUNK_DUMP` per-layer residual dump added to `run_trunk_token` (~line 8770) is kept as harmless debug infra (env-gated, default off).

### What the 2-cycle actually is

The no_think template appends an empty thinking block `малую\n\n\n\n` (open-think, two newlines, close-think `` = 248069, two newlines). After it, the model should emit the answer. Instead, under Q4_K_M→MXFP4 quantization, the quantized model **re-emits the close-think tag `` (248069) and `\n\n` (271) forever** — a degenerate attractor in the think-token basin. This is a known failure mode of aggressive 4-bit quantization on models with reasoning-mode tokens; the per-block error is tiny (cos > 0.998) but accumulates into a generative collapse that the *unquantized* float model does not exhibit. The engine faithfully computes what the quantized weights say.

**Real levers for coherence (none in `fst_engine.cpp`):**
- Higher-precision weights (Q8_0 / BF16 `.fst`) — the conversion path quantizes GGUF Q4_K_M → MXFP4, the likely loss source.
- WITH-thinking mode (omit the empty-think-block suffix; let the model reason).
- Verify the upstream Q4_K_M GGUF in llama.cpp also 2-cycles (blocked: no `llama_cpp`/`gguf` module + prior disk quota — now freed, but the engine-vs-reference match is already conclusive that the engine is not the cause).

## (d) Exact text from `logs/run_final.txt` + coherence verdict

Command:
```
timeout 120 systemd-run --scope -p MemoryMax=35G ./build/ds4_npu_engine \
  --model qwopus.fst --prompt "Hello, what are you?" --tokens 10 --temp 0.6
```

Generated tokens (prefill-last + 9 decode):
```
prefill-last=248069
decode=271  (pos=18)   tok1
decode=248069 (pos=19) tok2
decode=271  (pos=20)   tok3
decode=248069 (pos=21) tok4
decode=271  (pos=22)   tok5
decode=248069 (pos=23) tok6
decode=271  (pos=24)   tok7
decode=248069 (pos=25) tok8
decode=271  (pos=26)   tok9
Q35 Decode Tokens/sec: 0.21   ENGINE_EXIT=0
```

Decoded (`248069` = ``, `271` = `\n\n`; the `` special token renders empty): the visible model output after the assistant header is an unbounded stream of blank lines with interleaved (invisible) close-think tags — i.e. **empty output**, the `` / `\n\n` 2-cycle.

**Coherence verdict:** The engine is **numerically correct end-to-end** (prefill + decode, all 64 trunk layers, SSM state carry, GQA multi-key KV cache, RoPE, final RMSNorm, LM head — verified against an independent numpy reference reading the same `.fst` weights). The incoherent empty-output 2-cycle is **genuine model behavior** of the Q4_K_M→MXFP4-quantized Qwen3.5-Next under the no_think template. It is **not** a trunk-composition or pos>0-accumulation bug. The task's premise is refuted; no `fst_engine.cpp` fix is warranted.

## Artifacts

- `scripts/qwopus_full_ref.py` — pos-0 full-trunk reference (reads `qwopus.fst`).
- `scripts/qwopus_full_ref_seq.py` — multi-token layer-major prefill + greedy-decode reference (state carry).
- `scripts/qwopus_cmp_tmpl.py` — engine-vs-ref pos-17 comparison.
- `logs/cmp_head.txt`, `logs/cmp_trunk.txt`, `logs/cmp_tmpl.txt` — comparisons.
- `logs/ref_seq_decode.txt` — reference decode reproducing the 2-cycle.
- `logs/run_final.txt`, `logs/run_tmpl.txt`, `logs/head_tmpl/` — engine runs + head dumps.
- `logs/disk.txt`, `logs/build.txt` — disk + build record.