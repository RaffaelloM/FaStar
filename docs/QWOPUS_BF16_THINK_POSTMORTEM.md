# Qwopus3.6 — BF16 Conversion + Thinking Mode Post-Mortem

**Date:** 2026-07-13
**Branch:** `hy3-pivot-openmp-simd`
**Task:** Enable thinking mode on the (proven bit-correct) FaStar Qwen3.5-Next engine; if it still loops, the Q4_K_M→MXFP4 double quantization is destroying the reasoning model — convert the original BF16 safetensors directly to MXFP4.
**Verdict:** **Coherence restored.** A single-step BF16→MXFP4 `.fst` (plus a critical RMSNorm `+1` convention fix in the converter) makes the model reason coherently. The Q4_K_M→MXFP4 double-quantization was the cause of the degenerate 2-cycle / 4-token repetition; the C++ engine math was left untouched (it was already proven bit-correct end-to-end).

---

## (a) Thinking mode — enabled; alone insufficient on Q4 weights

`src/fst_main.cpp` (Q35 template block, ~line 857) was changed so thinking mode is **ON by default**: the templated prompt now ends at `<|im_start|>assistant\n` and the model emits its own `малую … малую` reasoning. The old forced empty-think block (`малую\n\nмалую\n\n`, the GGUF `enable_thinking=false` branch) is opt-in via `FST_Q35_NO_THINK=1`. Verified by tokenizing the template: `…<|im_start|>assistant\n` → `[…, 248045, 74455, 198]` — the last input token is `198` (`\n`), i.e. no forced think tags. `248068=`малую (open), `248069=`малую (close).

On the **existing Q4 weights**, thinking mode does **not** fix coherence — it merely moves the attractor. Instead of the no-think ``/`\n\n` 2-cycle, the model enters a 4-token repetition basin:

```
The user is asking the user is asking the user is asking the user is asking …
   (1156 / 369 / 9859 / 279  =  " user " / " is " / " asking " / " the ")
```

So the Q4_K_M→MXFP4 weights are too degraded to reason; thinking mode alone is not enough → proceed to BF16.

## (b) BF16 conversion — success, after a critical RMSNorm `+1` fix

Downloaded the BF16 safetensors snapshot `Jackrong/Qwopus3.6-27B-Coder` (52 GB, 15 shards + `model.safetensors.index.json` + `config.json` + `tokenizer.json`) via `hf download` into `models/Qwopus3.6-27B-Coder-BF16/`. Wrote `convert_qwen35_safetensors(model_dir, output)` in `scripts/fst_converter.py` (wired into the `qwen3`/`qwen35` CLI; auto-detects a directory → safetensors path, a `.gguf` → GGUF path). Output: `qwopus_bf16.fst`, **14 GB, `verify_fst` PASSED**.

The converter reuses the proven GGUF path's MXFP4 packing (`_float32_to_dense_blocks` — identical layout, only the input precision differs: lossless BF16 vs lossy Q4_K_M). All HF tensor names were verified against the index exactly (SSM `linear_attn.{in_proj_qkv,in_proj_z,conv1d,A_log,in_proj_a,in_proj_b,dt_bias,norm,out_proj}`, GQA `self_attn.{q,k,v,o,q_norm,k_norm}`, `mlp.{gate,up,down}`, globals `embed_tokens`/`norm`/`lm_head`). Transforms: `neg_exp` (`A_log → -exp(A_log) = ssm_a`), `squeeze_conv1d` (`[CDIM,1,CK]→[CDIM,CK]`, channel-outer flat `c*CK+t` matching the engine's `w.ssm_conv1d[c*ck+r]`). 64 trunk layers L0..L63 emitted; `n_layers=65` with an empty L64 MTP slot — the loader tolerates the missing L64 tensors (`get()` returns `found=false`; all loaders check it) and generation never runs L64 (`ndec = n_layers-1 = 64`).

### The bug the first conversion exposed

The first `qwopus_bf16.fst` was **incoherent-on-arrival**: a tensor-by-tensor diagnostic (`scripts/diag_bf16.py`) against the known-good Q4 `.fst` showed the norm weights were off by **exactly +1.0**:

| tensor | Q4 .fst[0] | HF[0] | Δ |
|--------|-----------|-------|---|
| output_norm | 1.96875 | 0.96875 | **1.0** |
| input_norm L0 | 1.05469 | 0.05835 | ≈1.0 (BF16) |
| q_norm L3 | 1.18164 | 0.18164 | 1.0 |

**Root cause:** Qwen3.5-Next uses **Gemma-style RMSNorm** — `y = x/rms(x) * (1 + weight)`. HF stores the *delta* `weight`; the engine's `q35_rmsnorm` (`fst_engine.cpp:4763`, `out[d] = h[d]*rcp*w[d]`) and `rmsnorm_head` (`:5224`) multiply the stored value **directly, with no `+1`**. The GGUF/llama.cpp convention (and the engine's expectation) is that the `.fst` holds the **full scale `1+delta`**. The first BF16 converter stored the bare HF delta → the engine would have multiplied by ~0.058 instead of ~1.058 (a ~18× norm error → instant garbage).

`ssm_norm` (the GDN group norm, `linear_attn.norm.weight`) is a **plain scale, not `1+delta`** — confirmed: Q4 `.fst` ssm_norm == HF ssm_norm **exactly** (max|Δ|=0). Left unchanged.

### The fix

Added an `add_one` transform (`arr = arr + 1.0`) applied to the 5 Gemma-style norms in the converter: `TID_INPUT_NORM`, `TID_POST_ATTN_NORM`, `TID_OUTPUT_NORM`, `TID_Q35_Q_NORM`, `TID_Q35_K_NORM`. Re-converted. Post-fix validation (`scripts/validate_bf16_fst.py` + `diag_bf16.py`):

| tensor | BF16-.fst vs HF source | note |
|--------|------------------------|------|
| 5 Gemma norms (input/post_attn/output/q/k) | match Q4 .fst **exactly** (max\|Δ\|=0) | `+1` fix landed |
| ssm_norm | == HF **exactly** | plain scale, no `+1` |
| conv1d | == HF **exactly** (max\|Δ\|=0) | channel-outer layout correct |
| ssm_a | == `-exp(A_log)` **exactly** | `neg_exp` correct |
| dt_bias | == HF **exactly** | |
| in_proj_qkv (MXFP4 dequant) | shape (10240,5120), **cos 0.990** | 4-bit quant of BF16 |
| q_proj L3 (MXFP4 dequant) | shape (12288,5120), **cos 0.990** | |
| ffn_gate L0 (MXFP4 dequant) | shape (17408,5120), **cos 0.991** | |
| embed (MXFP4 dequant) | shape (248320,5120), **cos ≈1.0** | |

(`ssm_a`/`dt_bias` are a *permutation* of the GGUF order — the GGUF and HF disagree on v-head ordering — but this is harmless: the 48 SSM v-heads are independent, so a consistent head order across all SSM tensors produces an identical final output. The BF16 `.fst` is consistent in HF order throughout, so it matches HF transformers; the engine-vs-ref bit-correctness proof already showed head order is not load-bearing.)

## (c) Decode throughput

`XILINX_XRT=/usr ./build/ds4_npu_engine --model qwopus_bf16.fst --prompt "Write a C++ function to add two numbers." --tokens 50 --temp 0.6`

```
Q35 Prefill: 85166.8 ms (18 prompt tokens, 64 trunk layers)
Q35 Decode Tokens/sec: 0.21
Peak RAM (RSS): 20.9 GB
NPU Contexts: creates=6 evictions=0 hits=0 active=6
```

**0.21 tok/s** — identical to the Q4 run (0.20). Expected: the BF16→MXFP4 `.fst` is the same MXFP4 format/size as the Q4_K_M→MXFP4 one, so dispatch/throughput is unchanged. **The win is coherence, not speed.** (The ~0.16–0.21 tok/s floor is the host-attention + NPU-dispatch bound analyzed in prior post-mortems; untouched here per the "do not touch engine math" rule.)

## (d) Exact generated text + coherence verdict

### BF16 weights, thinking mode (50 tokens) — `logs/run_bf16.txt`, `logs/decode_bf16.txt`

The model's first predicted token is `248068` = `малую` (it opens its own think block), then produces structured reasoning inside it:

```
малую
1.  **Analyze the Request:**
    *   Goal: Write a C++ function to add two numbers.
    *   Language: C++
    *   Input: Two numbers (implicitly, can be integers,
```

### BF16 weights, thinking mode (200 tokens, partial — hit the 10-min tool wall) — `logs/decode_bf16_long.txt`

```
малую
1.  **Analyze the Request:**
    *   Language: C++
    *   Task: Write a function to add two numbers.

2.  **Determine the Function Signature:**
    *   Needs two input parameters (let's call them `a` and `b`).
    *   Needs a return type (let's use `int
```

### Contrast: Q4 weights, thinking mode — `logs/run_think.txt`

```
малую
The user is asking the user is asking the user is asking the user is asking …
```

### Coherence verdict

**COHERENT.** With BF16-source weights the model (a) opens a genuine think block, (b) decomposes the request into Goal / Language / Input, then (c) progresses to a second structured step (Function Signature: two parameters `a`/`b`, return type `int`…) — progressive, non-repetitive reasoning toward writing the C++ function. The degenerate 2-cycle (no-think) and 4-token repetition (think) are both gone. This is exactly the behavior expected of a reasoning "Coder" model, and confirms the hypothesis: the Q4_K_M→MXFP4 double quantization was destroying the reasoning weights; a single-step BF16→MXFP4 conversion restores coherence.

## What changed / what did not

**Changed (converter + CLI only — no engine math):**
- `src/fst_main.cpp`: thinking mode default ON; `no_think` opt-in via `FST_Q35_NO_THINK`.
- `scripts/fst_converter.py`: `convert_qwen35_safetensors` (BF16→MXFP4 single-step); `add_one` transform for the 5 Gemma-style RMSNorm weights; `neg_exp` / `squeeze_conv1d` transforms; `qwen3` CLI auto-detects directory vs GGUF.

**Not changed:** `src/fst_engine.cpp` (engine proven bit-correct end-to-end; left untouched per the task rules). The fix lives entirely in weight preparation.

## Artifacts

- `qwopus_bf16.fst` — 14 GB BF16→MXFP4 weights (norms `1+delta`).
- `models/Qwopus3.6-27B-Coder-BF16/` — 52 GB BF16 safetensors snapshot (downloaded).
- `scripts/convert_qwen35_safetensors` (in `fst_converter.py`), `scripts/validate_bf16_fst.py`, `scripts/diag_bf16.py`, `scripts/decode_run.py`.
- `logs/run_bf16.txt`, `logs/decode_bf16.txt`, `logs/run_bf16_long.txt`, `logs/decode_bf16_long.txt`, `logs/validate_bf16.txt`, `logs/diag_bf16.txt`, `logs/convert_bf16.txt`, `logs/run_think.txt` (Q4 contrast).