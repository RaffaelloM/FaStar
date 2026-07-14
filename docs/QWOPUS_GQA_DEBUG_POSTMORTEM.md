# QWOPUS GQA/FFN Debug Postmortem (2026-07-12)

Continues the "CRITICAL RESUME" task: SSM (Gated Delta Net) block was already
proven bit-correct, so the incoherence (`\n<|im_start|>assistant<|im_end|>`
empty-turn loop) had to be downstream in GQA, FFN, MTP, lm_head, or the chat
template. Rules: ZERO CPU fallbacks for math; use the bit-correct 3-pass GDN
kernel (the fused kernel has a race condition).

## (a) Terminal Status

The Bash tool recovered but has a glitch: it returns exit code 1 and reports
"no stdout" even on success. **Workaround that works reliably:** redirect
stdout/stderr to an absolute repo path, then `Read` the file. Every command in
this investigation used that pattern (e.g. `... > logs/x.txt 2>&1; ...` then
`Read logs/x.txt`). CWD is `/home/raffaele/Progetti/FaStar/kernels`, so
absolute paths (or `cd /home/raffaele/Progetti/FaStar && ...`) are required.

## (b) 3-Pass Restore

Restored and verified. `gdn_scan_vheads()` (fst_engine.cpp ~4996) dispatches
the 3 proven xclbins `q35_gdn_passA` / `q35_gdn_delta` / `q35_gdn_passB` in
sequence, each via `kernel_cache_->run_registered_blob(..., 2).wait()`. The
fused-kernel call path is removed. The 6 GDN BOs are allocated once
(`q35_gdn_bos_ready_`). S-state lives in the host vector `q35_ssm_state_[l].S`,
packed into `q35_bo_pA_in_` / `q35_bo_pB_in_` each call and copied back after
passB. Builds clean (`cmake --build build`, with
`TMPDIR=…/build/tmp` to dodge the tmpfs quota). The engine registers all 5
NPU contexts (expert_gemm_vec, expert_gemm_down, gdn_passA/delta/passB, ew_unified)
and dispatches.

## (c) GQA / FFN Bug

**Finding: GQA and FFN are bit-correct at position 0. There is no bug there.**

`process_gqa_q35()` (fst_engine.cpp ~5124) and `process_ffn_q35()` (~5314) were
instrumented with `FST_GQA_DUMP`-gated dumps for the first GQA layer (L3) at
pos 0: residual_in, attn_norm_out, q_proj/q_gate/k_proj/v_proj,
q/k_postqknorm, q/k_postrope, attn_scores, attn_softmax, attn_out, attn_gated,
o_proj, attn_residual_out, and the full FFN chain. Raw MXFP4 weight bytes +
fp32 norms were dumped so the numpy ref uses the EXACT same weights.

`scripts/qwopus_gqa_ref.py` recomputes every stage per the HF
`Qwen3NextAttention` / `Qwen3NextMLP` math contract (dequant_mxfp4 with the
correct interleaved nibble order, bf16 = **truncation** `u & 0xffff0000`
matching the engine's `f2bf`, RMSNorm, NeoX RoPE n_rot=64, per-head q/k norm,
sigmoid gate, o_proj).

After matching the engine's bf16 truncation, **all 22 GQA+FFN tensors are
bit-correct (max|Δ| ≤ 4.5e-5).** The one apparent divergence (attn_scores) was
a *reference artifact* — the ref initially used round-to-nearest-even bf16
while the engine truncates; the bf16 KV-cache round-trip error accounted for
it. There is no GQA/FFN bug at pos 0.

### Chat-template hypothesis (tested, NOT the root cause)

`fst_main.cpp` line ~862 built the Qwen3.5-Next prompt as
`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n` — leaving the model
in **thinking mode**. The GGUF `chat_template`'s `enable_thinking is false`
branch injects the empty thinking block `малую\n\n\n\n`
(hex `3c7468 696e6b3e 0a0a 3c2f7468696e6b3e 0a0a`) after `assistant\n`.
The fix was applied: the template now appends `малую\n\n\n\n`. This is
correct (matches the GGUF template) and the HF tokenizers bridge recognizes
`малую`/`</малую>` as special tokens 248068/248069.

**But the incoherence persists.** With the fix, greedy (temp=0) decode now
emits a deterministic 2-cycle `271`/`248069` = `\n\n` / `</малую>` loop:
```
prefill 18 tokens: [...,74455(assistant),198,248068(малую),271(\n\n),248069(</малую>),271(\n\n)]
decode: 271, 248069, 271, 248069, 271, 248069, ...
```
Both the old (thinking-mode) and new (no_think) templates degenerate, just to
different loops. **The chat template was NOT the root cause** — it only
selected which degenerate attractor the model falls into.

### Localization

Greedy is deterministic (same loop as temp=0.6), so this is not a sampling
artifact — the trunk's top-1 at decode positions is wrong. A raw single-token
prompt `Hello` (FST_Q35_RAW_PROMPT=1) also degenerates: `9419("Hello")` repeated
forever. The first-decode logits (head_step call 0, |h|²/hd=118.6) have
top1 = 9419 = the **input token itself**, the classic signature of a hidden
state dominated by the residual (token embedding) rather than transformed by
the trunk.

Ruled out by code inspection:
- KV-cache reset: `q35_kv_cache_[l].n = 0` runs only once, before prefill
  (line ~8788); the cache persists prefill→decode.
- GDN S-state staleness: S lives in `q35_ssm_state_[l].S`, zeroed at generation
  start (line ~8786); the BOs are per-call scratch.
- KV-cache layout: write `kv.k[pos*kvrow + kvh*dh + d]` matches the read
  `kv.k[kk*kvrow + kvh*dh + d]` and the verified pos0 path.

The remaining, **untested** gap is the **end-to-end trunk composition at
pos>0**:
1. **GQA multi-key KV cache** — the pos0 verification attends to a single key
   (self-only), so the multi-key attention at prefill pos1-17 and decode pos18+
   was never exercised against a reference.
2. **Full trunk → final_norm → lm_head logits** — every "bit-correct" claim so
   far is *block-level* (SSM block, GQA block, FFN block, each fed identical
   inputs). The *composition* across all 64 layers + final RMSNorm + lm_head
   was never compared end-to-end against HF/llama.cpp. `head_step` (fst_engine.cpp
  ~8707) now dumps the full pos0 hidden state + logits under `FST_HEAD_DUMP`
   (`head_p0_h_prenorm.f32`, `head_p0_hn.f32`, `head_p0_logits.f32`) to enable
   that comparison.

### Blocker

The decisive reference comparison is blocked by the environment:
- No `llama_cpp` / `gguf` Python module, so the GGUF model can't be run for
  reference logits. `scripts/qwopus_ssm_ref.py` is a self-contained numpy/torch
  math reference (the GDN scan), not a HF model loader.
- **User disk quota exceeded** (Errno 122 on flush; `/` has 314G free but the
  user quota is at its limit — the 14.5GB `.fst` + 14.5GB GGUF + build consume
  it). `pip install llama-cpp-python` is therefore not attempted.

## (d) Output Text

- **Before the no_think fix** (old template, ending `assistant\n`): degenerate
  `\n<|im_start|>assistant<|im_end|>` empty-turn loop (the originally reported
  incoherence).
- **After the no_think fix** (greedy, temp=0, `--prompt "Hello, what are you?"`):
  ```
  малую\n\n</малую>\n\n</малую>\n\n</малую>\n\n</малюу>\n\n...
  ```
  i.e. tokens `248068, 271, 248069, 271, 248069, 271, 248069, 271` — a
  `\n\n`/`</малую>` 2-cycle. **Still incoherent.**
- **Raw `Hello`** (no template, greedy): `Hello Hello Hello Hello Hello Hello
  Hello Hello` (token 9419 repeated).

**Not coherent.** The no_think template fix is correct and kept (it matches the
GGUF template), but it does not resolve the incoherence — proving the template
was not the root cause.

## Next step (task #6)

Build a full-trunk numpy/torch reference that combines `qwopus_gqa_ref.py`
(GQA + FFN, MXFP4 dequant, bf16 truncation) + `qwopus_ssm_ref.py` (GDN scan) +
the SSM-block projections/conv1d/RMSNorm-gate/out_proj, reading **all 64
layers'** weights from `qwopus.fst`, computing pos0 and pos1 logits, and
comparing to the engine's `logs/head_dump/head_p0_logits.f32` (re-dump with
`FST_HEAD_DUMP=…/logs/head_dump FST_Q35_RAW_PROMPT=1 --prompt Hello`). The
first diverging tensor in that end-to-end comparison localizes the real bug.
Failing that, free disk quota and `pip install llama-cpp-python` to run the
GGUF directly for reference logits.