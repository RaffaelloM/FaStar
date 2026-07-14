# GDN Scan → Engine Wiring & First Run (Postmortem, Steps 3-5)

**Status:** Steps 3-5 of the raw-MLIR-AIE GDN task are **DONE — wired, builds green,
runs end-to-end on hardware.** The GDN scan itself remains bit-correct (steps 1-2,
see `docs/GDN_SCAN_BITCORRECT_POSTMORTEM.md`). **But the surrounding SSM block is a
blind implementation and the run is NOT coherent**, and throughput is far below the
>10 tok/s target. This doc reports that honestly.

The five deliverables (a-e):

| # | Deliverable | Verdict |
|---|-------------|---------|
| a | Compile / tile-fit | ✅ DONE (prior). Row-streaming kernel, state `S[128,128]` in DDR BO, 128-fp32 rows streamed through depth-2 objectfifos — fits the 64 KB AIE2P tile. |
| b | HW bit-correct | ✅ DONE (prior). Probe `tools/gdn_probe.cpp`: max\|Δy\|=5.8e-10, max\|ΔS\|=1.5e-8. |
| c | Engine wiring (48 SSM layers) | ✅ DONE this session. `process_ssm` + `gdn_scan_vheads` + `process_gqa_q35` + `process_ffn_q35` + `generate_qwen35` + ChatML template + xclbin registration. Build green. 48 SSM layers × 3 dispatches/v-head × 48 v-heads = **6912 NPU dispatches/token**, all via `run_registered_blob` (zero-CPU scan). |
| d | Decode tok/s | ❌ **~0.03-0.5 tok/s** (2 decode tokens in ~70 s; trunk bimodal 4-66 s). Target was >10. |
| e | Exact output + coherence | ❌ **Output is `...  ` (ellipsis + spaces), degenerate, NOT coherent.** |

## What ran

```
XILINX_XRT=/usr FST_KERNEL_DIR=kernels \
  systemd-run --user --scope -p MemoryMax=35G ./build/ds4_npu_engine \
  --model qwopus.fst \
  --tokenizer …/Qwen3.5-9B-NPU2/tokenizer.json \
  --prompt "Hello, what are you?" --tokens 10 --temp 0.6
```

- **Load** (7.8 s): 65 layers (48 SSM + 16 GQA + 1 MTP), `lm_head` dequanted once to
  fp32 (5.09 GB resident), embed kept raw MXFP4 (675 MB), 6 permanent hw_contexts
  (under the 9 limit). RSS fits the 35 GB cgroup.
- **Prefill** (14 ChatML tokens): bimodal — tok 1 = 5.1 s (fresh state), toks 2-13 =
  ~13-65 s **each** (state-accumulated), tok 14 = 4.9 s. `prefill-last=1076`.
- **Decode**: `head_step` (fp32 lm_head argmax over 248320×5120, OpenMP) is **fast
  (75 ms)**. The **trunk** is the cost: tok 1 = 4.3 s, tok 2 = 65.7 s (bimodal stalls).
  Produced 2 decode tokens (both id 220) before the 600 s budget elapsed.

Measured throughput: 2 decode tokens in ~70 s effective decode time ≈ **0.03 tok/s**
(best-case single-token trunk ≈ 4.3 s → ~0.23 tok/s). Either way, far below 10.

## Why throughput is bad — the dispatch count, not the math

The GDN scan is on the NPU with **zero CPU fallback** (per the rule), but it is
**6912 sequential NPU dispatches per token**:

```
48 v-heads × 3 dispatches (passA, delta, passB) × 48 SSM layers = 6912 xrt::run/token
```

Each dispatch is a full `aiebu_assembler_get_elf` → `xrt::module` → `xrt::run` →
`wait()` → `bo.sync` round-trip. The blob is hash-cached so the ELF isn't rebuilt
every time, but the XRT dispatch + syncobj wait + 3 BO syncs per dispatch still
dominate. At ~0.5 ms/dispatch that is ~3.5 s/token floor; the observed 4-66 s
bimodal spread is AMDXDNA **syncobj stalls** stacking across thousands of sequential
dispatches (and likely aggravated by NPU context state left from earlier
timeout-killed runs — `fst_main` calls `_exit(0)` to avoid exactly this, but a
SIGTERM from `timeout` bypasses it). This is the same dispatch-count ceiling the
HY3 work hit; the "1-dispatch/layer" Phase-2 target is **not met** — it is 6912.

`head_step` is NOT the bottleneck (75 ms) — the fp32 lm_head argmax with OpenMP is
fine. The host MXFP4 matvecs (`mxfp4_matvec_f32`, fused dequant+dot, no scratch) are
also not dominant. The trunk cost = NPU dispatch overhead × 6912.

## Why output is incoherent — blind SSM block (the real blocker)

The GDN recurrence math is bit-correct in isolation, but the SSM block around it
was written to the transformers contract from a `WebFetch` read, **never validated
against HF `transformers`**. The output (`...` then spaces, token 220 = `Ġ` = space
repeated) is degenerate. At least one of these blind spots is wrong:

1. **RoPE partial rotation (flagged, most likely culprit).** `qwen3.5-next` uses
   `rope.dimension_sections` — partial-rotation splits
   `rope_sections=[11,11,10,0]` over the 256-dim head (32-dim no-op tail). The wired
   code applies **full-head NeoX rotate-half** over all 256 dims. This mis-rotates
   every GQA Q/K and corrupts attention. (`process_gqa_q35`, `fst_engine.cpp:4989`.)
2. **conv1d state convention.** Newest-at-index-0 causal conv1d + SiLU
   (`process_ssm:4787`). Plausible but unverified against `F.conv1d(..., padding=...)`.
3. **RMSNormGated eps.** Uses `rms_eps` (1e-6); transformers may use a different eps
   for the per-v-head group norm (`process_ssm:4849`).
4. **dt_bias / A_log / softplus.** `g_logit = -exp(A_log)·softplus(a+dt_bias)`
   (`process_ssm:4815`); numerically soft but unverified.

The recurrent kernel being bit-correct proves the scan is right; the incoherence is
**upstream/downstream of it**, in the block we did not validate. Fixing RoPE to honor
`dimension_sections` is the first thing to try.

## What was built this session (step 3 wiring)

- `include/fst_engine.h`: `ARCH_QWEN35` path — `Q35LayerWeights` (large projections
  kept raw MXFP4 `std::vector<uint8_t>`, small F32/BF16 kept), `Q35SSMState` (matrix
  `S[nV·128·128]` + conv buffer `[ck·Cdim]`), 6 reuse GDN BOs
  (`q35_bo_spktA_/ab_/din_/dout_/spktB_/s2_`), `q35_embed_raw_`,
  `q35_lm_head_` (fp32), `q35_final_norm_`, `rms_eps`.
- `src/fst_engine.cpp`:
  - `mxfp4_matvec_f32` — fused MXFP4 dequant+dot, no scratch (avoids 69 GB/token
    full-dequant; the 27 B model stays raw MXFP4 ~14.5 GB, dequant on demand).
  - `load_qwen35_shared_weights` — parses `TID_Q35_CFG` (23 F32) +
    `TID_Q35_LAYER_TYPES`, loads raw MXFP4 + F32/BF16, dequants `lm_head` to fp32
    ONCE (~5 GB), embed stays raw.
  - `process_ssm` (kind 0): rmsnorm → qkv/gate/alpha/beta matvecs → depthwise causal
    conv1d + SiLU → split → g_logit/beta → l2norm+repeat_interleave →
    `gdn_scan_vheads` (NPU) → RMSNormGated → ssm_out matvec → residual.
  - `gdn_scan_vheads`: lazy-alloc 6 BOs, per-v-head loop builds spktA→passA→a,b;
    din→delta→delta,y; spktB→passB→S2; persists. `run_registered_blob` (zero-CPU scan).
  - `process_gqa_q35` (kind 1): rmsnorm → q/k/v matvecs → per-head Q/K RMSNorm →
    RoPE (full-head rotate-half, **flagged**) → KV cache append (bf16) → causal GQA
    attention → o_proj → residual. Host-only (no NPU dispatch).
  - `process_ffn_q35`: rmsnorm → gate/up → silu(gate)·up → down → residual.
  - `generate_qwen35`: zero SSM/conv/KV state → embed row-dequant → token-by-token
    M=1 prefill+decode → `head_step` (rmsnorm + fp32 lm_head argmax + sample) →
    `run_trunk_token` dispatches kind 0→`process_ssm` else `process_gqa_q35`, then
    `process_ffn_q35`.
  - `generate()` dispatch: `if (config_.arch == ARCH_QWEN35) generate_qwen35(...)`.
  - `reset_session()`: zeros S/conv, resets KV `n`.
  - xclbin registration under `ARCH_QWEN35`: `q35_gdn_passA/delta/passB`
    (`fst_engine.cpp:1675`).
- `src/fst_main.cpp`: `ARCH_QWEN35` Qwen3 ChatML template
  (`<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n`), `FST_Q35_RAW_PROMPT`
  bypass.
- Fix: guarded the unconditional `host_embedding_table_[0..3]` embed-preload debug
  print (it segfaulted on QWEN35, which uses `q35_embed_raw_` instead of the bf16
  table).

## Tokenizer note

Used `/home/raffaele/.config/flm/models/Qwen3.5-9B-NPU2/tokenizer.json` (vocab 248070)
against a model vocab of 248320. Safe: all prompt/decode ids (< 248070) fall under
both, so embed/lm_head lookups are in-range; the 248320-row `lm_head` has ~250 extra
dummy rows. The model's own tokenizer.json was not present locally. Decoded tokens:
220 → `Ġ` (space), 1076 → `...`, 271 → `\n\n`.

## Honest summary

The hard, previously-blocking milestone — a bit-correct NPU GDN scan with state in
DDR, **zero CPU fallback** — is done and wired into a full 65-layer hybrid engine
that runs on hardware within the 35 GB cgroup. The wiring is complete and the
build is green. **But the run does not meet the Phase-2 targets**: throughput is
~0.03-0.5 tok/s (target >10; the floor is the 6912-dispatch/token overhead, not the
scan math), and the output is degenerate/incoherent (the surrounding SSM block —
RoPE `dimension_sections` above all — is unvalidated and at least one piece is
wrong). Next steps: (1) honor `rope.dimension_sections` in `process_gqa_q35`,
(2) validate the SSM block against HF `transformers` recurrent (like
`scripts/qwopus_ssm_ref.py` did for the scan), (3) collapse the 6912 dispatches —
batch v-heads into one passA/delta/passB dispatch (the kernels are per-v-head today;
a multi-v-head or per-layer fused blob would cut dispatches ~6912→~3/layer).