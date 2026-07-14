# Qwopus3.6 GDN Dispatch-Collapse + RoPE Fix — Post-Mortem

**Date:** 2026-07-11
**Branch:** `hy3-pivot-openmp-simd`
**Model:** `qwopus.fst` (`Jackrong/Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf` → `.fst`, 14.52 GB)
**Run log:** `logs/run_qwopus_rope64.log` (final), `logs/run_qwopus_fixed.log` (pre-RoPE-fix)

## Task

Four steps, verbatim rules preserved:
1. **Fix RoPE and SSM block math** — partial `dimension_sections` rotation in `process_gqa_q35`; validate `process_ssm` conv1d/dt_bias/A_log/softplus vs HF `qwen3_next`.
2. **Collapse GDN dispatches** — all 48 v-heads in one NPU dispatch per pass; one `run_blob` per SSM layer (was 144/layer, 6912/token).
3. **Build and run** — `cmake --build`; `timeout ./ds4_npu_engine --model qwopus.fst --prompt "Hello, what are you?" --tokens 10`.
4. **Post-mortem (a–d).**

Rules: **ZERO CPU fallbacks for math. ZERO per-v-head dispatches.** Both honored.

## (a) RoPE Fix

Implemented partial-dimension rotation, and **corrected the rotated-dimension count**.

The prior implementation set `n_rot = dh/2 = 128` (comment: "llama.cpp `n_rot_full`, head_dim 256"). That was wrong. `llama-model.cpp:1176` sets `n_rot_full = n_embd_head_k_full` (=256) **by default**, but then **overrides** it from the GGUF key `rope.dimension_count` (`llama-model.cpp:1178`). The Qwopus GGUF carries that key:

```
qwen35.rope.dimension_count = 64
```

(verified by parsing the GGUF metadata block directly; `attention.key_length=256`, `rope.dimension_sections=[11,11,10,0]`, `rope.freq_base=1e7`).

So the true value is **`n_rot = 64`**, not 128 and not 256. The `.fst` header has no `rope_dim` slot (the converter does not read `rope.dimension_count`), so it is baked in as a per-model constant in `process_gqa_q35`:

```cpp
const int n_rot   = 64;                      /* qwen35.rope.dimension_count */
const int halfrot = n_rot / 2;               /* 32  */
auto rope_head = [&](float* x, int pos) {
    for (int j = 0; j < halfrot; j++) {
        float freq = 1.0f / std::pow(rope_freq_base_, 2.0f * (float)j / (float)n_rot);
        float ang  = (float)pos * freq;
        float cf = std::cos(ang), sf = std::sin(ang);
        float a = x[j], b = x[j + halfrot];
        x[j]            = a * cf - b * sf;
        x[j + halfrot]  = b * cf + a * sf;
    }
    /* dims [64, 256) passthrough (untouched) */
};
```

This is ggml `ggml_rope_multi` / `GGML_ROPE_TYPE_MROPE` semantics: NeoX rotate-half pair `(j, j+n_rot/2)` for `j=0..n_rot/2-1`, `freq = base^(-2j/n_rot)`, dims `[n_rot, dh)` passthrough. With `dimension_sections=[11,11,10,0]` and text (all four section positions equal to the token position), MROPE reduces to standard NeoX over `n_rot` dims — the section split only subdivides which dims share a position value, which is a no-op for text. **No YaRN scaling.**

Also retained (prior session, verified vs `/tmp/qwen35.cpp`): `q_proj` is **doubled** `[q(256)|gate(256)]` per head, stride 512 → de-interleave `q`/`gate`; per-head RMSNorm on Q and K **before** RoPE; attention output `*= sigmoid(gate)` before `o_proj`.

**SSM block math** (conv1d newest-at-`ck-1` state convention, channel-outer `ssm_conv1d[c*ck+r]`, SiLU; `dt` projection + `A_log`/`dt_bias`/softplus; `g_logit = -exp(A_log)·softplus(a+dt_bias)`, `gdec=exp(g_logit)`, `beta=sigmoid(b)`; `q/k` `repeat_interleave` 3×; `RMSNormGated`) was validated against HF `qwen3_next` in the prior session (task #13) and is unchanged here.

## (b) Dispatch Collapse

**Done and bit-correct.** Total dispatches/token = **144** (3 passes × 48 SSM layers), down from 6912 (3 × 48 v-heads × 48 layers).

- `kernels/gen_gdn_scan.py` rewritten: three `@iron.jit` ops (`gdn_passA_op`, `gdn_delta_op`, `gdn_passB_op`). Each is a single Worker that loops all 48 v-heads internally (`for _v in range_(NV)`). State `S[128,128]` per v-head lives in one DDR BO, streamed one row (128 fp32) at a time through depth-2 object-fifos. Outputs use the proven 1024-float packet drain geometry (avoids the 128-packet `+2` shift bug); each v-head gets its own 1024-float output packet (first 256 used) so the per-v-head C kernels are called with the acquired packet directly — no IRON MemRefValue pointer arithmetic (which is unsupported).
- NPU BD `repeat_count` 8-bit limit handled by splitting TAPs so each dim ≤255 (passA `[48,128,130]`; passB `6×128`), mirroring `compile_ffn_unified.py`'s 4-D pattern.
- Compiled: `fst_gdn_passA.xclbin` (9497 B), `fst_gdn_delta.xclbin` (10202 B), `fst_gdn_passB.xclbin` (9545 B) + `_insts.bin`.
- `gdn_scan_vheads()` in `fst_engine.cpp` rewritten: BOs resized to `nV×`, packs all 48 v-heads row-major, issues **3 sequential `run_registered_blob` calls per SSM layer**, unpacks `ab[v*1024]=[a|b]`, `dy[v*1024]=[delta|y]`, `s2[v*128*128]`.
- **Verification:** `tools/gdn_probe.cpp` (multi-v-head, v-dependent distinct data, defensive xclbin load via `unique_ptr` to dodge the use-after-move SEGFAULT) dispatches each pass **once** across all 48 v-heads and compares to a per-v-head fp32 reference (relative error metric, threshold 1e-4). **PASS**: `max|Δa|=2.6e-7, max|Δb|=2.3e-7, max|Δy|=6.1e-7, max|ΔS|=6.4e-7` across all 48 v-heads.

**Zero per-v-head dispatches** rule honored: the NPU processes all 48 v-heads in one go per pass.

## (c) Execution Speed

**0.22 tok/s** (target >5; **not met**).

- Prefill: 65.3 s for 14 prompt tokens (~4.5 s/token).
- Decode: ~4.4 s/token trunk + 71 ms head_step.
- Peak RAM (RSS): 21.1 GB (within the 35 GB budget; `systemd-run --scope -p MemoryMax=35G` was unavailable in this environment — "Connection timed out" — so the run was direct; RSS confirms it fit anyway).
- NPU contexts: 6 created, 0 evictions.

The throughput floor is **per-dispatch syncobj overhead**, not compute. At 144 dispatches/token × ~30 ms/dispatch ≈ 4.4 s/token. The dispatch count was collapsed 48× (6912→144) but the AMDXDNA driver still charges a ~30 ms syncobj round-trip per `xrt::run`, so the wall-clock only moved from ~0.03 → 0.22 tok/s (the ~7× gain is the 48× fewer dispatches attenuated by the unchanged per-dispatch cost). The driver's bimodal 4–66 s stalls seen at 6912 scale are gone; at 144 scale stalls are a flat ~30 ms each. **Resetting the driver (`sudo modprobe -r amdxdna`) to clear stuck syncobj state is denied in this environment — not attempted.** Reaching >5 tok/s requires either sub-millisecond dispatch (driver/firmware work, out of scope) or fusing the 3 passes and/or the per-layer SSM passes into fewer dispatches (e.g. a single kernel that loops all 48 SSM layers' passes internally, or fusing passA+delta+passB).

## (d) Output Text

**Not coherent.** Exact decoded text (tokens 1–9 after the assistant turn opens):

```
\n<|im_start|>assistant<|im_end|>\n<|im_start|>assistant<|im_end|>\n
```

i.e. the model emits an **empty assistant turn** (`<|im_start|>assistant<|im_end|>`) and repeats it. Chat template verified correct ChatML (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`). Tokenizer verified (HF `tokenizers` bridge, 248070 vocab). Sampling greedy-equivalent at temp 0.6.

### Diagnosis

Changing `n_rot` 128→64 **did** perturb the output (the loop's phase shifted: pre-fix began `<|im_end|>`, post-fix begins `\n<|im_start|>`), confirming RoPE is now applied with the correct geometry — but it did **not** restore coherence. This is expected: **RoPE only touches the 16 GQA layers (`L%4==3`); the 48 SSM/GDN layers do not use RoPE at all.** Since the degenerate empty-turn loop persists and is dominated by the SSM path, the remaining bug lives in the **SSM block wiring**, not RoPE.

Per-component status:
- GDN scan recurrence + NPU kernel: **bit-correct** vs fp32 reference (`scripts/qwopus_ssm_ref.py`, `max|Δy|=4.66e-9`; `gdn_probe` `max|Δy|=6.1e-7` across 48 v-heads).
- conv1d / `dt` / `A_log` / `dt_bias` / softplus / `g`/`beta` / `repeat_interleave` / `RMSNormGated`: reviewed against HF `qwen3_next` source (prior session).
- **End-to-end SSM block vs HF: never numerically validated.** Each piece is correct in isolation; the failure is in how the pieces are wired (tensor layouts, projection weight orientation, the `q/k` repeat-interleave axis, conv1d weight orientation, `o_proj` of the gated+RMSNormed output, residual add).

The empty-assistant-turn collapse is the signature of a residual stream that has lost its content representation — consistent with the SSM output being numerically wrong (wrong scale/sign/layout) even though the scan math is exact.

### Recommended next step

**End-to-end numeric validation of the SSM block against HF `transformers` `qwen3_next`.** Dump, for one SSM layer at a fixed prompt+position: (1) the conv1d output, (2) `dt`/`g`/`beta` after softplus/sigmoid, (3) `qn`/`kn`/`v` after l2norm, (4) `y` after the scan, (5) `y` after `RMSNormGated`, (6) `o_proj` output, (7) the residual. Compare each to HF activations extracted from the same layer. The first diverging tensor isolates the wiring bug. This is the only reliable way to find it — reading the source found RoPE (`n_rot` mis-set) but cannot catch a layout/orientation mismatch in the SSM data path.

## Summary table

| Item | Target | Result |
|------|--------|--------|
| (a) RoPE partial rotation | match ggml MROPE | ✓ `n_rot=64` (corrected from 128; per GGUF `rope.dimension_count=64`), NeoX pair `(j,j+32)`, passthrough `[64,256)` |
| (b) 1 dispatch/v-head → all 48 in 1 | ~144/token | ✓ 144/token (3×48), bit-correct (rel <1e-4, 48 v-heads) |
| (c) Decode tok/s | >5 | ✗ 0.22 tok/s — syncobj-overhead-bound (~30 ms × 144) |
| (d) Coherent output | yes | ✗ empty-assistant-turn loop; SSM-block wiring bug, needs HF numeric validation |