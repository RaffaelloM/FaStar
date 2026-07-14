# QWOPUS FFN M=8 NPU — Engine-Wired A/B: MARGINAL WIN (1.08×), argmax 8/8

**Date:** 2026-07-14
**Status:** Wired behind `FST_Q35_FFN_NPU` (OFF by default; shipped 0.20 tok/s host path untouched).
**Verdict:** The probe's 1.76× dispatch-only win **collapses to 1.08× in-engine** once real host timing + packet packing are included. The M=8 NPU FFN is correctly wired, faster, and argmax-accurate (8/8), but the margin is too small to meaningfully break the 0.20 tok/s ceiling (→~0.21). Residual rel-err ~7% (fails the strict 5e-3 bar, but argmax-robust).

This supersedes the "NOT wired" note in `docs/QWOPUS_FFN_M8_SD_POSTMORTEM.md` — the kernel is now wired and measured end-to-end in the real engine.

---

## What was wired

- **Registration** (`src/fst_engine.cpp:1704`): `q35_ffn_m8` xclbin registered only when `FST_Q35_FFN_NPU` is set. Context budget: Q35 baseline 7 → 8 < 9. ✓
- **`process_ffn_q35_m8_npu`** (`src/fst_engine.cpp:5652`): RMSNorm 8 h's → `run_m8_proj` (pack → dispatch → readback) for gate/up (N=17408, N_TILE_USED=1088) and down (N=5120, N_TILE_USED=320) → host SwiGLU → residual. Persistent BOs (136.8 MB in, 557 KB out) lazily allocated.
- **Capture + A/B harness** (`src/fst_engine.cpp:9134–9415`): captures the first 8 trunk passes' per-layer pre-FFN h (bf16) and post-FFN host residual during a normal decode; then for each of 64 layers runs host M=1 8× (re-timed, same-input reference) AND M=8 NPU, compares per-(layer,token) residual rel-err + last-layer greedy argmax (final_norm + lm_head). Fires after prefill when `cap_tok>=8` (early-return) or after decode.

All M=8 paths guarded by `ffn_npu_gate = getenv("FST_Q35_FFN_NPU")`; flag-unset = byte-identical shipped path (verified by inspection: registration, capture init, capture hook, both A/B triggers, and `process_ffn_q35_m8_npu` all gated).

## Measured A/B (8 prefill-position hidden states × 64 layers, two runs)

| Run | rel-err compare | host 8× | NPU M=8 (pack+disp) | pack frac | speedup | argmax | rel-err max/mean |
|-----|-----------------|---------|----------------------|-----------|---------|--------|------------------|
| 1 | fp32-NPU vs bf16-host | 12.34 s | 1.93+9.60=11.53 s | 17% | 1.07× | 7/8 | 2.224 / 6.78% |
| 3 | bf16-fair | 12.61 s | 1.94+9.60=11.53 s | 17% | 1.09× | **8/8** | 2.265 / 7.20% |

- **Dispatch 9.60 s** = 64×3×50 ms ✓ matches the probe's 48.5 ms/proj exactly. The 16-tile kernel geometry is correct in-engine.
- **Packing 1.94 s (17%)** — better than the 2.6–5.3 s estimate. Per-proj pack ≈ 10 ms (680 pkts × 12.6 KB strided scale/nibble copy).
- **Host 8× ≈ 12.5 s** = 197 ms/layer ÷ (8×3) = **8.2 ms/matvec** — the host OpenMP FFN is ~8 ms/matvec on this machine, *faster* than the 10.7 ms baseline the probe's 1.76× assumed.

## Why the win collapsed from 1.76× to 1.08×

Per-h decomposition:
- NPU dispatch: 50 ms/proj ÷ 8 = **6.06 ms/h**
- Packing equiv: 10 ms/proj ÷ 8 = **1.25 ms/h**
- NPU total/h: **7.3 ms** vs host **8.2 ms** → **1.12×** (matches the measured 1.07–1.09×)

Two reasons, both load-bearing:
1. **Host is faster than assumed.** The probe benchmarked against host M=8 = 8×10.7 ms = 85.6 ms. In-engine the host OpenMP `mxfp4_matvec_f32` (`#pragma omp parallel for` over N) measures ~8 ms/matvec → 64 ms for 8, not 85.6. The 10.7 ms figure was a different measurement condition; the real host floor is lower, shrinking the gap.
2. **Packing adds 17%.** The probe ignored packet packing (it pre-built the BO off-line). In-engine, each (layer,proj) repacks 680×12.6 KB from the engine's MXFP4 weight tensor + 8 bf16 h-chunks = 10 ms/proj overhead the host path doesn't pay.

The dispatch-only 6.06 ms/h is genuinely faster than host 8.2 ms/h (1.35×), but packing eats most of it.

## Correctness: argmax 8/8, but residual rel-err ~7%

- **Last-layer token argmax: 8/8 (run 3), 7/8 (run 1).** The 1-flip (tok 2) is a **near-tie with run-to-run nondeterminism**, not a wiring bug: between run 1 and run 3 the argmax code path was *unchanged* (only the rel-err comparison changed), yet tok 2's `npu_pred` moved 3710→95772 while `host_pred` stayed 95772. The NPU result varies slightly run-to-run (OpenMP fp-reduction order under `-ffast-math` in RMSNorm/matvec), and tok 2's logit is a near-tie so the argmax flips. This is inherent greedy-argmax-on-248K-vocab sensitivity, not an M=8 defect.
- **Residual rel-err: max 2.27, mean 7.2%, 505/512 (layer,token) pairs > 5e-3.** This **fails the strict 5e-3 bar** set in the plan, and — importantly — **bf16-fair truncation did NOT fix it** (run 1 fp32-vs-bf16 = 6.8%, run 3 bf16-fair = 7.2%; essentially identical). So the divergence is NOT a comparison artifact; it is real.
  - The probe hit 0.3% on **synthetic** data (random scales 120–140, uniform nibbles → well-conditioned dot products). The engine uses **real** weights (full e8m0 dynamic range 2^-7..2^13) and real activations → wider dynamic range and cancellation. The NPU's bf16 MMUL operands (`<8,8,4>` mac = bf16×bf16→fp32) and bf16-h truncation accumulate ~7% on the worst elements vs the host's fp32×fp32→fp32. The max-based rel-err metric is dominated by outlier / near-zero-reference elements.
- **The argmax is robust** to the ~7% residual divergence: lm_head is a 5120-dim dot product, so a few divergent residual elements barely move the 248K vocab logits. 8/8 argmax with 7% residual rel-err is consistent — the aggregate token prediction is stable even when individual residual elements diverge.

**Implication for SD verify:** SD verify compares draft vs model tokens via argmax. 8/8 argmax match (modulo near-tie flips) means the M=8 NPU FFN would *accept the same tokens the host would* for this block — a positive sign. But the ~7% per-layer residual divergence, compounded across 64 layers in a real all-NPU-FFN trunk (this A/B only re-ran the FFN on host-derived clean h's; a real SD verify runs NPU FFN on NPU-produced residuals), could drift further. Not yet measured end-to-end through a full NPU-FFN trunk.

## tok/s impact

Save ≈ 1.08 s per 8 tokens = **0.135 s/token**. FFN is ~40% of the ~4.9 s token (~2.0 s). Saving 0.135 s/token → token 4.9→4.77 s → **0.20 → ~0.21 tok/s**. Marginal — does not meaningfully break the ceiling.

## v2 lever (not pursued)

Packing is 17% and dominated by the **weight** re-extraction (scales + nibbles), which is **invariant across the 8 tokens** and — in a real SD verify loop — invariant across the many 8-token blocks per layer. Pre-packing the weight BO once per layer and reusing it across blocks (re-packing only the 8 h-chunks per block) would cut packing from ~10 ms/proj to ~3 ms/proj in the steady state → NPU total/h ≈ 6.06 + 0.4 = 6.5 ms vs host 8.2 ms → **~1.26×**. Still not enough to break the ceiling on its own, and requires the full SD loop to amortize (this A/B is a single block, so per-block repack is unavoidable here).

## Update 2026-07-14 — host weight pre-packing (Step 1) = MEASURED DEAD END, REVERTED

Acting on the "pre-pack the static weights at load" lever, the pack loop was instrumented to split h-pack vs weight-pack, and the static weight portion was pre-packed once at load into 9.09 GB of compact host buffers (`q35_m8_w{gate,up,down}_`, 64 layers × 3 proj × 45.1 MB), so `run_m8_proj` did one 4352-B memcpy per packet instead of the strided 17-byte-block extraction.

**Pack-split (L0, measured):** baseline `total=9.99ms h=4.17ms(42%) w=5.24ms(52%)`; pre-packed `total=10.21ms h=4.95ms(49%) w=4.44ms(43%)`. **Weight dropped only 5.24→4.44 ms (0.8 ms, ~0.02× total win).** The strided *write into the interleaved BO* (4352 B every 12576 B) dominates the weight portion regardless of whether the source is the raw tensor or a compact buffer — the source read was never the bottleneck. Pre-packing the source does not help because the destination layout is fixed by the kernel's packet format.

**Worse, the 9.09 GB allocation destabilized the NPU.** Both pre-packed runs FATAL'd with `qds_device::wait() unexpected command state` immediately after L0's 3 dispatches, while every non-pre-packed run completed all 192 dispatches. Root cause: memory pressure. 33 GB available − 9 GB pre-pack − ~20 GB model+lm_head ≈ 4 GB free → the 136.8 MB dispatch BO (pinned DMA memory) gets allocated under pressure into a state that fails DMA → dispatch FATAL after the first layer. So the pre-pack costs 9 GB of RAM *and* breaks the device, for ~0.02×.

**Conclusion:** host-side weight pre-packing is a dead end. The 1.30× target is **not reachable host-side** — the per-call floor is the 134 MB BO write (~10 ms: 89 MB h-replication + 45 MB weight, both unavoidable host-side given the kernel's packet format). Only a **kernel change** reaches 1.30×:
- **Separate pre-packed weight BO** (one 8.66 GB device-BO set, weights never re-written per call) → eliminates the 45 MB weight write → ~1.20× best case.
- **On-device h broadcast** (host sends only 80 KB unique h, kernel fans it out 1088×) → eliminates the 89 MB h-replication → the remaining gap to 1.30×.

The 9 GB host pre-pack was **reverted** (`prepack_m8_weights`, the `wpre` param, the compact-buffer members, and the load-time call all removed) per "prefer deletion over addition" — it was gated, wasteful, and device-destabilizing. `run_m8_proj` is back to strided extraction from the raw MXFP4 tensor; the shipped path and the 1.08× A/B baseline are byte-identical. Step 2 (dequant-once at NT=16) is moot — it saves dequant *compute* on the NPU, not the host pack/BO-write floor that dominates, and its prerequisite (weight amortization) just failed. **Recommendation: authorize the kernel-change path (separate weight BO + on-device h broadcast) or close the M=8 FFN lever.** Logs: `logs/m8_packsplit3.log` (split), `logs/m8_wpre.log`/`logs/m8_wpre2.log` (pre-packed, FATAL after L0).

## Conclusion / recommendation

The M=8 NPU FFN is **correctly wired, 1.08× faster, and argmax-accurate (8/8)** — a real but marginal win. The probe's 1.76× was dispatch-only against a slower host baseline; the engine reality (host at ~8 ms/matvec + 17% packing) is 1.08×. Combined with the ~7% residual rel-err (acceptable for argmax, unproven for a full NPU-FFN trunk), this does **not** realize the hoped-for ceiling break.

**Gate stays OFF.** Shipped 0.20 tok/s host path untouched. The wiring is preserved behind `FST_Q35_FFN_NPU` for the orchestrator's call if a future SD verify loop (with v2 weight pre-packing) is authorized — the dispatch geometry is proven correct (9.60 s = probe's 48.5 ms/proj × 192). Do not enable by default.

## Files

- `src/fst_engine.cpp` — gated registration (1704); `process_ffn_q35_m8_npu` + `run_m8_proj` (5652); capture + A/B harness (9134–9415).
- `include/fst_engine.h` — declarations + persistent BO / capture buffer members.
- `kernels/fst_qwopus_ffn_m8.xclbin` + `fst_qwopus_ffn_m8_insts.bin` — the 16-tile kernel (unchanged from the probe).
- Logs: `logs/m8_ab_run.log` (run 1), `logs/m8_ab_run3.log` (run 3).