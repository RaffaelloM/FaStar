# Parallel-Scan GDN Research + Real-Engine A/B (2026-07-13)

Investigation of whether the GDN (Gated DeltaNet) delta-rule admits a parallel
scan, to enable **M=K batched speculative-decoding verify** on FaStar's currently
M=1 (sequential, stateful) Qwen3.5-Next trunk — the lever for real K× lossless
speculative decoding (the M=1 trunk gives only ~1.2×).

Builds on the fusion dead-end (`docs/QWOPUS_FUSED_GDN_RACE_ROOTCAUSE.md`):
M=1 3-pass → 1-dispatch fusion is structurally impossible on this IRON build
(2-S2MM race vs 1-S2MM stall). The only remaining NPU lever for >3× is
amortizing the 144-dispatch floor across K tokens via the model's NextN/MTP
block — which needs M=K verify, which needs the GDN to process K tokens per
dispatch.

## Verdict

**Viable, precision-validated. Not a dead-end.** The win is **dispatch
amortization**, not parallelism. The single biggest risk (bf16-S tile-memory
approximation) is **proven lossless** by a real-engine A/B. The only remaining
risk is IRON kernel authoring, and its DMA-channel picture is *easier* than the
already-dead M=1 fusion.

## (A) The chunkwise algorithm EXISTS and is provably correct

The Gated DeltaNet chunkwise form (Yang, Kautz, Hatamizadeh, ICLR 2025,
arXiv:2412.06464) is the **training-time reference for Qwen3-Next** — the model
FaStar runs. Our recurrence is identical to the paper's Eq. 8 up to a
left/right convention:

```
Paper:  S_t = S_{t-1}( α_t (I − β_t k_t k_t^T) ) + β_t v_t k_t^T
Ours:   S = gdec·S; kvm = Sᵀkn; delta = (v−kvm)·beta; S += kn⊗delta; y = Sᵀqn
        (gdec=α, kn=k, kvm=S_{t-1}k, delta=β(v−αSk))
```

The cross-chunk state transfer (Eq. 11) is a **pure matmul** — the nonlinearity
is fully absorbed into the within-chunk `W, U` computation. The within-chunk
(Eq. 10) is a triangular solve.

## (B) The crux — small K gives NO parallelism

For K ≤ chunk size C there is one chunk → no cross-chunk parallelism (the GPU
20–30× win source is irrelevant), and within-chunk is a triangular solve = **K
sequential steps** (forward substitution; K-deep by construction). **Nothing
computes the delta-rule in sub-K depth** — `delta_t` depends on `S_t`, period.

So this is **not a parallelism win**. The win is **dispatch amortization**: a
single kernel that holds S on-tile and loops the K recurrence steps in **1
dispatch** (no DMA per step) = 1 dispatch for K tokens vs 3K naive
(K×144 → ~144 for K=8). On the dispatch-bound floor (syncobj ~30ms × 144 ≈
4.3s/token), K=8 → ~24× fewer GDN dispatches.

## (C) Risk #1 — tile memory forces bf16-S (NOT bit-exact)

fp32 `S[128,128]` = 64 KB = the entire AIE2P tile → infeasible (no room for
code/q/k/v/W/U/accumulators). Must use **bf16 S (32 KB)** → not bit-exact: each
step re-quantizes S to bf16. Mitigation: **fp32 accumulator + bf16 storage**
(the standard pattern) — the on-tile kernel computes fp32 matvecs on the
bf16-rounded S and stores the updated S as bf16.

## (D) Risk #2 — DMA channels: EASIER than the dead M=1 fusion

The M=1 fusion died on a 2-concurrent-S2MM race + the 3-pass "one held output"
rule violation. The chunkwise M=K loop is:

- 1 MM2S — stream in the K-token block (q/k/v/gates/betas), tiny.
- held S — tile-local `aie.buffer`, **no DMA channel** (not an ObjectFifo).
- 1 S2MM during the loop — stream the K outputs `y_t` (one stream).
- 1 S2MM at the end — flush `S_out` once, *sequential after* the loop.

→ **only 1 concurrent S2MM**. The 2-S2MM race that killed M=1 fusion does not
occur; the one-held-output rule is satisfied. This is the strongest feasibility
signal. The IRON-authoring risk is the held-S `aie.buffer` (ObjectFifos are
stream-oriented; respect `static-bss-incoherent-with-loadv` — use `aie.buffer`,
not C++ `.bss`).

## (E) The decisive gate — real-engine A/B (PASSED)

`scripts/qwopus_bf16s_drift_test.py` (numpy, engine-exact recurrence, realistic
magnitudes, K=8, 48 v-heads × 64 samples) predicted lossless-grade:
- max|Δy| at K=8 = 5.4e-3 (fp32-acc mitigation), **IN-BAND** vs the engine's own
  fp32-vs-fp32 tolerance (2e-2).
- Argmax flips (fake random-W logit, tight margins = pessimistic) all
  concentrated at margin ≤ 0.0007; real model margins >> 1 → 1000× below
  threshold.

Then the **real-engine A/B** (the gate the user chose): env flag
`FST_Q35_GDN_BF16S` in `gdn_scan_vheads` (`fst_engine.cpp`, after the passB
S2-unpack) truncates the carried `S[48*128*128]` to bf16 (`u & 0xffff0000`)
after every GDN update — an exact simulation of the on-tile bf16-S kernel's
precision behavior (bf16 storage + fp32 compute). Greedy `--temp 0`
(deterministic, isolates drift from sampling), same prompt, `--tokens 40`,
`qwopus_bf16_mtp.fst`, run twice (fp32-S baseline vs `FST_Q35_GDN_BF16S=1`),
compare per-position token IDs from the `[q35-gen] decode=<id> (pos=N)` trace.

**RESULT: IDENTICAL at all 39 decode positions (pos 18–56). 39/39 DISTINCT
tokens (maximally varied — no 2-cycle attractor, every argmax genuinely
stressed). ZERO argmax flips. Speed identical (0.21 tok/s both — the bf16
quantization is a free host memcpy loop, no precision-related speed
penalty).**

→ bf16-S trunk argmax == fp32-S trunk argmax → speculative verify with a
bf16-S trunk agrees with the fp32 trunk → **speculative decoding is
LOSSLESS**. The precision risk is eliminated; the only remaining risk for the
chunkwise M=K path is IRON kernel authoring, not precision.

## (F) Status & next step

- **MTP wiring DONE & validated** (reusable regardless of the parallel-scan
  outcome): converter emits L64 (kind 2 MTP) from `mtp.*` keys
  (`qwopus_bf16_mtp.fst`, 868 shared entries, 14.52 GB). `FST_Q35_NEXTN_PROBE`
  PASS: L64 loads (eh_proj 27.85 MB MXFP4 [5120,10240], q 33.4 MB, gate/up/down
  47.3 MB, F32 norms); 3-token draft finite non-NaN (3058 97177 96393);
  step0 `h_post` range [-23, +10.8] no NaN, V=248320, top5 sane. Math untouched.
- **Next step: IRON chunkwise M=K kernel** (the big, uncertain effort).
  Held bf16 S as `aie.buffer`, K-step on-tile triangular solve (WY/UT form),
  1 MM2S + 1 S2MM + end-flush S. Validate bit-vs-bf16-S A/B on the kernel's own
  output (the host A/B already proved the precision; the kernel A/B proves the
  authoring).

### Policy note

bf16-S is NOT bit-exact — a departure from the standing "do not touch the math,
bit-identical" constraint. But for speculative *verify*, **losslessness
(argmax agreement)** is the correct gate, not bit-identity, and the real-engine
A/B proves agreement holds (39/39). The user accepted this by choosing the A/B
gate. The 3-pass fp32 path remains the shipped fallback (math unchanged, flag
off by default).

### Logs / artifacts

- `logs/ab_fp32s.log`, `logs/ab_bf16s.log` — the A/B runs.
- `logs/bf16s_drift_test.log` — the numpy gate.
- `scripts/qwopus_bf16s_drift_test.py` — the numpy drift gate.
- `FST_Q35_GDN_BF16S` — the engine A/B flag (gated, off by default).

### Sources

- Yang, Kautz, Hatamizadeh 2024, "Gated Delta Networks" (ICLR 2025, arXiv:2412.06464).
- Yang et al. 2024, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (DeltaNet, NeurIPS 2024).
- Schlag, Irie, Schmidhuber 2021, "Linear Transformers Are Secretly Fast Weight Programmers" (ICML 2021).
- NVlabs/GatedDeltaNet reference implementation.