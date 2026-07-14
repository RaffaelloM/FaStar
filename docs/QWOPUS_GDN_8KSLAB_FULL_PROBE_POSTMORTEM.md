# QWOPUS GDN ≤8 KB-Slab Full-Recurrence Probe — POSTMORTEM

**Date:** 2026-07-14
**Status:** MEASURED — the empty cell is filled. Verdict: **FAST (viable)** by the
pre-set decision rule, with a caveat (see below). The ≤8 KB stack does **not**
collapse to the ≥16 KB ~1.3 µs/vec degradation under the real K=8 × 48-v-head
recurrence. This unblocks the 4-tile bf16-S engine kernel for the orchestrator's
authorization.

Companion analysis: `docs/QWOPUS_GDN_8KB_SLAB_ANALYSIS.md` (the structural argument +
the empty cell). Artifacts: `kernels/fst_gdn_8kslab_full_kernel.cc`,
`kernels/gen_gdn_8kslab_full.py`, `tools/gdn_8kslab_full_probe.cpp`,
`logs/gdn_8kslab_probe.log`.

## The question (the empty cell)

Reconstructed test matrix (from `qwopus-chunkwise-gdn-2tile-stack-deadend`):

| design | S size | storage | full recurrence? | result |
|---|---|---|---|---|
| 1-tile 32 KB Buffer | 32 KB | Buffer | yes | 4878 ms (slow) |
| 8-tile 4 KB Buffer | 4 KB | **Buffer** | yes | 1094 ms, bit-correct (slow) |
| 2-tile 16 KB stack | 16 KB | stack | yes | 2153 ms (slow, ≥16 KB threshold) |
| 4-tile 8 KB micro | 8 KB | **stack** | **NO — toy, 1 recur step** | 5.87 ms (fast — not representative) |

**Empty cell:** ≤8 KB *stack* on a *full* recurrence. The only fast result was a toy
one-step; every full-recurrence test used ≥16 KB stack (over threshold) or persistent
Buffer (slow at any size). The 8-tile 4 KB was slow *because it was Buffer, not
because 4 KB is slow*. **This probe fills the cell.**

## Probe design

1 tile, 32-col bf16 column-slab, `S` row-major `[128][32]` bf16 = **8 KB stack**,
single-call internal loop over **48 v-heads × K=8 sequential tokens** (the real M=K
recurrence — `delta_t` depends on `S_t`). Body = the real GDN 3-pass math reused
verbatim from the bit-correct 8-tile / 2-tile kernels (`passA` a/b reductions →
`delta`/`y` → `passB` RMW of the 8 KB stack in place, store bf16). A `noop` variant
(same 8 KB stack alloc + tiny DMA, no recurrence) isolates the stack-alloc + DMA +
loop floor for subtraction. Probe inputs (`kn/qn/v/gdec/beta/S0`) are deterministic
in-kernel values so DMA does not confound the stack-RMW latency — the recurrence
*structure* is real; only the input *values* are synthesized. The stack-access
pattern (and thus the measured latency) is identical to a real-data engine kernel.

Shim discipline: 1 MM2S + 1 S2MM (the probe streams only a 48-float in/out for the
checksum; S lives on the stack). `stack_size=0x3000` (12 KB) — 8 KB S + ~1.5 KB
recur scratch fits; `0x5000` overlaps the ObjectFifo buffer at 0x4000.

## Measured result (hardware, best of 5 + 1 warmup)

```
full_ms    = 80.55 ms
noop_ms    =  0.82 ms   (stack-alloc + DMA floor — negligible)
recurrence = 79.73 ms   (full − noop)
S-vec accesses (passA-read + passB-write per token) = 393216
per-vec S-stack latency = 202.77 ns/vec
checksum: finite 48/48, range [-1.895, 2.573] (recurrence ran, varies per v-head — OK)
```

`logs/gdn_8kslab_probe.log`.

## Verdict

**FAST, by the pre-set rule (< 250 ns/vec → viable):** 202.77 ns/vec.

The 8 KB stack does **not** degrade to the ≥16 KB ~1.3 µs/vec (1300 ns/vec) regime
under the real K=8 × 48-v-head recurrence. The empty cell resolves in favor of the
slab structure: ≤8 KB stack on a full loop is fast, distinct from the ≥16 KB / Buffer
slow designs. The "≤8 KB ⇒ slow" link that load-bore the prior impasse **does not
hold** — 4 KB Buffer was slow *because Buffer*, and 16 KB stack was slow *because
≥16 KB*; 8 KB stack on the real loop is neither.

### Caveat the orchestrator should weigh

202.77 ns/vec is **3.4× the toy's ~60 ns/op**, not the optimistic ~60 ns case. It
clearly clears the 250 ns viability bar (and is 6.4× under the 1300 ns degradation
regime), so the slab does not collapse — but it is not the toy's flat rate either.
The 4-tile engine kernel's per-layer cost should be estimated from **79.73 ms
recurrence / 48 v-heads = 1.66 ms per v-head per tile**, with 4 tiles each covering
one 32-col slab in parallel = ~**80 ms/layer for the K=8 recurrence** (the 48 v-heads
are the per-tile work; 4 tiles parallelize across the 128 cols, not the v-heads).

Rough M=K vs shipped comparison (for the orchestrator's decision, **not** a claim):
- Shipped GDN = 3 dispatches × ~30 ms syncobj floor ≈ 90 ms **per token** → K=8
  tokens sequentially ≈ 720 ms.
- Slab M=K = one dispatch, K=8 recurrence ≈ **80 ms** for all 8 tokens.
- That is the dispatch-amortization win the analysis predicted — **but only if** the
  80 ms fits under the syncobj floor in the real engine (the 3pass-onekernel dead-end
  showed serializing on-tile work can "expose" ~480 ms; here ≤8 KB stack keeps it at
  ~80 ms, not 480 ms). Whether 80 ms is a net win vs the shipped path's pipelined
  floor is the engine-kernel question — untested, gated on orchestrator sign-off.

## What this does NOT prove

- It does **not** prove the 4-tile engine kernel is faster than shipped end-to-end.
  It proves only the **latency primitive** (≤8 KB stack RMW under the real
  recurrence) is fast — the one assumption that was untested. Dispatch-amortization,
  bf16-S correctness on real data, and real-engine integration remain untested.
- bf16-S was A/B-lossless for argmax on the *shipped row-streaming* path; the slab
  path's bf16-S rounding over K=8 sequential RMW is a separate (lower-risk) question.

## Recommendation (per Plan 2 Step 4)

Per the pre-set decision rule, **the slab structure is VIABLE** → the next step the
orchestrator authorized ("we will authorize the full 4-tile bf16-S slab kernel for
the engine") is in order. **Not wired** — no engine change made. Standing gates
unchanged (FST_Q35_FFN_NPU OFF, shipped 0.20 tok/s host path untouched).

Suggested follow-up if authorized: build the 4-tile bf16-S slab engine kernel
behind a new `FST_Q35_GDN_8KSLAB` gate, real-data (not deterministic) inputs, 4 tiles
× 32-col slabs = 128 cols, 2 MM2S + 2 S2MM per tile, real `c = kn·qn` host- or
scalar-tile-broadcast, and A/B it vs the shipped 3-pass row-stream for both
bit-correctness and per-layer latency.