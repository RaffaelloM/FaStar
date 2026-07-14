# Qwopus3.6 on FaStar — Orchestrator Handoff Report

**Date:** 2026-07-13
**Model:** Qwopus3.6-27B (dense Qwen3.5-Next: 48 GDN SSM + 16 GQA + 1 MTP, hidden=5120, inter=17408, vocab=248320)
**Target:** ~10× over the shipped 0.20 tok/s (i.e. ~2.1 tok/s), matching FastFlow's 20 tok/s on Qwen3.5-35B-A3B (MoE).
**Status:** 10× is **structurally unreachable** in-constraint. The realistic ceiling with all safe levers is **~0.3–0.5 tok/s**. This report maps every approach tried, where and why each failed (with measured evidence), the real bottleneck profile, and the precise levers/forks an orchestrator can drive with further prompts.

---

## 1. Scope of the job

FaStar runs DeepSeek/Hunyuan/Qwen dense+MoE models on the AMD Ryzen AI 9 365 (XDNA2 NPU / AIE2P) by splitting execution across SSD, RAM, and NPU. The current pivot is **Qwopus3.6-27B dense** (Qwen3.5-Next), shipped bit-correct at **0.20 tok/s** (greedy decode, `qwopus_bf16_mtp.fst`).

**Hard constraints (must hold across all work):**
1. **No CPU math.** Everything on the NPU, fused, state on-chip, no DDR round-trips. (The shipped 0.20 path currently *violates* this — FFN, GQA, and the SSM projections run on host. Moving them to NPU is the aligned goal.)
2. **Do not touch the proven C++ engine math.** Dispatch/BO-packing changes must pass a **bit-identical golden-token gate** (a lossless-argmax relaxation is a separate, sign-off-gated fork).
3. **All kernels always vectorial** — no slow scalar loops.
4. **Hardware budget:** 8 AIE2P tiles, ~64 KB tile data memory, 2 MM2S + 2 S2MM shim per tile, ≤9 simultaneous hw_contexts, `aie::load_v` reads from the nearest 8-float-aligned address (offsets/strides must be %8==0), BD length ≤4096 words, BD repeat ≤255.

**What "10×" would require (the structural ceiling):** Every decode token must read ~14 GB of weights from DDR. At the platform's ~20 GB/s DDR peak, the **absolute M=1 decode floor is ~0.7 s/token = ~1.4 tok/s**, *if every weight read streamed at full DDR bandwidth with zero overhead.* We are at 4.9 s (7× off that floor). **10× (2.1 tok/s) needs ~2× the DDR peak — physically impossible.** The honest target is closing the 7× gap to the 1.4 tok/s ceiling, i.e. a ~3–5× win to ~0.5–0.7 tok/s, not 10×.

---

## 2. What we intended

The original theory: the token is dominated by **144 GDN dispatches/token × ~30 ms syncobj floor ≈ 4.3 s** (48 SSM layers × 3 passes, each `.wait()` blocking). If true, the lever is **dispatch amortization** — collapse many dispatches into few — and the 10× path is **speculative decoding via the model's MTP/NextN block** with a **chunkwise M=K GDN kernel** that processes K tokens in 1 dispatch.

That theory drove three successive NPU-kernel efforts:
- **Chunkwise M=K GDN, on-tile state S** (hold S[128,128] on-tile, loop K recurrence steps) → ~18× plan.
- **One-xclbin 3-pass-sequential GDN** (collapse 3 dispatches → 1, S DDR-streamed) → ~3× plan.
- **Fused M=1 3-pass→1 GDN** (earlier) → the predecessor that motivated the above.

All three died. The 30 ms-floor premise that motivated them was then **falsified by measurement**, redirecting to the **NPU FFN** lever (the 40% host stage). That Stage-1 de-risk is the final and most informative piece.

---

## 3. The real measured profile (the foundation every lever must be measured against)

`FST_TRUNK_TIME=1` per-stage timing in `run_trunk_token` (`src/fst_engine.cpp`) + per-pass timing in `tools/gdn_probe.cpp`:

| stage | time/token | % | per-layer | where |
|---|---|---|---|---|
| **FFN** (64 layers) | **1.95 s** | **40%** | 30 ms | **host** (MXFP4 matvec, OpenMP) |
| **GDN passB** (48 layers) | **1.61 s** | **33%** | 33.6 ms | NPU (bidirectional shim DMA) |
| SSM other (48 layers) | 0.91 s | 19% | 19 ms | host (conv1d + q/k/v/gdec/beta + norm + out_proj) |
| GDN passA+delta (48 layers) | 0.28 s | 6% | 5.8 ms | NPU |
| GQA (16 layers) | 0.26 s | 5% | 16 ms | host |
| lm_head | 0.07 s | 1.5% | — | host |
| **total** | **~4.9 s** | | | **0.20 tok/s** |

**Per-pass (the falsification of the "30 ms floor"):** passA = 5.44 ms, delta = **0.34 ms**, passB = 33.56 ms. There is **no uniform 30 ms floor**; the real scheduling floor is ~0.3 ms (the delta pass). passB's 33.6 ms is **real bidirectional shim DMA** (9.4 MB at ~280 MB/s per direction: 6.3 MB read incl. 3.1 MB of delta replicated 128×/v-head + 3.1 MB S2 write), not scheduling overhead. Cutting the "floor" saves 0.3 ms × 144 = **43 ms/token — nothing.**

**Consequence:** ~64% of the token (FFN 40% + SSM-other 19% + GQA 5%) is **host compute**, in violation of the no-CPU-math north star. The NPU-side GDN (passA+delta+passB = 39%) is the only part already on NPU.

---

## 4. How we did it — and where we failed (every approach, measured)

### 4.1 Fused M=1 3-pass→1 GDN — DEAD (earlier)
M=1 fusion of passA/delta/passB into one kernel: **2-S2MM race vs 1-S2MM stall**; `aie.iron.Buffer` RMW **600× slow**. The 3-pass dispatch boundary is load-bearing. `docs/QWOPUS_FUSED_GDN_POSTMORTEM.md`. Do not enable `fst_gdn_fused.xclbin`.

### 4.2 Chunkwise M=K GDN, on-tile state S — DEAD (2026-07-13)
Hold S[128,128] on-tile (bf16 32 KB or col-split), RMW across K recurrence steps. Tested every tile count and storage class:
- 1-tile 32 KB Buffer: 4878 ms. 8-tile 4 KB Buffer (col-split, bit-correct): 1094 ms = 137 ms/token/layer = **33× worse than shipped.** 2-tile 16 KB stack (user-authorized pivot): 2153 ms AND incorrect (argmax 13.3%).
- **Root cause: on-tile S RMW is slow at ≥16 KB in ALL storage classes** (Buffer 4 KB/32 KB AND 16 KB stack all ~1.3 µs/vec ≈ 340–700× tile-local). The "fast" 4-tile 8 KB result was a toy doing one recurrence step; the fast-stack threshold is **≤8 KB**. Real GDN needs ≥16 KB. **Structural impasse:** ≤8 KB ⇒ ≤32 cols ⇒ ≥4 tiles ⇒ >2+2 shim ⇒ chain ⇒ persistent Buffer ⇒ slow. No tile count (1/2/4/8) escapes. `docs/QWOPUS_CHUNKWISE_GDN_*_POSTMORTEM.md`. Memory: `qwopus-chunkwise-gdn-2tile-stack-deadend`.

### 4.3 One-xclbin 3-pass-sequential GDN — DEAD (2026-07-13)
Collapse the 3 xclbins/3 dispatches into 1 xclbin/1 `xrt::run` per SSM layer; S DDR-streamed (never on-tile); small a/b/delta in self-loop ObjectFifos (IRON supports prod==cons self-loops). Compiles + runs first try. **Result: 512 ms = 5.7× SLOWER than shipped 90 ms, AND incorrect** (max|Δy|=70; not debugged — already 5.7× too slow).
- **Root cause: the dispatch-amortization premise is false.** The shipped 30 ms/dispatch is real pipelined per-pass work, not scheduling overhead. Serializing 3 passes on one tile forces an **interleaved multi-fifo acquire pattern per v-head** (16 in_S → 1 par → 1 out → 16 in_S → 16 out) that shim DMA **cannot pipeline** — each acquire stalls on a BD re-arm. Per-v-head interleaving is forced by the strict sequential recurrence (all-48-then-all-48 would need 48 KB ab on-tile = the 4.2 dead-end). `docs/QWOPUS_3PASS_*` (if present). Memory: `qwopus-3pass-onekernel-deadend`. Artifacts: `kernels/fst_gdn_3pass*`, `tools/gdn_3pass_probe.cpp`. **Do not wire.**

### 4.4 "Attack the 30 ms floor" — premise FALSIFIED (2026-07-13)
Chosen direction after 4.2/4.3 died: cut the per-dispatch syncobj floor at the driver/runtime level. **Measurement falsified the floor's existence** (§3): real floor 0.3 ms; cutting it saves 43 ms/token. The only real GDN-side remnant is passB's redundant delta stream (§5.1).

### 4.5 NPU FFN — Stage-1 de-risk DONE, viable path identified, NOT yet built (2026-07-13)
The 40% host lever, aligned with no-CPU-math. Built three probes (`kernels/fst_qwopus_ffn_probe*`, `tools/qwopus_ffn_*`). Qwopus FFN decode = M=1 MXFP4 matvec (gate/up [17408,5120], down [5120,17408]); weights raw MXFP4 17-byte blocks, ~142 MB/layer, ~9 GB/token; host `mxfp4_matvec_f32` (OpenMP, fp32 accumulate) at ~4.7 GB/s (memory+dequant bound, not lazy). M=1 ⇒ 2 FLOP/byte ⇒ memory-bound on any processor; NPU reads the same DDR, so the only possible win is higher aggregate DDR read bw + on-tile dequant.

**Three measured results:**
1. **8-tile aggregate weight-read DMA = 51.61 GB/s** (`tools/qwopus_ffn_noop8`: 50.1 MB across 8 tiles in 0.97 ms) — **11× host 4.7 GB/s.** NPU weight-DMA is **not** the bottleneck. (1-tile = 11.74 GB/s.)
2. **20 KB read-only held input vector h = 1470× slow** (8 MB/s/tile, 779 ms for 6.27 MB). No-op variant with the same fifo footprint but h **unread** = 0.53 ms = 11.74 GB/s/tile ⇒ the slowdown is **entirely the 20 KB h-array access** (2176 rows × 20 KB = 44 MB of repeated on-tile-array reads at ~1.3 µs/vec). **This generalizes the §4.2 on-tile-array wall to READ-ONLY arrays ≥16 KB** — killing the held-h matvec design for **all** q35 projections (every one has K=5120 ⇒ 20 KB held input: FFN gate/up/down, ssm qkv/gate/alpha/beta/out, attn q/k/v/o).
3. **Scalar reads of a FLOAT-typed (byte-reinterpreted) ObjectFifo return garbage** (max|Δ|=3.4, argmax wrong) — the fused-kernel memory contract generalizes: scalar reads of a DMA-filled element return a ramp unless the fifo is **pure uint8**. Fix: make the weight stream pure uint8.

**Viable B path (fully specified, measured anchors, not yet built):** one uint8 stream per tile, packet = `[h(20 KB) | RPB weight rows]` (re-stream h per packet — reads h from a *streamed* element at 11.74 GB/s, not a *held* one; pure-uint8 fixes the scalar-read bug); 8-tile N-split; fp32 accumulate (bit-identical-achievable if same order, but vectorised 8-lane MAC differs in rounding ⇒ likely needs the lossless-argmax gate relaxation). **Ceiling ~0.27–0.32 tok/s** (FFN 1.95 s → ~0.5–0.7 s; DMA-bound at 51 GB/s agg with ~137 MB/proj h-redundancy ≈ 0.51 s, but **dequant compute unmeasured** — scalar nibble extract × 1.39 M vec-MAC/proj/tile may dominate). Memory: `qwopus-npu-ffn-stage1-measured`.

---

## 5. The surviving levers (ranked, with measured ceilings)

### 5.1 passB delta-broadcast — ~0.24 tok/s (small, safe, bit-identical)
Hold the 128-float delta on-tile per v-head (≤512 B, under the 8 KB fast threshold) via a 2nd MM2S; drop the 3.1 MB replicated delta read from passB. passB 33.6 ms → ~18 ms (bidirectional S0-read + S2-write still bounds at ~280 MB/s/dir). Saves ~0.75 s/token → ~0.24. Bit-identical (same math, delta re-sourced). Touches the shipped passB kernel (gate-protected). Row stride must stay %8==0 (use 136, not 130).

### 5.2 NPU FFN re-stream-h kernel — ~0.27–0.32 tok/s (the aligned lever, §4.5)
Build the re-stream-h uint8 8-tile kernel. Best single lever, beats 5.1 and host FFN, aligned with no-CPU-math. **Key generalization: this kernel is a general M=1 MXFP4 matvec that serves EVERY q35 projection, not just FFN** — gate/up/down, ssm qkv/gate/alpha/beta/out, attn q/k/v/o are all the same shape (K=5120 ⇒ 20 KB h). So one kernel can move **~3.1 s (64%) of host work** to NPU, not just the 1.95 s FFN. If dequant compute is fast, moving all host projections could approach the DDR floor (~0.5–0.7 tok/s).

### 5.3 Constraint relaxations (sign-off-gated forks)
- **bf16 on-tile / lossless-argmax gate** instead of bit-identical: already A/B-proven lossless for GDN S (`FST_Q35_GDN_BF16S`, 39/39 argmax). Would let the NPU FFN use BF16 MMUL (vectorial, fast) instead of fp32-accumulate, and dodge the precision-order fight. Needs explicit user sign-off (it's a gate change, not a math change).
- **Speculative decoding** (MTP/NextN draft + verify): only helps if a GDN M=K path exists — but both GDN M=K branches are dead (§4.2/4.3). Without a chunkwise GDN, SD verify costs K trunk-passes ⇒ ~1.2× lossless, not K×. **SD is blocked on a GDN breakthrough that is currently dead-ended.**

### 5.4 Accept ~0.2–0.3 tok/s as the in-constraint ceiling
Combine 5.1 + 5.2 → ~0.30–0.35. That is likely the practical ceiling without a GDN M=K breakthrough or a constraint relaxation.

---

## 6. Forks for the orchestrator (precise next prompts)

The orchestrator should drive these with targeted prompts. Each is independently measurable.

**Fork A — Build & measure the NPU FFN re-stream-h kernel (highest value).**
1. Author `kernels/fst_qwopus_ffn_kernel.cc` + `gen_qwopus_ffn.py`: 8-tile, one uint8 stream/tile, packet `[h(20 KB) | RPB weight rows]`, scalar dequant (pure-uint8 ⇒ scalar reads work) + vectorised 8-lane fp32 MAC, fp32 accumulate. Reuse `fst_qwopus_ffn_probe_kernel.cc` dequant math.
2. Probe (`tools/qwopus_ffn_probe.cpp` adapted): validate max|Δ| vs host `mxfp4_matvec_f32` ref **and** measure compute-vs-DMA (compare full-matvec time to the no-op 51 GB/s DMA time). **If compute-dominated → vectorise nibble extract (aie::unpack + vector LUT) before proceeding.**
3. Decide the gate: bit-identical (replicate host fp32 order — may violate vectorial) vs lossless-argmax (BF16 MMUL or vectorised fp32 — needs sign-off).
4. Wire behind `FST_Q35_FFN_NPU` in `process_ffn_q35` (`src/fst_engine.cpp:5471`): dispatch + BO-packing only, no math change. Golden-token diff vs shipped host path.
5. **Generalize**: the same kernel serves ssm/attn projections — wire those too (`process_ssm` 4826, `process_gqa_q35` 5281) to move the full 64% host workload.

**Fork B — passB delta-broadcast (small, safe, ship first).** Author the held-delta passB variant (§5.1), probe bit-identical, wire behind a gate. Quick win to ~0.24 while Fork A proceeds.

**Fork C — Re-examine the GDN M=K wall with the new on-tile-array knowledge.** The §4.2 wall is "≥16 KB on-tile arrays are slow (RMW and read-only)." A GDN M=K design that keeps **all** on-tile state ≤8 KB (e.g., stream S in ≤8 KB column-slabs, never hold full S) might escape — but the ≤8 KB ⇒ ≤32 cols ⇒ ≥4 tiles ⇒ chain ⇒ Buffer ⇒ slow impasse still applies. Likely still dead, but the orchestrator may find a slab-streaming restructuring that holds no ≥16 KB array. Low priority.

**Fork D — Constraint relaxation decision.** Present the bit-identical vs lossless-argmax fork to the user. The lossless-argmax gate (proven for GDN) unlocks BF16 MMUL NPU kernels (vectorial, fast) for FFN and all projections, and is the cleanest path to the ~0.5 tok/s range.

---

## 7. Artifacts produced this effort (repo-clean)

- `kernels/fst_qwopus_ffn_probe_kernel.cc` — `ffn_matvec_rows` (held-h matvec, the 1470×-slow design) + `ffn_noop_rows` (DMA-only diagnostic).
- `kernels/gen_qwopus_ffn_probe.py`, `gen_qwopus_ffn_noop.py`, `gen_qwopus_ffn_noop8.py` — IRON generators (1-tile matvec, 1-tile no-op, 8-tile no-op).
- `tools/qwopus_ffn_probe.cpp`, `qwopus_ffn_noop.cpp`, `qwopus_ffn_noop8.cpp` — host probes.
- `kernels/fst_qwopus_ffn_probe*.xclbin`, `fst_qwopus_ffn_noop*.xclbin` + `_insts.bin` — compiled.
- `kernels/fst_gdn_3pass*`, `tools/gdn_3pass_probe.cpp` — the dead one-kernel 3-pass (do not wire).
- `src/fst_engine.cpp` — `FST_TRUNK_TIME` per-stage timing in `run_trunk_token` (gated, off by default; safe to keep).
- Memory: `qwopus-real-profile-no-30ms-floor`, `qwopus-npu-ffn-stage1-measured`, `qwopus-3pass-onekernel-deadend`, `qwopus-chunkwise-gdn-2tile-stack-deadend`, `qwopus-fused-gdn-race-rootcause`.

---

## 8. One-line summary for the orchestrator

The 10× goal is physically impossible (DDR-bandwidth-bounded at ~1.4 tok/s absolute for M=1 decode of this 27B dense model). The shipped 0.20 tok/s is 7× off that floor. Two GDN NPU-speedup branches are dead; the "30 ms floor" was a measurement error. The real levers are (1) **NPU FFN re-stream-h kernel** — Stage-1 proved NPU aggregate DMA is 51 GB/s (11× host) and the only blocker is the 20 KB held-h array (re-streaming h fixes it), ceiling ~0.3 tok/s and **generalizes to all q35 projections (64% of the token)**; (2) **passB delta-broadcast** → ~0.24. The decisive unmeasured risk is **dequant-compute throughput** (scalar nibble extract) — measure it before wiring. The cleanest accelerant is the **lossless-argmax gate relaxation** (already A/B-proven for GDN), which unlocks vectorial BF16 NPU kernels.