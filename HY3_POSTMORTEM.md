# HY3 Post-Mortem — Final Push toward >2 tok/s

**Date:** 2026-07-09
**Goal:** push FaStar decode from ~0.05 tok/s toward >2 tok/s (DDR5-bandwidth-bound) via
(1) MLA consolidation 9→5 contexts, (2) fused FFN via raw MLIR-AIE, (3) zero-wait async,
(4) predictive expert pager.
**Direction taken (user):** *Hybrid — land the safe MLA async win + measurement first, then spend
remaining effort attacking the MLA consolidation insts-extraction blocker.*

## TL;DR

**No tok/s improvement was achieved this session. Decode stays at 0.04 tok/s (50× below the >2 target).**
Two concrete advances and one hard toolchain blocker:

1. **Safe MLA async win — FAILED and reverted.** Per-slot BO ping-pong reproduces the documented
   29506-garbage failure mode. The "safe" win is provably *not* safe: it fixes BO reuse but not
   hw_context DMA contention. Reverted; serial path byte-identical to shipped; zero regression.
2. **MLA kernel-correctness fix compiles** — the row-major gather/scatter C-layout kernel is now
   linked into a fresh `fst_mla_unified.xclbin` (44010B, was 11386B buggy tile-major). **Unverified
   on HW and undeliverable** (see blocker).
3. **MLA consolidation blocked at the IRON toolchain layer** — per-sequence `insts.bin` extraction is
   impossible from a unified single xclbin with the current aiecc (precise root cause below). Even
   unblocked, consolidation is context-cap relief (9→5), **not a tok/s lever** — the 0.04 floor stands.

The accumulated evidence (memory) says the only path past 0.04 is **IRON MLA+FFN fusion** (fewer
dispatches per layer), a greenfield raw-MLIR effort, not async/packing/pager. That was not landable
this session.

## (a) NPU context count

From `run_baseline2.log`:
```
[aiebu] Created PERMANENT hw_context for ./kernels/fst_expert_gemm_vec.xclbin  (total: 1)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_expert_gemm_down.xclbin (total: 2)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_mla_qc.xclbin   (total: 3)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_mla_wqb.xclbin   (total: 4)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_mla_ob.xclbin   (total: 5)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_mla_qksv.xclbin (total: 6)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_mla_qck.xclbin  (total: 7)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_dequant_v4.xclbin(total: 8)
[aiebu] Created PERMANENT hw_context for ./kernels/fst_ew_unified.xclbin(total: 9)
[npu] Total active hw_contexts: 9 (limit 8)
Expert Pager: ... hit_rate=1.000 ...
NPU Contexts: creates=9 evictions=0 hits=0 active=9
```
- **9 contexts, 0 evictions, all permanent.** (The driver's "limit 8" line is a soft warning; the
  engine's 9-cap LRU is not exercised — every context is permanent.)
- **Target was 9→5** by merging the 5 MLA xclbins (qc/wqb/ob/qksv/qck) into one
  `fst_mla_all.xclbin` with 5 runtime_sequences sharing 1 hw_context.
- **NOT achieved.** Consolidation blocked (see §"MLA consolidation outcome"). Context count unchanged.

## (b) Fused-FFN dispatches per layer (was 42)

- **Still ~42** (6 routed experts × 6 dispatches [dequant, gate GEMM, up GEMM, silu EW, mul EW,
  down GEMM] + ~6 shared elementwise). `run_baseline2.log` shows 6 `L0 ffn cur(in)`/`ffn_out` blocks
  per layer pass, confirming 6 routed experts (topk=6).
- **Fusion NOT implemented this session.** The raw-MLIR-AIE fused-FFN kernel was not compiled or
  wired; per the hybrid direction, the "fusion attempt" effort was redirected to the MLA
  consolidation insts-extraction blocker specifically.
- Total layer dispatches (MLA + FFN + shared) ≈ 77 per prior measurement
  (memory: `bo-pool-racefree-lmhead-not-lever` — body 97% of time, 360 ms/L × 77 disp × 4.7 ms/disp).

## (c) Async pipeline + NPU busyness

**MLA K-chunk async ("safe win") — FAILED:**
- Implemented a 2-slot BO ping-pong ring (`bo_mla_atile_/atile2_`, `ctile_/ctile2_`,
  `bhost_/bhost2_`), gated on `FST_MLA_ASYNC`. The by-construction determinism argument addressed the
  BO-reuse race (per-slot BOs) but **missed the hw_context DMA-contention race**: `npu_sync_to`/
  `sync_from` on slot 0's BOs while slot 1's run is in-flight on the same hw_context.
- Result: **First token 29506 (garbage) instead of 671 ("The")** — exactly the failure mode the
  code NOTE at `fst_engine.cpp:501-512` documents. This is the same 671-vs-29506 nondeterminism
  seen in the earlier reverted async.
- **Reverted** via `git checkout src/fst_engine.cpp` (the only uncommitted change; 96 insertions).
  Rebuilt. `grep -c FST_MLA_ASYNC` = 0. Serial path byte-identical to shipped; **zero regression**.
- Conclusion: a per-slot BO ring is **not** sufficient. Zero synchronous `run.wait()` inside a layer
  is unachievable while MLA runs on a shared hw_context — the hardware serializes DMA on a context,
  so overlapping runs on the same context corrupt each other. (Cross-context overlap is the only
  safe concurrency, and that needs consolidation to free contexts — circular with the blocker.)

**FFN async ring — already shipped (unchanged):** 4-slot `scratch_pool_` ring (`pool_idx_ & 3`),
down-GEMM overlapped with next-expert dequant. Not touched this session.

**NPU busyness:** not re-measured this session (would need `xrt.ini` aie_trace/profile). Prior
measurement stands: body = 97% of wall time, ~4.7 ms/dispatch, dispatch-COUNT bound.

## (d) Expert pager — SSD stall time

```
Expert Pager: gets=3320 hits=3320 misses=2674 hit_rate=1.000 prefetched=5878 draft_prefetch=344 cache=5992MB
```
- **Zero synchronous SSD stall.** `hit_rate=1.000` — every `get()` found its expert already in RAM.
  The 2674 "misses" are cold first-touches, all covered by the 5878 prefetched (predict + draft
  prefetch overspec). No `get()` ever blocked on SSD.
- The pager was **already optimal at baseline** and was not the bottleneck. No work needed or done.

## (e) Decode tok/s vs >2 target

```
Prefill Time: 110887.0 ms (14 tokens)
Decode Tokens/sec: 0.04
Peak RAM (RSS): 53.6 GB
DSpark Accept Rate: 0.250 (2/8 draft tokens)
```
- **0.04 tok/s, target >2 tok/s → 50× gap.** No improvement vs the 0.04–0.05 floor established
  across prior sessions.
- The floor is **dispatch-COUNT bound**, not I/O (pager hit 1.0) and not lm_head (2.4% of time).
  Per accumulated memory, >15 tok/s needs IRON MLA+FFN **fusion** (collapse 77 dispatches/L → a
  handful), not async, not packing, not pager tuning. That fusion is greenfield raw-MLIR work and was
  not landable this session.

## (f) Output text + coherence

- **Coherent.** `run_baseline2.log`: `[DSpark] First token: 671` (= "The", the deterministic argmax
  anchor verified across many prior sessions). Established readable continuation from prior verified
  runs: **"The importance of NPU (Neural Processing Unit …"** — matches the prompt
  `"The importance of NPU technology in modern laptops is"`.
- DSpark accept rate 0.250 (2/8); 7 verify passes. Output is stable and coherent, **not** degraded by
  any change this session (the only engine change — MLA async — was reverted before this run).

## MLA consolidation outcome (the "fusion attempt")

**Kernel-correctness fix — DONE (source level, compiles, unverified):**
- The shipped unified kernel wrote C **tile-major**; IRON's drain TAP reads C **row-major** [m,n] →
  every output element scattered to the wrong host position (~3.7× magnitude / sign-flip). The fixed
  kernel `kernels/fst_mla_unified_kernels.cc` keeps the proven 2×2-vectorized `aie::mmul<4,8,8>`
  expansion but **gathers the running partial C from row-major and scatters back to row-major**
  (`gather_tile_rm`/`scatter_tile_rm`).
- Ported `mla_qk_scale`/`mla_sv_scale` (scale is layout-independent) into the fixed kernel so the
  IRON builders' `sc_k` ExternalFunction bindings resolve.
- Swapped `gen_mla_unified.py` SRC link from the buggy `fst_mla_unified_kernel.cc` to the fixed
  `fst_mla_unified_kernels.cc`; recompiled → **new `fst_mla_unified.xclbin` 44010B** (was 11386B).
  aiecc succeeded.
- **Unverified on HW and undeliverable** because per-sequence insts cannot be extracted (below).

**Insts extraction — BLOCKED (precise, new toolchain finding):**
The engine's `run_blob` control-kernel dispatch path needs one `insts.bin` per MLA op. Extracting
those from the unified single xclbin is impossible with the current aiecc:

1. `gen_mla_unified.py:extract_per_kernel_insts` runs `aie-translate` on
   `fst_mla_unified.mlir.prj/main_npu_lowered.mlir`. **That file is never emitted by the current
   aiecc** — the prj dir contains only `input_with_addresses.mlir` (plus per-core ELF/CDO bins).
2. aiecc's NPU lowering **strips the `@qc/@kvc/@oa/@ob/@wq_b/@k_pe/@qk/@sv` sequence names** present
   in the source MLIR → `_anonymous0.._anonymous7` in `input_with_addresses.mlir`.
3. `aie-translate --aie-npu-to-binary --aie-sequence-name=qc` on `input_with_addresses.mlir`:
   - With the anonymous names: silently falls back to a **trivial 16B stub** (name matches nothing).
   - After sigil-aware re-naming (`@_anonymous{i}→@name[i]`, `%_anonymous{i}→%buf_name[i]`): `rc=0`,
     sequence *found*, but **still 16B** — `input_with_addresses.mlir` is not the fully NPU-lowered
     stage; the real per-sequence NPU instruction sequences are baked into the xclbin by aiecc's
     internal `--aie-generate-npu-insts` pipeline and are **not exposed** for per-sequence extraction.
4. The 8 stale `fst_mla_{qc,kvc,oa,ob,wq_b,k_pe,qk,sv}_insts.bin` (420B each, **all distinct md5** —
   real per-kernel insts) were produced by IRON's **`design.compile()`** path (used by
   `compile_mla_unified_final.py` / `_native.py`), which compiles each kernel **separately** —
   producing a **separate xclbin per kernel**, defeating the single-xclbin consolidation. They do
   not match the new 44010B unified xclbin.
5. The working FFN-unified insts path (`compile_ffn_unified.py`: aiecc's single `insts.bin` →
   `fst_ffn_unified_insts.bin`) works because FFN has **one** runtime_sequence; MLA unified has
   **eight**, so a single default `insts.bin` cannot serve all eight ops.

**Engine rewire:** not attempted (downstream of the blocked insts).
**HW verification of the kernel fix:** not done (can't dispatch without insts).
**Net:** consolidation delivers context-cap relief (9→5) only — **not a tok/s lever** — and is
blocked at the toolchain layer regardless.

## Bottom line

- Safe MLA async win: **not achievable** (hw_context DMA contention → garbage; reverted).
- MLA consolidation: **kernel fixed & compiles, but insts extraction blocked at IRON toolchain**;
  not a tok/s lever anyway.
- FFN fusion: **not attempted** (effort redirected to consolidation per hybrid choice).
- Expert pager: **already optimal** (zero stall, hit 1.0).
- Contexts: **9, zero evictions** (consolidation would relieve, blocked).
- **Decode: 0.04 tok/s, unchanged. Output coherent ("The importance of NPU (Neural Processing Unit…").**

The honest path to >2 tok/s remains **IRON MLA+FFN fusion** (collapse per-layer dispatch count),
a greenfield raw-MLIR-AIE effort. This session delivered the row-major MLA kernel fix (compiles,
unverified) and a precise characterization of the consolidation toolchain blocker — but no tok/s move.