# HY3 Fused-FFN Dispatch Collapse — Post-Mortem (2026-07-10)

## Goal (user directive)

Starting point: **0.05 tok/s** on HY3 decode. The user's premise: the floor is
**dispatch-count** — ~3840 NPU micro-dispatches/token (8 routed experts × 6 ops ×
80 layers) × ~5 ms flush ≈ 19-24 s/token, while "NPU compute itself takes <1s".
Directive: stop treating the NPU like a GPU; collapse each layer to 1-2 NPU
dispatches via on-device expert loops ("ZERO CPU intervention inside a layer").
Chosen scope: **FFN fusion + parallelize host GEMMs**. Target: **~0.8-1.0 tok/s**.

## What shipped (FST_HY3_FUSED_FFN guard)

The 8-routed-expert FFN per layer collapsed from **48 per-op dispatches → 4 NPU
dispatches** (`dequant_8exp`, `gate` MC, `up` MC, `down` MC) + host silu/mul:

| Stage | Before (per-op) | After (fused) | ms/layer (M=1) |
|-------|-----------------|---------------|----------------|
| dequant | 8× `hy3_dequant` (serialized on 1 ctx) | **1× `hy3_dequant_8exp`** (16 cores, 4.7M blocks) | 19 |
| gate+up | 16× GEMM | **2× MC GEMM** (8 cores, 1 expert/core) | 10 |
| silu/mul | 16× ew dispatches | **host CPU** (see below) | 3 |
| down | 8× GEMM | **1× MC GEMM** (8 cores) | 5 |
| pack (8 experts → 80 MB BO) | n/a | OpenMP-parallel, from pager Expert structs | 3 |
| **FFN total** | ~211 ms | **~41 ms** | **5× faster** |

Key wiring (`src/fst_engine.cpp`, `process_expert_ffn_hy3_fused`):
- **Pack bypasses `get_expert_bo`**: copies directly from the pager's RAM-cached
  `Expert` structs into `bo_fused_weight_`, parallelized with OpenMP (8 cores
  aggregate ~33 GB/s into uncacheable host_only memory vs ~1 GB/s single-thread).
  `get_expert_bo`'s BO cache added a serial 10 MB host_only write per BO-miss
  that the fused path didn't need (dequant_8exp reads the packed BO, not
  per-expert BOs). A `sync(TO_DEVICE)` fences the parallel writes for the NPU
  DMA (a no-sync parallel pack raced: dequant cos 1.0→0.99995, one stale nibble).
- **silu/mul on host CPU**: the `ew_unified` `silu_b`/`mul_b` kernels are
  **single-core** (`gen_ew_unified.py`: one `Worker`) — 196608 elems on one AIE
  core = **94 ms** (2×47 ms), the NPU regressing ~100× vs CPU. Same accepted
  small-op host deviation as GQA/shared/dense. Float silu is also *more*
  accurate than the bf16 ew kernel, so the down-GEMM cos held (0.998).
- DS4 path byte-identical (arch-gated `config_.arch == ARCH_HY3`).

**Correctness (FST_HY3_FUSED_AUDIT, L1):** dequant cos=**1.00000** (bit-exact,
maxdiff=0.0000), down-GEMM cos=**0.99807-0.99812** (DS4 gold 0.998). 5 NPU
hw_contexts (< 9-cap). Builds clean (OpenMP 4.5 linked).

## The answer to (a/b/c/d)

**(a) Dispatch count per layer:** 48 (per-op) → **4** (fused: dequant + gate +
up + down; silu/mul on host). Target was 1-2. The remaining 4 are genuine NPU
GEMM/dequant compute, not micro-dispatches. Reaching 1-2 needs fusing the GEMMs
themselves (gate+up, or dequant+gateup) — deferred (see Why not 1-2).

**(b) On-device expert loop?** Partial. `dequant_8exp` processes all 8 routed
experts in **one dispatch** (16 cores, per-core TAP offsets over 4.7M blocks).
The MC GEMMs process 8 experts in one dispatch (8 cores, 1 expert/core). But
this is **not one kernel doing all 8 experts end-to-end** — it is 3 NPU
dispatches that each internally parallel the 8 experts, plus host silu/mul. A
single fused gate+up+silu+mul+down kernel (the original Task #22 plan) was
**deferred** — it gives no tok/s gain (SSD/pack-dominated, see below) at real
IRON risk (unproven large ObjectFifo elements / in-kernel full-K accumulation).

**(c) Decode tok/s — did we break 1?** **No.** 0.05 → **0.07 tok/s** (measured,
`run_fused_sync.log` / `run_fused_cap8.log`, final shipped path: sync fence +
host CPU silu). The dispatch collapse made the FFN **5× faster (211→41 ms/layer)**
but tok/s barely moved because **the real floor is SSD expert loading, not
dispatch count** (see next section). Per-step timing confirms it: when the
pager's SSD fetch is a page-cache hit (`fetch=0.14 ms`) the layer is **42 ms**;
when it misses (`fetch=83-90 ms`) the layer is **125-130 ms** — the layer time
tracks `fetch`, not the NPU dispatches. The 0.8-1.0 target is **not achievable
on this hardware** for this model.

Final shipped-path stats (`run_fused_sync.log`, raw M=1 "Hi"):
- Prefill: **20,380 ms** (M=1, 80 trunk layers)
- Decode: **0.07 tok/s**
- Peak RSS: **33.5 GB** (< 35 GB cap)
- Expert pager: gets=1996 hits=1996 **hit_rate=1.000**, cache=5995 MB
- NPU contexts: creates=5, evictions=0, **active=5** (< 9 cap)

**(d) EXACT output text: `Hello! How can I help you today`**

Coherent output requires the chat template (M=16 prefill). Verified on the
shipped fused path (`run_fused_coherence.log`, `FST_HY3_FUSED_FFN=1`, templated
"Hello", `--tokens 8 --temp 0`, exit 0):

```
prompt: "Hello"  →  chat template  →  16 prefill tokens
[hy3-gen] prefill-last=16883 (M=16)         → "Hello"
[hy3-gen] decode=0    (pos=16)              → "!"
[hy3-gen] decode=1992 (pos=17)              → " How"
[hy3-gen] decode=539  (pos=18)              → " can"
[hy3-gen] decode=360  (pos=19)              → " I"
[hy3-gen] decode=1854 (pos=20)              → " help"
[hy3-gen] decode=444  (pos=21)              → " you"
[hy3-gen] decode=5247 (pos=22)              → " today"
```

This **matches the per-op phase-5 baseline** ("Hello! How can I help you today?"
— the "?" is the 8th decode token, just beyond `--tokens 8`). The fused path
(dequant cos **1.00000** bit-exact; down-GEMM cos **0.998**; host float silu
*more* accurate than the bf16 ew kernel) produces the **same greedy decode** as
the verified per-op path — the cos-0.998 difference did not flip any argmax.

> **Caveat on the 0.07 tok/s measurement (c):** that number was measured on a
> **raw M=1** prompt (`FST_HY3_RAW_PROMPT=1`, "Hi" → 1 token `[28036]`), whose
> decode emits token **58933** repeatedly → text **"KK"**. This is the **known
> flat-logit argmax instability of raw M=1** (the `hy3-npu-ffn-wired` memory:
> "raw M=1 3048-vs-58933=flat-logit argmax instability NOT bug"), **not a
> fused-path defect** — the per-op path produces the same "KK" on raw M=1.
> Decode is M=1 regardless of prefill, so it is a valid tok/s-floor measurement;
> the templated run above confirms 0.07 tok/s on the coherent path too.

Templated-run final stats: prefill **185,443 ms** (M=16, 80 trunk layers),
decode **0.07 tok/s**, RSS **25.5 GB**, pager hit_rate **1.000**, **5 NPU
contexts** (< 9). (The `Header version 0.1 / Device Generation: 4` noise in the
log is the xdp AIE plugin dumping xclbin headers per kernel run — the
`xrt-ini-spam` known cosmetic, not output.)

## The real floor: SSD expert loading (the premise was wrong)

> **⚠️ CORRECTION (2026-07-10, levers 1+3 measurement):** the "SSD is the floor"
> conclusion below is **WRONG**. It came from misreading the fused-step `total=42ms`
> (FFN-only) as the full layer dt. The cache-hot row (`fetch=0.14ms`) in
> `run_fused_sync.log` shows `ffn_step=42.7ms` but the **full L1 layer dt = 150ms**.
> The real per-layer decode split (480 samples, median 187ms):
> - **host GQA attention + router + shared + norms: ~105ms (≈55%) — THE floor**
> - NPU FFN compute: ~37ms (≈19%)
> - SSD fetch: ~50ms avg (≈26%), range 0.14 (hot)–118 (cold)
>
> The ~105ms host attention is the **4 scalar projection matvecs** (`q/k/v/o_proj`,
> `host_gemm_bnk_f32`, fst_engine.cpp:98 — pure scalar triple loop, no SIMD, no
> OpenMP; task #24's OpenMP covered the FFN pack but **missed the attention
> projections**). ~107M MACs/layer scalar ≈ 105ms. So:
> - Even **perfect SSD hiding** caps at 150ms/layer = **0.083 tok/s** (+19%).
> - Host attention alone caps at ~0.119 tok/s.
> - The "NPU compute floor ~0.45 tok/s" below is wrong by ~4×.
>
> Levers 1+3 (both SSD-side) were implemented + measured: **neither moves tok/s**
> (baseline AHEAD=0/6GB = 0.07 is best; prefetch is net-negative 0.02–0.06; cache
> alone neutral 0.07, redundant with the OS page cache). **Draft-driven prefetch
> is also SSD-side → also caps at 0.083 → NOT the lever.** The real lever is
> **OpenMP/SIMD-parallelize the 4 host attention projections** (row-independent,
> ~8× from 8 cores → 105→13ms → ~0.15–0.2 tok/s). See the levers-1+3 section at
> the end of this file for the full matrix.

The user's premise — "dispatch overhead 19s, NPU compute <1s" — does **not**
hold for HY3. Per-step timing of the fused layer revealed where the 155 ms/layer
actually goes:

```
[hy3-fused-pack] fetch=83ms memcpy=3ms sync=0.7ms   <- SSD expert load
[hy3-fused-step] pack=87 deq=19 gateup=10 silumul=3 down=5  total=~125ms
                                                  + attention/router/host ~30ms
                                                  = ~155 ms/layer = 0.07 tok/s
```

- **`fetch=83 ms`** is `pager_->get` waiting for **8 SSD reads** (8 experts ×
  10 MB at ~1 GB/s = 80 ms). The pager's RAM cache `hit_rate=1.000` — these are
  counted as "hits" because they were *prefetched* (queued one layer ahead by
  `predict_and_prefetch`), but the single prefetch worker reads serially, so the
  reads are not done when `get()` needs them → `get()` blocks on the SSD.
- The experts are on SSD (expert virtual memory): 8 experts/layer × 80 layers =
  640 expert loads/token × 10 MB = **6.4 GB/token**. At the measured **~1 GB/s
  SSD bandwidth** that is **6.4 s/token = 0.15 tok/s absolute ceiling** (zero
  cache reuse). With cache reuse (experts recurring across coherent tokens) the
  reads fall and tok/s rises toward the NPU-compute floor.

**Honest ceiling:**
- SSD bandwidth floor: **~0.15 tok/s** (6.4 GB/token ÷ 1 GB/s, no reuse).
- NPU compute floor: **~0.45 tok/s** (deq+gateup+down ~33 ms/layer × 80, full
  cache reuse, SSD fully hidden by prefetch).
- Measured: **0.07 tok/s** (partial reuse, prefetch lead < SSD load time).

**Breaking 1 tok/s is physically impossible here** without (a) a faster SSD,
(b) smaller experts (compression), or (c) much higher expert-reuse / cache hit
rate. Collapsing NPU dispatches cannot move a bandwidth-bound workload.

### Worker-pool experiment (tried, reverted)
Parallelized the pager's single SSD-read worker into a 6-thread pool to aggregate
SSD throughput at queue depth. **No gain** (fetch 83→57-99 ms, tok/s flat 0.07):
the SSD is **~1 GB/s bandwidth-bound, not queue-depth-bound**, so parallel reads
do not aggregate, and the extra workers only contended on `cache_mtx_`. Reverted
to the single worker (kept the harmless dedup check + `notify_all`).

## Why not 1-2 dispatches (Task #22/26 deferred)

A single NPU kernel fusing gate+up+silu+mul (Task #22) or dequant+gateup+down
(Task #26) would cut dispatches 4→3→2. **Deferred because it buys no tok/s:**
with pack+SSD dominating (~90 ms/layer), shaving the 10 ms gateup dispatch is
invisible, and the IRON mechanics are unproven here (large ObjectFifo elements
for full-K in-kernel accumulation; the proven `fst_fused_ffn_direct.cc` only
K-accumulates a single 64-K tile, emitting partials the host sums). Revisit only
if the SSD floor is solved.

## No-regression (A/B)

`FST_HY3_FUSED_FFN` **unset** → the per-op 6-dispatch/expert NPU path runs
unchanged. The fused code is purely additive behind the guard (arch-gated
`config_.arch == ARCH_HY3` + env guard), so the per-op path is untouched at the
source level. Confirmed at runtime (`run_ab_guard_unset.log`): the guard-unset
binary loaded the model, created the 4 per-op NPU contexts, templated "Hi" → 16
tokens, and ran prefill **L0→L61 cleanly** (~2.5 s/layer, no crash, no FATAL,
exit 124 only because M=16 prefill ~200 s exceeded the 180 s timeout before
decode started). The per-op path's decode tok/s (**0.05**) and coherence
("Hello"→"Hello! How can I help you today?") were verified in prior phases
(`hy3-npu-ffn-wired`, `hy3-engine-phase5` memories) and are unchanged. DS4 path
byte-identical (arch-gated).

## Future levers (ranked, honest)
1. **Hide SSD behind compute** — the prefetch is only 1 layer ahead (~71 ms
   compute) vs ~80 ms SSD load. Needs either a faster SSD, or accurate
   multi-layer-ahead prediction (draft-driven `predict_and_prefetch_from_draft`)
   so reads complete before `get()`. This is the expert-virtual-memory core.
2. **Expert compression** — load <10 MB/expert (the .fst blocks are 17 B/32
   elems; dequant happens on-NPU; could store more aggressively).
3. **Higher cache reuse** — larger RAM cache + reuse-aware routing so the same
   experts recur across tokens (cache hits, no SSD).
4. **Task #22 NPU gateup-silu fusion** — only worth it for dispatch-count
   aesthetics once 1-3 move the SSD floor.

## Appendix: run commands
```
XILINX_XRT=/usr FST_KERNEL_DIR=$(pwd)/kernels FST_HY3_FUSED_FFN=1 \
  FST_HY3_FUSED_AUDIT=1 FST_HY3_LAYER_TIME=1 FST_EXPERT_BO_CAP_GB=8 \
  ./build/ds4_npu_engine --model hy3.fst --prompt "Hi" --tokens 5 --temp 0
```

## Levers 1+3 measurement — Hide SSD + Higher Cache Reuse (2026-07-10)

User directive: **"usa le leve 1 (hide ssd behind compute) e 3 (higher cache
reuse)"**. Both implemented (env-gated, defaults = old baseline, DS4 byte-identical)
and A/B-measured. **Result: neither lever moves tok/s. Baseline is the best.**

### What shipped (all env-gated, additive)
- **Lever 1 — predictive multi-ahead prefetch wired into HY3** (`fst_engine.cpp:4411`,
  MoE branch after `hy3_router_host`, fires for fused + per-op, gated
  `prefetch_ahead_>=1`). `ExpertPager::predict_and_prefetch` queues `cur` +
  next `FST_PREFETCH_AHEAD` layers' experts (Markov: same expert ids). `FST_PREFETCH_AHEAD=0`
  = old baseline (gate off).
- **Lever 3 — `FST_RAM_CACHE_GB` env** (`fst_main.cpp:787`, default 6000 = unchanged).
- **Instrumentation:** `get()` wait-timing (`stat_wait_us_`/`wait_count_`/`max_wait_us_`)
  + HY3 stats print now shows `misses` + `wait`. (`hit_rate` is misleading — a miss
  that later succeeds still counts as a hit; `misses`/`wait` are the real stall signal.)

### A/B matrix (raw M=1 "Hi" --tokens 6 --temp 0, FST_HY3_FUSED_FFN=1)

| config | tok/s | misses | wait | max_wait | prefetched | RSS |
|--------|-------|--------|------|----------|------------|-----|
| **AHEAD=0, 6GB (baseline)** | **0.07** | 3761 | 25.9s | 60ms | 0 | 25.5GB |
| AHEAD=0, 20GB (lever 3) | 0.07 | 1896 | 28.2s | 24ms | 0 | 37.3GB |
| AHEAD=1, 6GB | 0.04 | 865 | 84.6s | 233ms | 7431 | 25.5GB |
| AHEAD=1, 20GB | 0.06 | 601 | 49.6s | 273ms | 3803 | 39.6GB |
| AHEAD=3, 6GB | 0.04 | 898 | 100.9s | 2047ms | 13852 | 25.5GB |
| AHEAD=3, 20GB (lever 1+3) | **0.02** | 819 | 187.4s | 493ms | 14082 | 39.6GB |

`gets=3792`, `hit_rate=1.000` in every row (the misleading part).

### Why both levers fail

**Lever 1 (prefetch) is net-negative.** `misses` drop (3761→601) but `wait`
explodes (25.9→49–187s) and `max_wait` hits 2s. The single SSD worker (1 GB/s,
serial) gets buried by the ahead-queue; the **bigger cache makes it worse** (0.02)
because nothing evicts → 14082 future-layer reads queued, reactive `get()`s wait
behind them. Prefetch also floods **prefill 2.7×** (540 ms/L vs 200 ms/L). Markov
prediction itself is *not* wrong (misses did fall) — the SSD worker just can't
get ahead usefully, and SSD is only ~26% of the layer anyway.

**Lever 3 (cache) is neutral.** 20 GB halves misses (3761→1896) via cross-token
reuse, but tok/s + total wait are unchanged (28.2s). The **OS page cache (~30 GB
buff/cache) already provides fast re-reads**; the pager cache enlargement is
redundant — it shifts gets from "pread (OS-fast ~7 ms)" to "no pread", while the
remaining misses are true cold SSD reads (per-miss wait doubles 6.9→14.9 ms, net
zero). Experts don't recur across tokens (KV growth changes router picks even for
a repeated output token) so cold reads are irreducible by caching.

### The real floor (measured, supersedes the "SSD floor" section above)

Per-layer decode = ~200 ms (median 187, 480 samples) × 80 = 16 s/tok = 0.07. The
cache-hot row (`fetch=0.14ms`) in `run_fused_sync.log` proves the split:

| component | ms/layer | share | evidence |
|-----------|----------|-------|----------|
| **host GQA attn + router + shared + norms** | **~105** | **55%** | `layer_dt 150 − ffn_step 42 − 3` (cache-hot row); consistent 76–114 across 6 tokens |
| NPU FFN compute (deq+gateup+silumul+down) | ~37 | 19% | fused-step `total` minus pack |
| SSD expert fetch | ~50 avg | 26% | pager `wait` 25.9s/run ÷ 480 layers; range 0.14–118ms |

The ~105 ms host attention is the **4 scalar projection matvecs** (`q/k/v/o_proj`,
`host_gemm_bnk_f32`, fst_engine.cpp:98 — pure scalar triple loop, no SIMD, no
OpenMP). ~107M MACs/layer scalar ≈ 105 ms. Task #24's OpenMP covered the FFN pack
but **missed the attention projections**.

**Ceilings:** perfect SSD hiding → 150 ms/L = **0.083 tok/s** (+19%). Host
attention alone → ~0.119 tok/s. The post-mortem's "NPU compute floor ~0.45" was
wrong by ~4× (it assumed attention = 30 ms; real 105 ms).

### No-regression (A/B)
- **Fused + levers ON:** raw M=1 "Hi" emits 58933 ("KK") × 6 = same as baseline
  (correctness preserved even at 0.02 tok/s). Templated coherence ("Hello! How
  can I help you today") already verified on the fused path (`run_fused_coherence.log`);
  prefetch/cache are timing/capacity-only (same experts, same math) → output unchanged.
- **Per-op + levers ON** (`run_ab_perop_levers.log`, FUSED_FFN unset, AHEAD=1, 20GB):
  runs clean, emits 58933 ("KK"), 4 NPU contexts (<9), RSS 47 GB, exit 0. The 0.04
  tok/s (vs per-op 0.05) is the prefetch-flood timing penalty, not a correctness
  regression. DS4 path byte-identical (arch-gated).

### The real lever (ranked, honest)

1. **✅ DONE + MEASURED — OpenMP-parallelize the 4 host attention projections.**
   `#pragma omp parallel for schedule(static)` on the n-loop of
   `host_gemm_bnk_f32`/`xnk_f32`/`f32f32` (fst_engine.cpp:108/~123/~140). Covers
   attention q/k/v/o + dense L0 + shared expert + NextN + host-FFN fallback.
   Row-independent + serial inner k-loop ⇒ **bit-identical** (each `orow[n]`
   computed serially by one thread; only row scheduling changes) ⇒ DS4 byte-
   identical, no argmax flip. **Results (AHEAD=0/6GB = baseline config, only change
   = OpenMP):**
   - Raw M=1 "Hi": 0.07 → **0.10 tok/s (+43%)**, decode layer 200→130 ms
     (host portion 105→43 ms), pager stats **unchanged** (SSD untouched).
   - Coherent "Hello": 0.07 → **0.11 tok/s (+57%)**, decode 116 ms/L, output
     **"Hello! How can I help you today" = exact baseline** (tokens
     0/1992/539/360/1854/444/5247 match), exit 0.
   - Only ~2.4× not 8×: the projections are partly memory/overhead-bound
     (~200 MB weights/layer at aggregate BW + ~480 OpenMP spawns/token).
     Remaining decode floor = SSD ~50 ms (38%) + NPU FFN ~37 ms (28%) + host
     ~43 ms (33%) ⇒ ceiling ~0.115 tok/s; measured 0.10–0.11 ≈ 90% of ceiling.
   - **No-regression (A/B):** per-op path (`FST_HY3_FUSED_FFN` unset, `run_omp_perop.log`)
     0.05→0.06 tok/s, emits "KK" (58933), 4 NPU ctx, exit 0 — OpenMP parallelizes
     the per-op dense/shared/attention too, no breakage. DS4 byte-identical (no DS4
     caller of these helpers; bit-identical regardless).
2. **✅ DONE + MEASURED — SIMD (AVX2+FMA) the inner k-loop.** Added
   `bf16x8_to_f32x8` (`_mm256_cvtepu16_epi32` u16→u32 + `_mm256_slli_epi32(_,16)`
   bf16→fp32, bit-exact, no F16C) + `_mm256_fmadd_ps` 8-wide + `hsum256` tree-
   reduce to `host_gemm_bnk_f32`/`xnk_f32`/`f32f32` (fst_engine.cpp:123/159/~185,
   guarded `#if defined(__AVX2__))&&defined(__FMA__)`, helpers at :39/:44). Build
   `-O3 -march=native -fopenmp`, **NO `-ffast-math`** (it blocks auto-vectorization
   of FP reductions — that was the scalar bottleneck). **NOT bit-identical** (FMA +
   8-way tree reduction change rounding/order vs the serial k-loop) → verified
   **empirically, both gates green**:
   - Raw M=1 "Hi": 0.07 → **0.14 tok/s (+100% over baseline, +40% over OpenMP)**,
     decode emits 58933 ×5 = "KK" preserved, decode layer 125→~89 ms (host 43→~7 ms).
   - Coherent "Hello": 0.07 → **0.12 tok/s (+71%)**, output **"Hello! How can I
     help you today" = EXACT baseline** (tokens 0/1992/539/360/1854/444/5247 match)
     — **no argmax flip** despite non-bit-identical math. exit 0.
   - Per-op no-regression (`FST_HY3_FUSED_FFN` unset): prefill-last=58933 + decode
     58933 ×5 = "KK" preserved, 4 NPU ctx (<9), no FATAL/seg, **0.07 tok/s** (per-op
     0.05 → 0.06 OpenMP → 0.07 SIMD). The host helpers serve both fused + per-op
     paths (attention q/k/v/o, dense, shared, NextN) so SIMD speeds both; only the
     expert FFN differs.
   - Decode floor now **~89 ms/L** = SSD ~50 ms + NPU FFN ~37 ms + host ~7 ms →
     ceiling ~0.16 tok/s; measured 0.12–0.14 ≈ 75–87%. **Host attention is no
     longer the floor** — SSD is back to the largest share (~56% of the smaller
     layer). Further host-side gains are negligible; the remaining gap to the
     ceiling is SSD/NPU. Pager stats **unchanged** all runs (SIMD is host-compute-
     only, SSD untouched).
3. SSD-side levers (prefetch / cache / draft-driven prefetch) — capped at 0.083
   of the *old* 150 ms layer (now 89 ms → ~0.14 ceiling), proven net-negative or
   neutral above. **Do not chase for tok/s.**
4. Task #22 NPU gateup-silu fusion — dispatch-count aesthetics only (marginal
   tok/s; the 37 ms NPU FFN is already a minority share).

## Ping-pong double-buffer measurement — overlap SSD with NPU compute (2026-07-10)

User directive: **"implement Modern Double Buffering (Ping-Pong) to overlap SSD
I/O with NPU Compute. The SSD must read Layer L+1 while the NPU computes Layer
L."** Rules: ZERO synchronous `pread` in the main thread; the NPU must never
stall on SSD; use a background `std::thread` + `pread` (the existing worker) or
`aio_read`. Implemented + env-gated (`FST_HY3_PINGPONG`, default OFF = current
path) + A/B-measured. **Result: NEUTRAL — does not break the 0.11 ceiling.**

### What shipped (env-gated, additive, DS4 byte-identical)
- **`ExpertPager::wait_layer_loaded(L, eids, n)` — the BARRIER**
  (`expert_pager.cpp`). Blocks until every `(L, eids[i])` is resident in the LRU.
  Queues any that are neither cached nor in-flight, then sleeps on
  `loading_cv_` (no `pread` in the main thread — the worker reads). Re-queues
  missing experts on each pass with a 100 ms timed wait, so an expert evicted
  after load can't deadlock the barrier. After it returns, the FFN's `get()`s
  are guaranteed instant cache hits → **zero SSD stall on the compute path**.
- **`ExpertPager::prefetch_layer_async(L+1, eids, n)`** — pure enqueue (wraps
  `prefetch()`, which dedups). Issued right after the barrier so the worker
  reads L+1 during L's NPU FFN (Markov: `eids_{L+1} ≈ eids_L`).
- **Engine wiring** (`fst_engine.cpp`, `process_ffn_hy3` MoE branch, decode
  `M==1` only): `wait_layer_loaded(L)` → `prefetch_layer_async(L+1)` →
  `process_expert_ffn_hy3_fused` (whose `get()`s now fast-hit). Prefill (`M>1`)
  stays reactive (NPU-bound, SSD hidden by the larger M=16 GEMMs). Guard
  `lid+1 < n_layers-1` so the NextN MTP head (blk.80) is never prefetched (it
  isn't in the trunk FFN loop). `pingpong_` member set in the ctor from
  `FST_HY3_PINGPONG` (presence = on, matching `FST_HY3_FUSED_FFN`).
- **Instrumentation:** `stat_barrier_us_`/`barrier_count_`/`max_barrier_us_` +
  a `Ping-pong Barrier: stalls=…` line in the HY3 stats print — the real "did
  the SSD stall the compute path?" signal (analogous to `get()` `wait`).

### A/B (clean, same robust binary, back-to-back; raw M=1 "Hi" --tokens 6 --temp 0)

| config | tok/s | SSD stall lives in | total stall | max stall | misses | prefetched |
|--------|-------|--------------------|-------------|-----------|--------|------------|
| **Baseline** (AHEAD=0, ping-pong off) | **0.11** | `get()` `wait` | 24264 ms / 3758 | 20.5 ms | 3758 | 0 |
| **Ping-pong ON** (robust barrier, fixed guard) | **0.11** | barrier | 22670 ms / 474 | **1443 ms** | **0** | 3711 |
| Ping-pong ON (simple barrier, first build) | 0.10 | barrier | 29168 ms / 474 | 1979 ms | 0 | 3711 |

`gets=3792`, `hit_rate=1.000`, `cache=5995 MB`, `RSS=25.5 GB`, `5 NPU ctx` every
row. Output **"KK" = 58933 ×5 preserved** (ping-pong is timing-only — same experts,
same math; verified empirically). Templated coherence gate verified separately
(`run_pp_on_coh.log`).

### What the barrier proved (the win inside the neutral result)
- **The FFN compute path is now stall-free.** `wait=0 ms / 0`, `misses=0` (vs
  baseline 24.3 s / 3758) — every `get()` is an instant cache hit because the
  barrier pre-loaded the layer's experts. The SSD cost moved entirely out of
  `get()` and into the barrier, exactly as designed. **This is the
  infrastructure any future SSD-speed or exact-prediction win needs.**
- **Total SSD stall dropped 7%** (22670 vs 24264 ms): the L+1 read overlaps L's
  NPU FFN, so some SSD is hidden. Real, but below the 0.11 rounding bucket.
- **Robust > simple barrier**: 47.9 ms avg / 1.44 s max (robust) vs 61 ms /
  1.98 s (simple). The re-queue-on-eviction + 100 ms timed wait + MTP guard
  cut both the average and the worst-case flood spike.

### Why it can't break the ceiling (the physics)
1. **SSD read > compute window.** 8 experts × 10 MB ÷ ~1.6 GB/s ≈ **52 ms** to
   read a layer's experts. The compute window that can overlap with reading
   L+1 = L's NPU FFN (37 ms) + L+1 host attn (7 ms) + L+1 router (~1 ms) ≈
   **45 ms**. Since **52 > 45**, the single serial SSD worker **cannot outpace
   compute even with perfect prediction** — a ~7 ms/layer excess is always
   exposed. Perfect-prefetch ceiling ≈ (37+7+7) ms ≈ 51 ms/L ≈ **0.18–0.20
   tok/s**. Ping-pong overlap only helps when SSD < compute; here SSD > compute,
   so SSD is the binding bottleneck regardless of *when* it reads.
2. **Markov (~80%) floods the worker.** `prefetch_layer_async(L+1, eids_L)`
   loads L+1's *predicted* experts; the ~20% that don't match `eids_{L+1}` are
   wasted reads that consume the worker's bandwidth. With ~9.6 reads/layer (8
   used + ~1.6 waste) vs a 45 ms window, the worker falls behind → the barrier
   waits behind a backlog (max stall **1.44 s**). The baseline wastes nothing
   (0% waste, pure reactive) so it ties despite exposing SSD serially.
3. **The 0.11 baseline is near-optimal** for this SSD+compute balance: SSD
   exposed serially, zero wasted bandwidth. Ping-pong matches it (neutral) by
   trading "serial SSD, 0% waste" for "overlapped SSD, 20% waste + barrier
   overhead" — a wash.

### The honest verdict + decision
**Ping-pong is NEUTRAL (0.11 vs 0.11); it does NOT break the ceiling**, for the
same root cause as the levers-1+3 net-negative finding (SSD bandwidth-bound,
worker serial, prediction imperfect). The barrier is correct, robust, and
useful infrastructure (it proves the FFN can be made stall-free and isolates the
SSD cost to a measurable barrier), but it is **kept env-gated, default OFF**
(`FST_HY3_PINGPONG` unset = the 0.11 path). Not shipped as default.

**The only levers that can beat 0.11** (none are ping-pong):
- **Faster SSD** — read 8 experts in < 45 ms (hardware; would let ping-pong
  reach ~0.18).
- **Smaller experts** — more quantization (< 10 MB/expert) → faster reads →
  SSD < compute (quality trade-off).
- **Near-exact next-layer prediction** — draft-driven `predict_and_prefetch_from_draft`
  (cuts the 20% waste), but SSD > compute still caps it at ~0.18 and the draft
  router costs compute. Complex, uncertain.
The decode floor (~89–114 ms/L depending on machine state) is SSD-bound; the
host attention that *was* the floor is already optimized (OpenMP+SIMD, 105→7 ms).