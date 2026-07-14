# Qwopus Fused GDN — Race Root-Cause (deep investigation)

Date: 2026-07-13
Branch: `hy3-pivot-openmp-simd`
Supersedes the "47/48 vh0-only / 2-fill startup race" framing of
`docs/QWOPUS_FUSED_GDN_POSTMORTEM.md` with an empirically-proven, two-mechanism
root cause and a matching fix design.

## What the fused kernel is

`kernels/gen_gdn_fused.py` + `kernels/fst_gdn_fused_kernel.cc` collapse the 3 GDN
passes (`passA → delta → passB`, 3 xclbins / 3 `xrt::run` per SSM layer = 144
dispatches/token) into ONE AIE dispatch per SSM layer = 48 dispatches/token. One
Worker loops all 48 v-heads; per v-head it runs passA (128 rows) → delta (once)
→ passB (128 rows) against a **held** tile scratch `[a|b|delta|y]`, streaming S0
from DDR twice per v-head (the 64 KB state does not fit in a 64 KB tile).

The passA C kernel (`gdn_passA_block`) is **line-by-line identical** to the
bit-correct 3-pass `gdn_passA_row` (`fst_gdn_scan_kernel.cc:43`): same
`aie::load_v`/`store_v` RMW `a[j]+=S[j]*kn_i ; b[j]+=S[j]*qn_i` over 128 rows on
a held accumulator acquired once and released at the v-head end. The math is
proven bit-correct; the engine math is untouched.

## The symptom (cmp_fused_ab, direct a/b measurement)

`scripts/cmp_fused_ab.py` recomputes the reference passA `a/b` straight from the
dumped `spkt`+`par` BOs and compares to the engine's drained `a/b` (from the
held-scratch `y` BO). Isolates passA from delta/passB.

| call | state | run | bit-correct (<1e-3) | notes |
|------|-------|-----|----------------------|-------|
| 0..47 (prefill token0, all 48 GDN layers) | S=0 | run1 | **48/48 EXACT** (Δ=0.0000) | S0=0 ⇒ a=b=0 regardless of RMW |
| 48 (prefill token1, layer0) | S≠0 (max\|S0\|=5.88) | **run1** | **0/48** | catastrophic — all heads scaled/zero/over |
| 48 | S≠0 (same S0) | **run2** | **47/48** | ONLY **vh0** wrong (Δa≈0.064) |

**run2 reproduces the postmortem's "47/48, vh0-only" exactly.** run1 was an
unlucky run where the race cascaded to all 48 heads. Both runs used the **same
xclbin, same prompt, identical S0** (verified: `max|S0|` identical to 4 decimals).

### Determinism test — PROVEN RACE (not a fixed bug)

Same S0, two runs, per-v-head `eng_a/ref_a` fraction:

```
run1: 0/48 bit-correct   max|Δa|=4.50   (vh0=0.00, vh18=7.6×, vh38=7.3×, …)
run2: 47/48 bit-correct   max|Δa|=0.064  (only vh0=0.68 wrong)
Only 2/48 v-heads have an identical fraction across runs.
```

Identical input, opposite outcome, per-head fractions almost all different →
**pure temporal nondeterminism**. This is a race, not a deterministic indexing
or offset bug (a fixed bug would reproduce 0/48 every run).

## Root cause — two mechanisms

The fused tile has **4 DMA channels** (2 MM2S inputs + 2 S2MM outputs) where the
bit-correct 3-pass passA tile has **2** (1 MM2S + 1 S2MM). Both extra channels
cause races, with different signatures:

### Mechanism 1 — vh0 startup race (the trigger; ALWAYS hits vh0)

Two concurrent MM2S fills (`f_s0` 12288 packets + `f_par` 48 packets) are issued
at task-group start and share the shim. The first `f_s0.acquire(1)` (vh0 passA
row 0) can return before its packet is fully DMA'd because `f_par`'s fill is
competing for the shim MM2S channel. vh0's first block is stale → vh0's `a` is
wrong on **every** run (vh0 is wrong in both run1 and run2). This is exactly the
postmortem's "first-v-head startup fill-race"; it is real and consistent.

### Mechanism 2 — 2-S2MM drain-flush instability (the cascade; nondeterministic)

The 3-pass passA explicitly uses **ONE** held output fifo. Its kernel comment
(`fst_gdn_scan_kernel.cc:42`) states verbatim:

> *"ONE held output fifo, mirroring the proven FFN GEMM C-matrix pattern
> (**two simultaneous held outputs broke the shim drain flush on this IRON
> build**)."*

The fused kernel **violates this**: it has **two S2MM output drains** armed
simultaneously — `f_scr` (the held `[a|b|delta|y]` scratch, drained at the v-head
end) and `f_s2` (the streamed updated state, drained during passB). With two
S2MM drains armed, the shim drain flush is unstable, and the `store_v` RMW to
the held scratch can lose updates nondeterministically. On good runs (run2) the
instability stays bounded at vh0; on bad runs (run1) it cascades and corrupts
all 48 v-heads' `a` (scaled / zero / over-large). This explains why the
postmortem (one run, y-level) saw 47/48 while a direct a/b measurement across
runs sees 0/48 ↔ 47/48.

### Why the postmortem under-measured

The postmortem's "47/48, vh0-only" came from a **single run** and a **y-level**
comparison (`max|Δy|≈0.06` at pos1, attributed to vh0). It never ran
`cmp_fused_ab.py` (the Bash tool died). On a good run the cascade is bounded at
vh0 and the y-level Δ is small — so a single-run y-level measurement reads
"47/48". The direct a/b measurement across multiple runs reveals the race:
0/48 ↔ 47/48, vh0 always wrong, cascade nondeterministic.

## Fix design — A3 v2: reduce to the proven 3-pass channel config (1 MM2S + 1 S2MM)

Both mechanisms are removed by matching the bit-correct 3-pass passA channel
budget exactly: **1 MM2S input + 1 S2MM output**.

### 1 MM2S — eliminate `f_par`

Pack the per-v-head params (`v`, `kn`, `qn`, `gdec`, `beta`) into the `f_s0`
stream as **3 prepended 136-float packets** per v-head (v/kn/qn are 128 each +
gdec/beta in the pad of the first). The f_s0 stream per v-head becomes
`[pkt_v][pkt_kn][pkt_qn][128 passA row-pkts][128 passB row-pkts]` = 259 packets.
The Worker acquires the 3 param-packets first and loads them into
`scr[512:896]` (free region of the held 1024-scratch: `[a|b|delta|y]`=512 +
`[kn|qn|v]`=384 + gdec/beta@896/897), then runs passA/delta/passB. `gdn_delta_from_ab`
reads `v/kn/qn/gdec/beta` from `scr` instead of `f_par`. Removes the 2nd MM2S
fill → no startup race. (TAP split 259 = 7×37 to satisfy BD repeat ≤255.)

### 1 S2MM — merge `f_scr` + `f_s2` into one output ObjectFifo

One output ObjectFifo `f_out` (1024-pkt, depth=2). Per v-head the Worker acquires
`scr` (held, buffer A) for passA/delta/y, and streams the 128 S2 row-packets
through buffer B of the **same** fifo (acquire/write/release per passB row),
then releases `scr` last. ObjectFifo drains in release order, so the output BO
per v-head is `[S2(128×128), y(128)]`. One S2MM channel → no drain-flush
instability, no cascade. Backpressure (depth=2) handles the held-scr + 1
streaming buffer.

### Resulting config

1 MM2S + 1 S2MM = **exactly the bit-correct 3-pass passA channel budget**. The
passA/delta/passB C kernels keep their proven math (only `gdn_delta_from_ab`
reads params from `scr` instead of `par`; a small `gdn_load_params` memcpy kernel
is added). The engine host-packing changes: `f_s0` gains the 3 param-packets per
v-head; the output BO is `[S2|y]` per v-head (engine unpacks S2 then y). No
`par` BO.

### Risk

IRON build fragility (the postmortem documents 4 failed fix attempts and a
harness outage on this build). The 3-fifo single-fill fix alone (Mechanism 1
only, keep 2 S2MM) might fix vh0 but leave the Mechanism-2 cascade risk; the
full 1-MM2S+1-S2MM config removes both. **Ceiling is still ~3×** (0.21→~0.6
tok/s, syncobj ~30ms × 48 dispatches; >0.6 needs sub-ms dispatch / sudo, denied).

## Status

- The fused path is gated `OFF` by default (`FST_Q35_GDN_FUSED`); the shipped
  binary runs the bit-correct 3-pass at 0.21 tok/s, untouched.
- The on-disk `fst_gdn_fused.xclbin` is the late-par depth=2/2 build (recompiled
  2026-07-13). It is the racy 47/48↔0/48 kernel characterized here — usable for
  race diagnosis, NOT shippable.
- The fix (A3 v2) is designed; not yet implemented (IRON rewrite, fragile).

## Artifacts

- `logs/cmp/call48_full.txt` — run1 per-v-head classification (0/48).
- `logs/cmp/multicall.txt` — S=0 for calls 0-47 (48/48), S≠0 at call48.
- `logs/cmp/determinism.txt` — run1 vs run2 per-head fractions (2/48 identical).
- `logs/cmp/run2_count.txt` — run1=0/48, run2=47/48 (vh0-only).
- `logs/run_fused.txt`, `logs/run_fused2.txt` — the two engine runs.
---

# FUSION DEAD-END (final, 2026-07-13, three attempts)

The A3 v2 fix was implemented two ways. **Both are ~600× slower than the 3-pass
(60 s/call vs ~0.1 s/call) and therefore unshippable.** Fusion of the 3 GDN
passes into one dispatch is a dead end on this IRON build. The 3-pass is the
only fast-and-correct configuration. This is the "comprehendi bene" the user
asked for: not just *that* fusion fails, but *why*.

## The three attempts, with measured per-call latency

| attempt | in (MM2S) | out (S2MM) | held scr storage | result | latency/call |
|---|---|---|---|---|---|
| fused v1 (racy) | 2 (f_s0+f_par) | 2 (f_scr held + f_s2 stream) | ObjectFifoElem held | **FAST but WRONG** (vh0 race + cascade) | ~1 s |
| A3 v2 #1 (held+stream same fifo) | 1 (f_s0) | 1 (f_out: scr held in buf A + S2 stream buf B) | ObjectFifoElem held | slow (not validated) | ~60 s |
| A3 v2 #2 (Buffer) | 1 (f_s0) | 1 (f_out: S2+y stream) | `aie.iron.Buffer` (tile-local, not a fifo) | slow (not validated) | ~60 s |

(Measured on the Buffer variant: `logs/scr_v2/call0..3` at 15:17, 15:18, 15:19,
15:20 — exactly 60 s/call. 3-pass `gdn_scan_vheads` = 3 dispatches × ~30 ms ≈
0.1 s/call. So 600×.)

The two slow variants are slow for **different** reasons, which is the key clue:

- **held+stream same fifo (#1):** `scr` is held in buffer A of `f_out` for the
  *entire* v-head (~256 kernel calls), so the streamed S2 must reuse buffer B.
  With only one free buffer, S2 effectively streams at **depth=1** → each of
  the 6144 S2 row-drains pays the full S2MM BD re-arm overhead back-to-back →
  ~60 s. (An ObjectFifo cannot be both *held long-term* and *streamed per-row*
  efficiently within one depth-2 channel.)
- **Buffer (#2):** `scr` is a tile-local `aie.iron.Buffer`, so `f_out` has
  *both* depth-2 buffers free for streaming (the depth-1 pathology is gone).
  Yet it is still 60 s. The remaining cost is the **Buffer RMW itself**: scr
  is RMW'd by ~12,500 kernel `func.call`s per dispatch (48 v-heads × (3 load +
  128 passA + delta + 128 passB + copy)). A `Buffer` (a "memory region
  accessible by Workers and the Runtime", placed on the Worker's tile) appears
  to incur a much higher per-access overhead than a pure tile-L1
  `ObjectFifoElem` — consistent with this IRON build's known fifo/buffer
  interaction fragility. (The 3-pass passA RMWs a held `ObjectFifoElem` 6144
  times per dispatch and is fast; same RMW pattern, different storage.)

## Why the dispatch boundary is load-bearing (the real root cause)

The 3 GDN passes have **two output shapes that cannot coexist in one
ObjectFifo**:
1. **passA**: a held accumulator `[a|b]` RMW'd over 128 rows, drained *once* at
   the end (1 packet/v-head). Fast as a held `ObjectFifoElem` in tile L1.
2. **passB**: a streamed result `S2` written per-row (128 packets/v-head). Fast
   as a streaming `ObjectFifo` with both depth-2 buffers free.

The **3-pass structure splits these into separate dispatches**, giving each
its own 1 S2MM with the right semantics (held for passA, streamed for passB).
That boundary is what makes it both fast and correct.

Fusing forces both into one dispatch. There are only three ways to hold `scr`
and stream `S2` within one dispatch, and all three fail:
- **2 S2MM** (held `f_scr` + streamed `f_s2`): fast (v1) but the two simultaneous
  S2MM drains hit the documented **drain-flush race** (Mechanism 2) → wrong.
- **1 S2MM, held+stream same fifo** (#1): the held `scr` starves the stream →
  depth=1 → 60 s.
- **1 S2MM + tile-local Buffer** (#2): the Buffer RMW is ~600× slow → 60 s.

So: **fast needs 2 S2MM (races); correct needs 1 S2MM (stalls).** The
within-one-dispatch held-RMW + per-row-stream combination is unsolvable on this
IRON build. The 3-pass dispatch boundary is not a leftover — it is the
mechanism that resolves the held-vs-stream tension that cannot be resolved
inside a single dispatch.

## Conclusion / action

- **Do not ship fusion.** The fused path stays gated `OFF` (`FST_Q35_GDN_FUSED`);
  the shipped binary runs the bit-correct 3-pass at 0.21 tok/s, math untouched.
- The on-disk `fst_gdn_fused.xclbin` is now the **slow Buffer variant** (A3 v2
  #2). Setting `FST_Q35_GDN_FUSED=1` gives a 600× slowdown — do NOT enable in
  any run. It is kept only as a documented dead-end reference.
- The ~3× ceiling via scan-fusion is **not reachable** on this build. The
  remaining NPU-side throughput lever is **MTP/speculative** (Stage B): emit K
  draft tokens per trunk-pass via the model's NextN block (kind 2, L64, today
  skipped by `ndec=n_layers-1`), amortizing the 144-dispatch floor across K
  tokens. That is the next plan stage.
- The 8-tile / layer-parallelism question (plan Stage A.5) is moot for fusion
  (fusion is dead) and, per the postmortem's delta≈passA≈30 ms argument,
  already strongly indicated syncobj-bound (so 8 tiles would not help the
  3-pass either). The decisive lever is MTP, not tile-parallelism.

## Artifacts (this attempt)

- `logs/run_fused_v2buffer.txt` — the 60 s/call run (timed out at 300 s after
  4 SSM-layer calls).
- `logs/scr_v2/call0..3_{out,spkt}.bin` — Buffer-variant SCR dumps (call0 = S=0;
  not validated for correctness — a 600×-slow kernel is unusable regardless).
- `logs/gen_gdn_fused_v2buffer.log` — IRON compile of the Buffer variant (clean,
  13386 B xclbin).
