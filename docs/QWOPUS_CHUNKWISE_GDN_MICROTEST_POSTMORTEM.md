# Chunkwise M=K GDN — Stage 1.1 Micro-Test: 4-TILE 8 KB SPLIT PROVEN (2026-07-13)

Stage 1.1 of the chunkwise M=K GDN plan (`/home/raffaele/.claude/plans/misty-snuggling-alpaca.md`)
de-risks the #1 IRON-authoring risk before the full kernel: can the GDN
recurrence state `S[128,128]` be held **on-tile** (off the DMA channels, to
avoid the 2-S2MM race that killed M=1 fusion) and RMW'd with `aie::load_v`/
`aie::store_v` **FAST** (~30 ms, NOT the 60 s the M=1 `aie.iron.Buffer` RMW
hit) and **NUMERICALLY CLEAN** (not the garbage static `.bss` produces)?

**Verdict: FEASIBLE — the 4-tile 8 KB split is PROVEN end-to-end at the
micro-test level.** This **OVERTURNS** the earlier "INFEASIBLE" verdict
(below): the 32 KB single-tile stack is indeed unrealizable, but the 4-tile
8 KB/tile split clears every blocker and is clean + fast. The decisive test:
`tools/gdn_4tile_probe` → **all 48 v-heads total == 4128.0, latency 5.87 ms**.

## The discovery chain (what was tried, what broke, what proved out)

| S storage / split | Status | Why |
|-------------------|--------|-----|
| **single-tile 32 KB stack** `bfloat16[128*128]` | ❌ INFEASIBLE | AIE2P load/store immediates limited to `[-32768,-64]` (9-bit signed, 64-byte step); Peano emits no register+register fallback → `-32896 out of range` backend crash. (Stands.) |
| **`aie.iron.Buffer`** (MLIR tile mem) | ❌ 600× SLOW RMW | M=1 finding (`12500 func calls/dispatch`). |
| **static `.bss`** | ❌ garbage with `aie::load_v` | standing `static-bss-incoherent-with-loadv` rule. |
| **2-tile 16 KB/tile** | ❌ BANK-ALLOCATION WALL | AIE2P tile data mem = 4 banks × 16 KB; a ≥16 KB stack+frame spans bank 0 into bank 1 where the ObjectFifo buffer is placed → `out_buff_0 at address 0x4000 overlaps with stack (size: 20480 bytes)`. Both bank-aware and basic-sequential allocators fail. |
| **4-tile 8 KB/tile** (32 rows each) | ✅ **PROVEN** | 8 KB fits bank 0 with frame room, leaving banks 1-3 for ObjectFifo buffers; far inside the immediate limit. |

## The three load-bearing IRON patterns discovered (each PROVEN)

### 1. core↔core ObjectFifo WORKS on this NPU2/IRON build
A worker→worker stream (`W0.prod` → `W1.cons`) routes over the **stream-switch**
(separate from shim DMA). PROVEN for 1 hop (`tools/gdn_cross_probe`, 48/48) and
for a 3-hop chain (T0→T1→T2→T3, the 4-tile test). This was the 2-tile split's
#1 risk and it is RESOLVED — worker-to-worker ObjectFifos are real on this build.
(Corroborates the research's hope that cross-tile traffic can avoid the shim DMA
race; refutes the old postmortem's "unproven, likely dead" framing.)

### 2. 8 KB stack RMW is FAST (~1.6–6 ms), NOT 60 s
The M=1 `aie.iron.Buffer` 600× slowdown does **not** apply to **stack-local
arrays**. A `bfloat16 S[32*128]` (8 KB) RMW'd K=8 × 48 v-heads runs in
1.61 ms (single-tile) / 5.87 ms (4-tile). Stack arrays are plain tile data
memory with fast `load_v`/`store_v`; the Buffer abstraction (MLIR-allocated,
12500 func calls/dispatch) was the slow path. The chunkwise speed premise holds.

### 3. THE KEY PATTERN: single-call internal-loop (avoids the repeated-large-frame breakage)
**A worker core that calls an external C function with a LARGE stack frame
(8 KB array) REPEATEDLY in a `range_(N)` loop BREAKS after the first iteration**
— the SP/call restoration fails, so v-heads 1..N-1 see garbage/stale state.
This was the hardest bug of the stage: it is NOT a simple stack-size overflow
(stack_size up to 15 KB all failed; MLIR placement was clean — `out_buff_0` sat
adjacent to the stack with no overlap). It is the **re-entry** of the large
frame that breaks.

**The fix (and the full chunkwise kernel's structure anyway):** the worker
calls the C function **ONCE** (`range_(1)`), and the C function loops the
48 v-heads **internally** — the 8 KB frame is entered once, so the SP never
restores between v-heads. Helpers that operate on `S` take `S*` (a passed-in
pointer) and have tiny frames, so they can be called freely inside the loop.

```cpp
extern "C" void gdn_micro_t0(const float *in, float *xout) {
    bfloat16 S[QUART * HV];            // 8 KB — ONE frame for all 48 (no re-call)
    for (int v = 0; v < NVH; ++v) {
        const float gdec = in[v*INPKT + 0];
        xout[v]       = gdec;
        xout[NVH + v] = gdn_quart_partial(S, gdec, 0);  // helper takes S*, tiny frame
    }
}
```
Worker: `for _ in range_(1): pi=fin.acquire(1); px=fc.acquire(1); k0(pi,px); …`

## Shim discipline (PROVEN minimal-cross pattern)

**ONE shim MM2S + ONE shim S2MM total.** Four separate shim MM2S (one per tile)
overload the shim DMA → `qds_device::wait() unexpected command state`. The fix:
T0 takes the single shim MM2S input; T3 drains the single shim S2MM output;
T1/T2 are compute-only (core↔core fifos, no shim). gdec + the per-token params
**propagate down the core↔core chain** in the cross packet (not via separate
shim feeds). This keeps every tile's shim at ≤1 MM2S + ≤1 S2MM (race-free).

## The decisive 4-tile test

`kernels/fst_gdn_chunkwise_4tile_micro_kernel.cc` + `gen_gdn_chunkwise_4tile_micro.py`:
4 workers, 8 KB stack each (32 rows of S), single-call internal-loop over 48
v-heads, 3-hop core↔core chain carrying ONE 96-float packet per tile pair
`[gdec(48) | partials(48)]` (gdec rides the chain; the running partial sum
accumulates one tile's contribution per hop). Toy math: fill rows `r←(r+1)`,
RMW K=8 with `gdec=0.5`, sum; per-tile partials 264/776/1288/1800 → total 4128.

```
[probe] vh  0 total=4128 (Δ=0)
[probe] vh  1 total=4128 (Δ=0)
...
[probe] dispatch latency = 5.87 ms
[probe] PASS: all 48 v-heads total == 4128.0  (8KB stack RMW + 3-hop core<->core chain OK)
```

This PROVES, at micro-test scale, the entire chunkwise 4-tile premise:
- the 4-tile S split (32 rows/tile, 8 KB) works;
- the cross-tile partial-sum reduction works;
- the single-call internal-loop pattern works across 4 tiles;
- all fast (~6 ms).

### 4. K=8 re-call of the 8 KB-frame function is CLEAN (the recurrent-case enabler)
The 3 patterns above were proven for the ONE-SHOT reduction (no recurrence). The
full chunkwise kernel has a **per-token recurrent cross-tile reduction** (passA
`a,b=Sᵀ@kn,Sᵀ@qn` reduced across tiles; `delta` broadcast back) — token t+1's
passA depends on passB(t)'s S, which depends on delta(t) from the chain. The
single-call internal-loop pattern does **NOT** extend: batching all K tokens
into one ObjectFifo acquire deadlocks (token t+1 can't start until the chain
finishes token t). So the worker core must drive **per-token** fifo handshaking
→ call the 8 KB-frame C function **once per token** (K=8 re-calls).

The repeated-8 KB-frame-call breakage was observed at N=48 in the ORIGINAL
4-tile per-v-head test (worker `range_(48)`, C fn 1 v-head/call, per-v-head
cross-tile handshakes). That was **misdiagnosed as a re-call-count limit**.
The decisive re-test (`tools/gdn_8kstack_recall_probe`, the 8kstack kernel —
48 v-heads INTERNAL per call, NO cross-tile — re-called N times) **PASS at
N=8, 16, 32, AND 48** (all N×48 == 264.0). So the 8 KB frame survives 48
re-calls; SP does NOT drift. The original N=48 breakage was the **cross-tile
per-v-head ObjectFifo handshake structure**, not the re-call count. The
single-call-internal-loop fix (pattern 3) was correct for the one-shot case
but the re-call itself is NOT the blocker — per-token re-call at K=8 is clean.

**Caveat — O(N²) re-call overhead:** the recall test latencies are
89.78 ms (N=8) → 356 (N=16) → 1424 (N=32) → 3204 (N=48): **4× per 2×N**,
i.e. O(N²). The ObjectFifo depth grows with N (depth=N+2), and the
allocation/drain is O(N²). For K=8 (depth-10) this is ~89 ms (acceptable —
~11 ms/token/layer → ~1.9 tok/s, ~9× over 0.21); for large N it is fatal.
So the full kernel must keep the per-dispatch re-call count at K=8 (one
re-call per token), NOT per-v-head (which would be 48×8=384).

### 5. S-persistence across per-token re-calls is the real full-kernel blocker
Stack memory is FREED when the C function returns — S does **not** persist
across re-calls (the toy fill re-initialized S each call, masking this). For
the real recurrence, S must persist across the K=8 per-token C-fn calls. The
single-call internal-loop (S persists on stack across the K-loop) is
incompatible with per-token worker-driven cross-tile fifos (the C function
cannot do ObjectFifo acquire/release). So the full kernel needs a **persistent
on-tile scratchpad** for S that survives across the per-token re-calls:
- NOT a stack array (lost on return) — but a **held ObjectFifo element** (one
  8 KB element per tile, acquired once and held for the whole dispatch),
  register-addressed (NOT the frame pointer → no [-32768,-64] immediate
  limit → a 32 KB full-S single-tile scratchpad would also be reachable this
  way), and accessed with plain `load_v`/`store_v` (fast, like the stack array
  — NOT the `aie.iron.Buffer` Python abstraction that was 600× slow).
- This held-scratchpad pattern is UNTESTED. It is the next micro-test
  (Stage 1.1d) before authoring the full kernel: an 8 KB held ObjectFifo
  element, RMW'd across N re-calls of a C function that takes the held pointer,
  verifying (a) it persists across re-calls (state carries over), (b) it is
  FAST (not the Buffer slowdown), (c) register-addressing avoids the immediate
  limit. If the held scratchpad works, the full 4-tile chunkwise kernel is
  unblocked: 4 tiles × 8 KB held S, per-token re-calls (K=8), forward+reverse
  cross-tile reduction (one big packet per token, 48 v-heads batched).

### 6. THE BREAKTHROUGH — persistent aie.iron.Buffer (C-kernel pointer access) is FAST + PERSISTENT + escapes the immediate limit
Stage 1.1d (`tools/gdn_scratch_test_probe`, `kernels/gen_gdn_scratch_test.py`):
an `aie.iron.Buffer` (named on-tile memory region) as S, passed to the C
function as a **pointer** and RMW'd with `load_v`/`store_v`. Results:
- **PERSISTENT across re-calls:** S[0] = 0, then 8 re-calls each `S[0] += 1` →
  per-call out = 1,2,3,4,5,6,7,8. State carries over (unlike a stack array,
  whose frame is freed on return).
- **FAST:** 0.80 ms / 8 re-calls = **0.10 ms/call** — faster than the stack
  array (1.6 ms), NOT the 600× slowdown.
- **Escapes the immediate limit:** a Buffer is **register-addressed** (the
  acquired pointer is the base, NOT the frame pointer), so `load_v(ptr+offset)`
  uses register+small-immediate — NO `[-32768,-64]` stack-immediate ceiling.
  A **32 KB** Buffer (`SLEN=8192` fp32) compiles, runs, persists, 0.86 ms/8
  re-calls. So **one tile holds the full 32 KB bf16 S[128,128].**

The M=1 "aie.iron.Buffer is 600× slow" finding (`QWOPUS_FUSED_GDN_RACE_ROOTCAUSE.md`,
"12500 func calls/dispatch") was **Python-level Buffer RMW** (the Buffer object's
`__getitem__`/`__setitem__` Python methods), NOT C-kernel-direct-access. A C
function with a Buffer pointer + `load_v`/`store_v` is plain fast tile memory —
same as the proven stack array, but **persistent across re-calls** and
**immune to the stack immediate limit**.

**This SIMPLIFIES the full kernel dramatically — the 4-tile split and the
entire cross-tile reduction are UNNECESSARY:**
- ONE tile, S = persistent Buffer (32 KB bf16 = full S[128,128]), no 4-tile
  split, no cross-tile partial-sum/delta reduction, no per-token cross-tile
  handshake deadlock (pattern-5 concern dissolved).
- The recurrence C function has a **SMALL frame** (S is a passed Buffer
  pointer, not a stack array) → the 8 KB-frame re-call breakage (pattern 3)
  does NOT apply → many small-frame re-calls are fine (like the shipped
  6144 per-row calls).
- S0/S_final stream per-v-head (48 v-heads; one v-head's S = 32 KB fits one
  packet or 128 row-packets); the K=8 recurrence runs on the persistent
  Buffer in 1 call per v-head (internal K-loop) or per-token re-calls — all
  small-frame, no deadlock (1 tile, no cross-tile).

The 4-tile cross-tile work (patterns 1-2, the 4-tile micro-test) is now
SUPERSeded for the full kernel — it remains a proven mechanism but is not
needed. The persistent Buffer is the load-bearing enabler.

## What the full kernel adds (Stage 1.2, now unblocked)

The micro-test's toy RMW (fill + scale) becomes the REAL GDN recurrence —
`passA` (`a,b = S@kn, S@qn`), `delta` (`kvm, delta, y`), `passB`
(`S += kn⊗delta`) — per K=8 tokens, reading/writing S rows from the stack-local
`Sbuf`. The cross-tile reduction carries per-token **partial `a,b`** (each
tile's 32-row contribution) reduced to the full 128 via the same 96-float-ish
cross packets; `delta` is broadcast/full. The math is reused verbatim from the
proven 3-pass C kernels (`fst_gdn_scan_kernel.cc`); only the S base pointer
moves (DDR packet → stack-local `Sbuf`) and S storage precision (fp32 DDR →
bf16 on-tile, the A/B-validated lossless mitigation). K=8 fixed (compile-time).

## Artifacts (repo-clean, in `kernels/` + `tools/`)
- `kernels/fst_gdn_cross_micro.{xclbin,_insts.bin,_kernel.cc}`,
  `gen_gdn_cross_micro.py`, `tools/gdn_cross_probe.cpp` — 1-hop core↔core proof.
- `kernels/fst_gdn_8kstack_micro.{xclbin,_insts.bin,_kernel.cc}`,
  `gen_gdn_8kstack_micro.py`, `tools/gdn_8kstack_probe.cpp` — single-tile
  single-call internal-loop proof (8 KB stack, 48 v-heads, 1.61 ms, all==264).
- `kernels/fst_gdn_chunkwise_4tile_micro.{xclbin,_insts.bin,_kernel.cc}`,
  `gen_gdn_chunkwise_4tile_micro.py`, `tools/gdn_4tile_probe.cpp` — the
  decisive 4-tile reduction test (5.87 ms, all==4128).

## Impact

The chunkwise M=K GDN path is **NOT a dead-end**: the 4-tile 8 KB split
realizes the held-S-off-DMA-channels premise that M=1 fusion could not. The
144-dispatch/token GDN floor is **NOT structural** — the K=8 chunkwise kernel
(Stage 1.2) should cut it to ~48 dispatches/token (1 dispatch/SSM layer), the
prerequisite for the ~18× speculative-decoding win (Stage 2). Proceeding to
Stage 1.2 (`gen_gdn_chunkwise.py` + `fst_gdn_chunkwise_kernel.cc`).

Supersedes the earlier "INFEASIBLE / structural floor" verdict recorded here
(preserved above in the git history). The precision A/B (`FST_Q35_GDN_BF16S`,
39/39 argmax-identical) and the parallel-scan research verdict both STAND; the
IRON-authoring path they assumed is now realizable via the 4-tile 8 KB split +
single-call internal-loop pattern.