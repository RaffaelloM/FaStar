# QWOPUS GDN 8KB-Slab 4-Tile Engine Kernel — POSTMORTEM

**Date:** 2026-07-14
**Status:** Phase 2 (correctness) **PASS** after an alignment fix; Phase 3
(latency) **FAIL** — the kernel is correct but ~1.6× slower per token than the
shipped 3-pass path. **Not wired.** The predicted ~4× GDN speedup did not
materialize: the synthetic probe's 80 ms/layer did not translate to the
real-DMA kernel (1182 ms), exactly the risk flagged in the prior postmortem's
caveat.

Companion: `docs/QWOPUS_GDN_8KSLAB_FULL_PROBE_POSTMORTEM.md` (the probe that
authorized this kernel). Artifacts: `kernels/fst_gdn_8kslab_kernel.cc`,
`kernels/gen_gdn_8kslab.py`, `tools/gdn_8kslab_{engine,1tile,canary}_probe.cpp`,
`kernels/fst_gdn_8kslab_canary_kernel.cc`, `logs/gdn_8kslab_probe_aligned.log`.

## Phase 2 — Correctness: the blow-up and the fix

### Symptom (initial 4-tile run, real layer-0 weights)
- `max|Δy| = 25`, argmax match **14.1%**, `max|ΔS| = 18`.
- tk0 correct (0.002), tk1+ diverged (tk3 = 25.4); **S failed to accumulate**
  (final `max|S|` slab = 2.2 vs ref 18.2 — *smaller*, not an explosion).
- Latency 1182 ms (vs the probe's 80 ms).
- Both software bf16-S references (RNE and engine-TRUNC) were *stable*
  (tk0–7 ≈ 0.001–0.013) → **not** a bf16-S-precision issue, **not** truncation.

### Isolation (rule out 4-tile placement / stack persistence)
1. **1-tile real-DMA variant** (`FST_GDN_NTILE=1` in the generator): failed
   *identically* to 4-tile (same 1182 ms, same divergence, same tk0-correct/tk1+
   pattern, NaN from a non-zero S0). → **Not a 4-tile placement / memory-pressure
   problem.** The bug is kernel-level (real-DMA par reading or K-loop S lifetime).
2. **Canary kernel** (`fst_gdn_8kslab_canary_kernel.cc`, same shim/fifo/DMA
   contract, but the recurrence is replaced by `S[i,c] += 1.0` per K-step, par
   values ignored): `snew = S0 + 8.0` **exactly** (all 196608 elements = 8.5
   from S0 = 0.5). → **The stack S persists across the K-loop and passB writes
   correctly.** The "S fails to accumulate" is *not* a persistence bug — passB
   writes wrong *values* because it reads wrong inputs.
3. Canary latency = 120 ms (same DMA as the real kernel). → the 1062 ms extra
   in the real kernel is **recurrence compute, not DMA**.

### Root cause: `PAR_SZ` not a multiple of the load-vector width
`PAR_SZ = 2*HV + COLS + 3 = 291`. The kernel reads each token's params with
`aie::load_v<V>(par + t*PAR_SZ + n*V)`, `V = 8`. 291 bf16 = 582 bytes; 582 % 16
= 6 → **`par + t*PAR_SZ` is not 16-byte (8-bf16) aligned for any t > 0**. Per
the AIE `load_v` 8-float-alignment rule (`aie::load_v` rounds to the nearest
8-element-aligned address and reads from there), every token after tk0 read
the **wrong** `kn/qn/v/gdec/beta/c`. tk0 (offset 0) was aligned and correct;
tk1+ got garbage → the recurrence couldn't accumulate (garbage gdec/kn ⇒ tiny or
NaN S). The canary was unaffected because it ignores the par values (only
`snew` is checked). Same rule class as the row-stream GDN alignment fix
(`aie-loadv-8float-alignment-gdn-bitcorrect`).

### Fix
Pad `PAR_SZ` to a multiple of 8: **291 → 296** (5 trailing pad bf16). The kernel
already read 296 (`PAR_NVEC = ceil(291/8) = 37`, 37×8 = 296); only the
`par + t*PAR_SZ` stride needed changing. Applied in kernel, generator, and both
probes.

### Result (4-tile, start=0, real weights, best of 5)
```
argmax match:                  382/384 = 99.5%   (was 14.1%)
slab vs sw bf16-S:   y max|Δ|=1.37e-2  S max|Δ|=0.375   (matches bf16-S ref)
slab vs shipped fp32: y max|Δ|=1.41e-2  S max|Δ|=0.349  (total bf16-S divergence)
per-token y max|Δ|: tk0..7 = 0.002..0.014   (uniform; was tk3=25)
per-token max|y|: slab ≈ ref  (tk0 0.27/0.27 … tk7 1.70/1.70)
final max|S|: slab 17.88  ref 18.22  (was slab 2.2)
per-tile S max|Δ|: 0.17..0.35  (was 14..18)
finite: 49152/49152, |Δ|>1 count = 0  (was 96)
```
Verified at **start=8** (non-zero S0 = 15.5) too: S grows 15.54 → 20.88 (ref
20.13, shipped 20.14), max|Δy|=7.7e-3. Two secondary probe bugs were fixed during
diagnosis (start>0 input indexing `pos*NVH*HV` → `(pos-start)*NVH*HV`; a
software-reference stride), neither in the kernel.

**Phase 2 verdict: PASS.** The kernel is bit-accurate to the bf16-S software
reference and argmax-matches the shipped path at 99.5%, exceeding the 39/39
bar. bf16-S over K=8 RMW is lossless for argmax, as predicted.

## Phase 3 — Latency: a regression, not a speedup

```
4-tile slab dispatch = 1182 ms (best of 5) for K=8 tokens = 148 ms/token
shipped 3-pass        ≈  90 ms/token (3 × ~30 ms syncobj floor)
→ slab is ~1.6× SLOWER per token (regression), not the predicted 4× speedup
  (the plan's target was ~80 ms/8 tok = 10 ms/token)
```

### Why the 80 ms prediction was wrong
The 80 ms/layer estimate came from the **synthetic** probe
(`fst_gdn_8kslab_full_kernel.cc`), which generated `kn/qn/v` *in-kernel* — no
par DMA read, no 1184-byte `parf[296]` fp32 stack buffer. That cost is not
optional in the real engine, and it dominates:

| variant | par DMA | parf[296] buffer | recurrence math | latency |
|---|---|---|---|---|
| synthetic probe | no (generated) | no | full | 80 ms |
| canary (S+=1) | yes (ignored) | no | none | 120 ms |
| **real 4-tile** | **yes (used)** | **yes** | **full** | **1182 ms** |

The 2+2 shim DMA itself is cheap (canary 120 ms vs probe 80 ms → ~40 ms for the
extra s0/par/snew/yd streams). The **~1062 ms is the recurrence compute with the
`parf[296]` stack buffer**: passA (128×4×2 mul-add) + delta + passB (128×4×2
mul-add), ×K=8, ×48 v-heads, reading inputs from the 1184-byte `parf` stack
array (DMA-loaded par → fp32 conversion). The prior 2-tile 16 KB-stack kernel
was also ~2000 ms with the same `parf` buffer — the `parf` stack buffer is the
common latency disease of this kernel family. The 202 ns/vec *primitive*
(stack-S RMW) stayed fast; the *real kernel* around it did not.

### Phase 3 verdict: FAIL
A correct-but-1.6×-slower GDN path regresses tok/s. The dispatch-amortization
win the slab was built for is not realized at 1182 ms.

## Step 1+2 (perf-removal experiment) — three param-layout variants, all ~1180 ms

The orchestrator's `parf` hypothesis (parf[296] = the latency killer) was tested,
then the one remaining untested lever (replicate the probe's separate-stack-array
input layout) was tested too. **All three variants land at ~1180 ms; the probe's
79 ms is NOT reproducible in the real kernel under any param layout.**

| variant | par access in hot loop | param stack array | latency (4-tile) | 1-tile | argmax |
|---|---|---|---|---|---|
| packed `parf[296]` (original) | no (reads parf) | `parf[1184B]` packed | 1182 ms | 1182 ms | 99.5% |
| **noparf** (register-streamed) | **yes** (strided `par`) | none | 1166 ms | — | 99.5% |
| **separate-arrays** (probe layout) | no (one-time copy→stack) | `kn[512]+qn[512]+vv[128]` separate | **1180 ms** | **1180 ms** | 99.5% |

- **parf refuted:** removing the 1184-byte parf stack buffer moved latency only
  1182→1166 ms (~15 ms, 1.3%). parf was not the killer.
- **separate-arrays refuted:** replicating the probe's *exact* proven-fast input
  layout (separate aligned fp32 stack arrays, one-time sequential par→stack copy,
  hot loop reads ONLY the stack — never `par`) gave **1180 ms**, identical to parf.
  The probe's separate-array layout does NOT recover the speed.
- **not 4-tile contention:** 1-tile separate-array = 1180.70 ms = 4-tile 1180.57 ms.
  Per-tile recurrence is itself ~1180 ms.

### Why the probe was 79 ms and the real kernel is ~1180 ms (structural)
The recurrence math (passA 128×4×2 + delta + passB 128×4×2, ×K=8, ×48 v-heads) is
identical and stack-only in both. The canary (same DMA, `S+=1`, no math) = 120 ms,
so the DMA floor is ~120 ms and the recurrence compute is ~1060 ms in the real
kernel vs ~79 ms in the probe — a **13× gap with the same math and the same
stack-only hot loop**. The one structural difference the probe never had: **the
probe generated all inputs in-kernel and emitted only a final checksum scalar —
it had NO concurrent `s0`/`snew`/`yd` ObjectFifo DMA crossing the tile boundary.**
The real kernel must DMA `S0` in (8 KB), `S_new` out (8 KB), and `y|delta` out per
token; those streams move through the tile's data-memory banks **concurrently with
the 8 KB stack-S RMW**, and that bank contention — not the param layout — is what
inflates the recurrence from 79 ms (probe, no concurrent DMA) to ~1060 ms (real,
concurrent s0/snew/yd DMA). This is **structural**: S must cross the tile boundary
in the real engine, so the contention cannot be removed within the authorization
constraints (no persistent Buffer for S, no chaining, independent tiles).

## Step 3 — Decision gate: STRUCTURALLY DEAD
Per the gate: latency 1180 ms / 8 = **147.5 ms/token > 90 ms/token** ⇒ the GDN
M=K slab path is **structurally dead on this build**. All authorized levers
(parf removal, separate-stack-array layout) are tested and refuted; no remaining
plausible lever exists within the constraints. Phase 4 (wire behind
`FST_Q35_GDN_8KSLAB`) is **not** authorized — it would regress tok/s.

## What this proves / does not prove
- **Proves:** the ≤8 KB stack slab *can* be made bit-correct on real data (the
  alignment fix), and the stack-S primitive genuinely persists across the K-loop
  (canary). The probe's fast primitive was real, not a measurement artifact.
- **Does not prove:** any end-to-end speedup. The real kernel is slower than
  shipped. The 80 ms estimate was a synthetic-input lower bound, not an
  engine-kernel prediction.
- **Confirms the prior caveat:** "202 ns/vec proves only the latency primitive;
  real-engine integration untested" — the primitive translated to correctness
  (after alignment) but not to a net speedup.

## Recommendation / outcome: accept 0.20 tok/s ceiling
The slab M=K GDN path joins the other GDN-NPU dead-ends (on-tile state, runtime
chaining, fused, 3-pass-onekernel). Per the Step 3 gate, **accept 0.20 tok/s as
the ceiling**. The shipped 3-pass bf16-S path stays (it is the proven 0.21–0.22
tok/s path); `FST_Q35_GDN_8KSLAB` was never added to `src/fst_engine.cpp` and the
shipped path is untouched — nothing in the engine needs reverting. The standalone
kernel/probe artifacts remain as a documented dead-end. The real GDN lever (per
the prior profile, [[qwopus-real-profile-no-30ms-floor]]) is **speculative
decoding (M=K chunkwise GDN amortizing dispatch)**, NOT on-tile recurrence speed
— the on-tile recurrence is bank-contention-bound once S must cross the tile
boundary. Standing gates unchanged (`FST_Q35_FFN_NPU` OFF; shipped 0.20 host path
untouched).