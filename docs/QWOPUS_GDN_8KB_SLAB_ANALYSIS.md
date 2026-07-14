# QWOPUS GDN ≤8 KB Slab-Streaming — Structural Analysis

**Date:** 2026-07-14
**Status:** Analysis (Step 3 of the M=1-dead / M=K-preserve plan). A viable
≤8 KB-per-array slab-streaming structure **exists structurally** and breaks the
prior "structural impasse" chain, but viability hinges on **one untested cell**
in the test matrix. This doc records the analysis; the micro-probe that resolves
the empty cell is `kernels/fst_gdn_8kslab_full_kernel.cc`.

## The GDN recurrence is column-independent (the structural lever)

From `kernels/fst_gdn_scan_kernel.cc`, per (v-head, token), `S` is `[128,128]`:

- `a[j] = Σ_i S[i,j]·kn[i]`  → `delta[j] = (v[j] − gdec·a[j])·beta`
- `b[j] = Σ_i S[i,j]·qn[i]`  → `y[j] = gdec·b[j] + delta[j]·c`,  `c = kn·qn` (global scalar, no `S`)
- `S_new[i,j] = gdec·S[i,j] + kn[i]·delta[j]`

**Column `j` is fully self-contained**: `delta[j]`, `y[j]`, `S_new[:,j]` need only
`S[:,j]` (one 128-float column) + the shared full vectors `kn`, `qn` + the global
scalar `c` + `v[j]`, `gdec`, `beta`. **No cross-column reduction.** The prior
chunkwise column-split designs already exploited this ("column-split ⇒
independent per tile, NO cross-tile reduction"); what is new here is the
**per-array ≤8 KB** constraint.

## The structure that fits ≤8 KB + 2+2 shim + no chain

Per tile, hold **one column-slab `S[:, c0:c1]`** as a **stack-local** array in a
single-call internal-loop kernel (the pattern the 4tile-8k-proven postmortem
flagged as the "viable fast path" but never validated at this size):

| config | slab width | slab size | tiles for 128 cols |
|---|---|---|---|
| **bf16-S** (FST_Q35_GDN_BF16S, A/B-lossless for argmax) | 32 cols | 128×32×2 = **8 KB** | **4** |
| fp32-S | 16 cols | 128×16×4 = **8 KB** | 8 |

Per-tile shim = **exactly 2+2**: 2 MM2S (`S_slab` in ; packed `kn|qn|v|gdec|beta`
in) + 2 S2MM (`S_new_slab` out ; packed `y|delta` out). `c = kn·qn` computed once
per (v-head, token) — host side or one scalar tile — and broadcast. **No chaining,
no cross-tile forwarding, no persistent `aie.iron.Buffer`** (the slow disease).
bf16-S is the favorable config: 4 tiles (8-tile topologies coexisted before, so
4 fits the shim budget comfortably) and lossless for argmax.

## Why this breaks the prior "structural impasse"

The prior dead-end's impasse chain was:
`≤8 KB ⇒ ≤32 cols ⇒ ≥4 tiles ⇒ >2+2 shim ⇒ chain ⇒ persistent Buffer ⇒ slow`.
Two load-bearing links break under column-independence + the 2+2-per-slab packing:

1. **">2+2 shim ⇒ chain" is not forced** when each slab needs only 2 in / 2 out and
   slabs are independent — 4 bf16 tiles each at 2+2 do not exceed the device's shim
   columns (8-tile topologies ran before). Chaining in the old 8-tile design
   ("2 chains of 4") was a packet-sharing choice, not a topological necessity.
2. **"chain ⇒ persistent Buffer"** disappears entirely: with no chain, `S` lives as
   a stack-local array in a single-call kernel, not a forwarded ObjectFifo. The
   slow persistent-Buffer access (register-addressed MLIR-allocated, ~1.3 µs/vec
   at any size) is never entered.

## The one untested assumption — the empty cell

Reconstructed test matrix (from `qwopus-chunkwise-gdn-2tile-stack-deadend`):

| design | S size | storage | full recurrence? | result |
|---|---|---|---|---|
| 1-tile 32 KB Buffer | 32 KB | Buffer | yes | 4878 ms (slow) |
| 8-tile 4 KB Buffer | 4 KB | **Buffer** | yes | 1094 ms, bit-correct (slow) |
| 2-tile 16 KB stack | 16 KB | stack | yes | 2153 ms (slow, ≥16 KB threshold) |
| 4-tile 8 KB micro | 8 KB | **stack** | **NO — toy, 1 recur step** | 5.87 ms (fast — not representative) |

**The empty cell: ≤8 KB *stack* on a *full* recurrence.** The only fast result was
a toy one-step; every full-recurrence test used either ≥16 KB stack (over
threshold) or persistent Buffer (slow at any size, including 4 KB). The 8-tile
4 KB was slow *because it was Buffer, not because 4 KB is slow* — 4 KB *stack* on
a full loop was never measured. The 4-tile toy's "15-cyc/op serial K-loop
dependency" warns the toy may not extrapolate.

Viability hinges on one question: **does ≤8 KB stack RMW hold the toy's ~60 ns/op
on a real K=8 × 48-v-head recurrence, or does it degrade to the ~1.3 µs/vec ≥16 KB
rate once the loop is real** (memory-bank conflict, stack-pointer pressure, or the
serial K-dependency the toy hinted at)?

## Secondary M=K risk

The recurrence is **sequential across K tokens** (`delta_t` depends on `S_t`), so
M=K does not parallelize the recurrence — it puts K sequential RMW passes on the
slab in one dispatch. The dispatch-amortization win (cut 144 dispatches × ~30 ms
syncobj floor) is real *only if* the K× on-tile work still hides under the syncobj
floor. The 3pass-onekernel dead-end showed serializing on-tile work "exposes ~480
ms" — but that was ≥16 KB / Buffer. With fast ≤8 KB stack it might fit; untested.

## The probe that resolves it

`kernels/fst_gdn_8kslab_full_kernel.cc` — 1 tile, 32-col bf16 slab (8 KB stack),
single-call internal loop over 48 v-heads × K=8 tokens, **real GDN 3-pass
recurrence** (passA/delta/y/passB, the bit-correct 8-tile math), + a noop variant
for DMA-floor subtraction. **Probe input sourcing** (transparent): kn/qn/v/gdec/
beta and initial S0 are deterministic in-kernel values so I/O stays tiny and DMA
does not confound the stack-RMW latency — the recurrence *structure* is the real
GDN 3-pass; only the input *values* are synthesized. The stack-access pattern
(and thus the measured latency) is identical to a real-data engine kernel.

**Decision rule:**
- ≤8 KB stack holds ~tens-of-ns/op on the full loop → slab structure VIABLE →
  proceed to a 4-tile bf16-S engine kernel behind a gate.
- degrades to ~1.3 µs/vec → impasse stands → GDN M=K on-tile-state confirmed
  dead; the only live GDN lever is bf16-S on the *shipped row-streaming* path
  (halve S0/S2 volume, no on-tile state).