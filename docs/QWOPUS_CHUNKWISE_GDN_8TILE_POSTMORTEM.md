# Chunkwise M=K GDN — 8-Tile Column-Split: BIT-CORRECT but Buffer-bound AGAIN (2026-07-13)

Follow-on to `docs/QWOPUS_CHUNKWISE_GDN_KERNEL_POSTMORTEM.md` (the 1-tile
Buffer dead-end).  Per the user directive ("usa tutte e 8 le tile", "tutti i
kernel sempre vettoriali", "lo speculative decoding non rallenta"), this stage
authors an **8-tile column-split** kernel to (a) use all 8 NPU tiles, (b) make
the recurrence independent per tile (no cross-tile reduction), and (c) shrink S
to 4 KB/tile (hoped to pipeline where the 32 KB Buffer did not).

## Verdict: BIT-CORRECT on the first hardware run, but the persistent
`aie.iron.Buffer` S does NOT pipeline at 4 KB either → still a perf dead-end.
Column-split was the right call for correctness/independence; it does NOT fix
the Buffer-access stall.  ~340× a tile-local load, identical to the 1-tile 32 KB
case.  The fast path is still stack-S in a single-call internal-loop C fn.

### The win (8-tile column-split is correct)
`kernels/fst_gdn_chunkwise_8tile_kernel.cc` + `gen_gdn_chunkwise_8tile.py`:
**2 chains of 4 tiles** (2 MM2S + 2 S2MM total — within the shim budget; the
4-separate-stream design failed placement "compute-peer DMA budget
unsatisfiable", fixed by combining S0+params → `f_in` and y+Sf → `f_out` as
unified PKT=328-float packets, 2 in + 2 out per middle tile).  S[128,128] bf16
split by COLUMN across 8 tiles → 4 KB/tile persistent Buffer.  Column-split ⇒
recurrence INDEPENDENT per tile (a[c]/b[c] for the tile's 16 c's need only the
tile's 16 cols + full kn,qn; c=kn·qn redundant; delta/y/passB elementwise) ⇒ NO
per-token cross-tile reduction, NO handshake deadlock.  Within-chain
CO=0,16,32,48.

`tools/gdn_chunkwise_8tile_probe.cpp` + `scripts/gdn_chunkwise_8tile_ref.py`
(the numpy ref math is REUSED verbatim — column-split output == full-128 ref;
only the DMA pack/unpack of 328-float unified packets is new):
```
y : max|Δ|=0.000408  mean|Δ|=6.1e-05  (ref |y|max=0.033)
Sf: max|Δ|=0.00122   mean|Δ|=2.8e-04  (ref |S|max=0.025)
argmax agreement: 99.2%  (381/384)     ← 3 misses on near-zero y (bf16 noise)
PASS (gate: max|Δy|<5e-3 AND argmax≥99%)
```
Identical accuracy to the 1-tile Buffer kernel.  **The 8-tile column-split
math, packet/alignment, 2-chain topology, unified-packet forwarding, and
bf16 round-trip are all correct.**  This is the load-bearing correctness
result — 8 tiles + column-split + 2-chain-of-4 compiles, places, runs, and is
bit-correct on the first hardware run.

### The loss (persistent Buffer S does not pipeline at 4 KB either)
Steady-state dispatch latency = **1094 ms** (vs ~30 ms premise, vs shipped 3-pass
~4 ms/token/layer syncobj floor → 137 ms/token/layer, ~33× WORSE than shipped).
Isolated by compile-flag no-op variants (`GDN_8T_RECUR_NOOP` /
`GDN_8T_LOADONLY` / `GDN_8T_FWD_NOOP` via `compile_flags` in the gen):
```
load/store + fwd (recur no-op):   208 ms   (12 672 calls, 16 µs/call — FAST)
+ S-stream load only (no MAC):    354 ms   → +146 ms for 98 304 S loads
                                              = 1.48 µs/load ≈ 340× tile-local
+ S RMW MAC (full recur):        1094 ms   → +742 ms dominated by passA/passB
                                              read-modify-WRITE of S
fwd cost (full − fwd-noop):       110 ms   (328-float vcopy × 3 hops, acceptable)
```
The LOADONLY figure is decisive and identical to the 1-tile 32 KB case
(1.36 µs/load there, 1.48 µs/load here): **the persistent `aie.iron.Buffer`
load_v does NOT pipeline, regardless of size (4 KB or 32 KB) or column-split.**
The 4 KB single-bank hope was wrong — the stall is the Buffer's
register-addressed MLIR-allocated access pattern, not the size or bank count.
The full recur's +742 ms is the passA/passB RMW (load_v S + store_v S per row,
per token) — the write side stalls the same way.

**Why the Stage-1.1c 4-tile micro (8 KB stack, 60 ns/op) was fast but this
isn't:** the micro's S was a **stack-local array inside ONE single-call C fn**
(entered once, looped 48 v-heads internally).  Here S is a **persistent
`aie.iron.Buffer` passed as a pointer** across 264 ExternalFunction calls per
v-head (the IRON worker loops in `range_(NVH)`, calling the C fn once per
packet).  A stack-local array cannot persist across those per-packet calls, so
the design used a persistent Buffer — which is exactly the non-pipelining
object.  THE STACK ARRAY IS THE FAST PATH (confirmed a third time); the
persistent Buffer is not, at any size.

### The structural tension (why 8-tile-stack is hard)
The fast path = stack-S in a single-call internal-loop C fn (one call streams
all input, loops internally, stack-S lives for the call).  But:
- The IRON worker calls the ExternalFunction **once per ObjectFifo packet** (it
  does the acquire/release in Python).  A single-call C fn that loops 48 v-heads
  needs ALL the input in one call — it cannot acquire ObjectFifos (that is an
  MLIR/IRON op, not a C intrinsic).
- Giving each tile its own shim stream (so a single-call C fn reads its own DMA)
  needs 8 MM2S + 8 S2MM — the shim budget is ~2+2 (the 4-MM2S overload broke
  placement before).  So at most 2 tiles get direct shim access; the other 6
  must be fed via core↔core chains.
- Core↔core chains force per-packet forwarding ⇒ the per-packet call structure
  ⇒ persistent Buffer ⇒ the non-pipelining stall.

So: **shim budget (2+2) ⇒ chains ⇒ per-packet calls ⇒ persistent Buffer ⇒ no
pipelining.**  Column-split (which removes the cross-tile reduction) does not
break this chain of implications.

### The viable fast path (next de-risk) — 2-tile column-split, single-call, stack-S
Drop to **2 independent tiles** (no chain), each 1 MM2S + 1 S2MM = 2+2 (within
budget).  Column-split 64 cols/tile ⇒ S[128,64] bf16 = **16 KB stack-local**
(fits the stack immediate limit [-32768,-64] with headroom, unlike 32 KB).
Each tile runs a **single-call internal-loop C fn** that streams its MM2S input
and loops 48 v-heads with stack-S (entered once, RMW'd across K=8, the proven
fast pattern).  No persistent Buffer, no chain, no per-packet calls.

- 2× compute parallelism (not 8×): 18.8 M MACs/dispatch ÷ 2 = 9.4 M MACs/tile.
  At the shipped ~2.4 cyc/MAC ⇒ 22.5 ms/tile < 30 ms syncobj floor ⇒
  **syncobj-bound ~30 ms** ⇒ the 18× premise holds (1 dispatch vs 144), IF the
  stack-S pipelines like the 4-tile micro.  2 tiles suffice because compute is
  under the syncobj floor; 8 tiles were only "needed" under the wrong
  single-tile-compute ceiling.
- Open IRON risk: the single-call C fn must stream ~800 KB/tile of input in one
  call.  Two options to de-risk: (a) `ObjectFifo.acquire(n)` batch-acquire (n
  up to depth) returns a contiguous indexable list — acquire per-phase batches
  (128 S0 rows × 16 floats, 8 params × 284, etc.) with depth-sized fifos of
  SMALL packets (16-float S0/y/Sf, 284-float par) so depth-128 ≈ 8 KB fits tile
  memory; pass the batch base pointer to the C fn, which strides internally;
  (b) one big MM2S-fed tile-local buffer the C fn reads via a plain pointer
  (no ObjectFifo).  (a) is more IRON-native; (b) escapes ObjectFifo entirely.
- The user's "8 tiles" directive was premised on the (wrong) single-tile
  compute ceiling; with stack-S pipelining, 2 tiles hits the syncobj floor and
  8 would not help.  This should be surfaced before proceeding.

### Status / next
- 8-tile column-split: DONE, bit-correct (gate PASS), but ~33× too slow — DO NOT
  wire (137 ms/token/layer).  Kept as a correctness/topology reference
  (diagnostic no-ops `GDN_8T_RECUR_NOOP` / `GDN_8T_LOADONLY` / `GDN_8T_FWD_NOOP`
  via `compile_flags`; default compile = clean bit-correct).
- **Decision point:** the persistent-Buffer approach is a dead-end at any
  size/split (proven 3×: 1-tile 32 KB, 8-tile 4 KB, and the M=1 fusion before
  that).  The only proven-fast S access is stack-local in a single-call C fn.
  The next de-risk is the 2-tile column-split + single-call + stack-S kernel
  (within the 2+2 shim budget, no chain), with the batch-acquire/direct-DMA
  streaming as the IRON risk to resolve first.  This supersedes the 8-tile
  design chosen under the wrong compute ceiling.

### Artifacts (repo-clean, kernels/ + tools/ + scripts/)
- `kernels/fst_gdn_chunkwise_8tile.{xclbin,_insts.bin,_kernel.cc}`,
  `kernels/gen_gdn_chunkwise_8tile.py` — the 8-tile 2-chain column-split kernel
  (bit-correct, slow; 2+2 shims, unified PKT=328 f_in/f_out).
- `tools/gdn_chunkwise_8tile_probe.cpp` (+ built `tools/gdn_chunkwise_8tile_probe`)
  — 2-BO host probe (inp_8t/out_8t), 3-iter latency + raw drain dump.
- `scripts/gdn_chunkwise_8tile_ref.py` — numpy ref (reused math) + pack f_in /
  unpack f_out for the 328-float unified packets, + the bit-correct gate.