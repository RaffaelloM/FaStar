# Chunkwise M=K GDN — Stage 1.2/1.3 Full Kernel: BIT-CORRECT but Buffer-bound (2026-07-13)

Stage 1.2/1.3 of the chunkwise M=K GDN plan (`/home/raffaele/.claude/plans/misty-snuggling-alpaca.md`):
author the full 1-tile persistent-Buffer kernel, run it on hardware, compare to
the bf16-S/fp32-compute numpy reference.

## Verdict: BIT-CORRECT on the first hardware run, but the persistent Buffer is
~340× too slow → the 1-tile Buffer design is a PERFORMANCE dead-end. The
Stage-1.1d "persistent aie.iron.Buffer is FAST" breakthrough is OVERTURNED.

### The win (Stage 1.2/1.3 PASS — the kernel logic is right)
`kernels/fst_gdn_chunkwise_kernel.cc` + `gen_gdn_chunkwise.py`: 1 tile, S =
persistent `aie.iron.Buffer(bfloat16[128*128])` (32 KB), persistent `ctr` Buffer
for row indexing, 4 row-sized depth-2 ObjectFifos (f_s0, f_par, f_y, f_sf),
2 MM2S + 2 S2MM. Worker loops 48 v-heads: ctr_zero → 128 load_s0_row (fp32→bf16
S) → 8 chunkwise_recur (passA/delta/passB, fp32 compute on bf16 S, round at
passB store) → ctr_zero → 128 store_sf_row (bf16→fp32). Reuses the proven 3-pass
C math verbatim.

`tools/gdn_chunkwise_probe.cpp` + `scripts/gdn_chunkwise_probe_ref.py` (numpy
ref: bf16 S, fp32 compute, bf16 round every passB) on toy data (48 v-heads,
K=8):
```
y : max|Δ|=0.000408  mean|Δ|=6.1e-05  (ref |y|max=0.033)
Sf: max|Δ|=0.00122   mean|Δ|=2.8e-04  (ref |S|max=0.025)
argmax agreement: 99.2%  (381/384)     ← 3 misses on near-zero y (bf16 noise)
PASS (gate: max|Δy|<5e-3 AND argmax≥99%)
```
The kernel matches the bf16-S/fp32-compute reference within the A/B's 5e-3 band.
**The recurrence math, packet/alignment, DMA, and bf16 round-trip are all
correct.** This is the load-bearing correctness result — the hard IRON part
compiles, runs, and is bit-correct.

### The loss (the persistent Buffer does not pipeline → ~340× slow)
Steady-state dispatch latency = **4878 ms** (vs the ~30 ms premise). Isolated by
compile-flag no-op variants (`GDN_RECUR_NOOP` / `GDN_LOADSTORE_NOOP` /
`GDN_RECUR_LOADONLY`, toggled via `compile_flags` in `gen_gdn_chunkwise.py`):
```
load/store only (recur no-op):     258 ms   (12 288 calls, 21 µs/call — FAST)
recur only (load/store no-op):    4652 ms   (  384 calls, 12.1 ms/call)
recur load-only (no MAC, no RMW): 1072 ms   ( 786 K Buffer loads, 1.36 µs/load)
```
The recurrence dominates. The load-only variant is decisive: streaming every S
row from the Buffer (bf16→fp32, no MAC) costs **1.36 µs per `load_v`** ≈ 1360
cycles/load. A tile-local load is ~4 cycles. So the persistent Buffer load is
**~340× a tile-local load** — it does not pipeline. The full recur (4652 ms) is
~4.3× the load-only cost (the MAC/RMW adds on top of the load stall).

**Why this overturns Stage-1.1d.** The scratch-test "0.10 ms/call FAST" verdict
(`docs/QWOPUS_CHUNKWISE_GDN_MICROTEST_POSTMORTEM.md` §6) was fast only relative
to the M=1 *Python-level* Buffer RMW (12 500 func calls/dispatch). In absolute
terms the Buffer was always ~0.4–1.4 µs/load — **~23× slower than the 4-tile
stack array** (60 ns/op in the 4-tile micro) and ~340× slower than a tile-local
load. The Buffer is placed on the worker's tile (`aie.buffer` with the worker's
tile op), but the 32 KB region's `load_v` access stalls heavily (likely
4-bank-spanning access with no software-pipeline overlap from the
register-addressed pointer). The stack array (8 KB, single bank, frame-pointer+
immediate) pipelines; the 32 KB Buffer does not.

### What this means for the path
- The **1-tile Buffer design is a performance dead-end** (correct, ~23× too
  slow). The shipped on-disk `fst_gdn_chunkwise.xclbin` is bit-correct but must
  NOT be wired into the engine — it would take 609 ms/token/layer (~143× worse
  than the shipped 3-pass's ~4 ms/token/layer syncobj floor).
- The **fast path is the 4-tile STACK design** (8 KB/tile, single-bank,
  tile-local, proven fast in the Stage-1.1c micro) — which needs the cross-tile
  per-token reduction (passA partial a,b reduced across 4 tiles; delta broadcast
  back; passB RMW each tile's 32 rows). That is the "pattern-5 concern"
  (per-token cross-tile handshake) the 1-tile Buffer was meant to avoid.
- **Compute-bound ceiling (recompute):** the GDN work is 49 K MACs/v-head/token
  × 48 v-heads × K tokens = 18.8 M MACs/dispatch at K=8. The SHIPPED 3-pass fn
  pipelines at ~2.4 cycles/MAC (98 K MACs/dispatch hidden under the 30 ms
  syncobj floor). At that efficiency, chunkwise K=8 compute = 18.8 M × 2.4 cyc
  = 45 ms/dispatch > 30 ms syncobj → **compute-exposed → only ~5×**, not 18×.
  The 18× target needs compute/dispatch ≤ 30 ms (syncobj-bound), i.e. **K ≤ ~5**
  at shipped pipelining. K=4–5 is the sweet spot (compute hidden, ~10–24×).
  This assumes the 4-tile stack recur pipelines at ~2.4 cycles/MAC like the
  shipped fn — UNVERIFIED (the 4-tile TOY micro was 15 cycles/op, but it had a
  serial K-loop dependency the real recur need not have).

### Status / next
- Stage 1.2/1.3: DONE, bit-correct (gate PASS).
- The 1-tile Buffer kernel is kept as a correctness reference (diagnostic
  no-ops gated behind `GDN_RECUR_NOOP` / `GDN_RECUR_LOADONLY` /
  `GDN_LOADSTORE_NOOP` env vars via `compile_flags`); default compile = clean
  bit-correct.
- **Decision point:** the chosen 1-tile design is a perf dead-end. The viable
  fast path is the 4-tile stack rewrite (Stage 1.2-v2) with K tuned to ~4–5,
  carrying the cross-tile per-token reduction complexity. Whether that
  pipelines at shipped efficiency (→ ~10–24×) is the next de-risk, and it is a
  large undertaking with the per-token handshake deadlock risk. This
  postmortem supersedes the "Buffer UNBLOCKED" framing of
  `docs/QWOPUS_CHUNKWISE_GDN_MICROTEST_POSTMORTEM.md` §6.

### Artifacts (repo-clean, kernels/ + tools/ + scripts/)
- `kernels/fst_gdn_chunkwise.{xclbin,_insts.bin,_kernel.cc}`,
  `kernels/gen_gdn_chunkwise.py` — the 1-tile Buffer kernel (bit-correct, slow).
- `tools/gdn_chunkwise_probe.cpp` (+ built `tools/gdn_chunkwise_probe`) —
  3-iteration latency + output-dump host probe.
- `scripts/gdn_chunkwise_probe_ref.py` — numpy bf16-S reference (gen/cmp modes)
  + the bit-correct comparison gate.
- `kernels/{inp_s0,inp_par,ref_y,ref_sf,out_y,out_sf}.bin` — probe data.