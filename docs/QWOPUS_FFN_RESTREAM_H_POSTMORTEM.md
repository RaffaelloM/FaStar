# NPU FFN Re-stream-h (Fork A / Phase 2a–2c) — CORRECT but COMPUTE-BOUND

**Date:** 2026-07-13
**Verdict:** The re-stream-h MXFP4 matvec kernel is **bit-correct** (rel err 2e-6,
argmax match) and **DMA-unblocked** (34 GB/s no-op vs the prior 0.29 GB/s wall),
but it is **catastrophically compute-bound** — the full matvec is **362× slower
than the DMA floor** (738 ms vs 2 ms). The scalar nibble extract + scalar fp32
dequant completely dominates, making the NPU FFN **slower than the host**
(~738 ms vs ~650 ms/matrix) and useless as-shipped. The bf16 MMUL path (Phase 2c /
Fork D) is mandatory to reach the 2 ms DMA floor.

## What we built (Phase 2a — Re-stream-h kernel)

Per the orchestrator directive (do NOT use chunked-h RMW; re-stream h from the
DMA stream). `kernels/fst_qwopus_ffn_kernel.cc` + `gen_qwopus_ffn.py`:

- **Packet = `[32 B header | h_chunk(4 KB) | M_n*G*17 weight bytes]` = 12832 B
  (< 16 KB).** Keeping the packet under 16 KB is the whole point — it stays on
  the 51 GB/s DMA side of the on-tile-array / packet-buffer wall that killed the
  prior 26240 B packet at 0.29 GB/s.
  - `h_chunk` = G*32 fp32 = 1024 fp32 = 4096 B (one k_chunk of K; G=32 groups).
  - weight run = M_n rows × G groups × 17 B = 16*32*17 = 8704 B.
  - header = `[row_base(int32) | kc(int32) | 24 B pad]` — row_base is the
    held-output row offset; kc lets the kernel CLEAR the held slot on the first
    k_chunk (the held output is a tile-memory fifo buffer the host does NOT
    pre-zero; the host only zeros the drain BO). 32 B header so h_chunk starts
    32 B-aligned for `aie::load_v`.
- **h is re-streamed every packet, never held on-tile** (a 20 KB held array is
  1470× slow). The kernel `load_v`s h from the stream at offset HDR.
- **8-tile N-split** of N=17408 (gate/up shape), K=5120 ⇒ k_chunks=5,
  n_chunks=136/tile, 680 pkts/tile. The kernel is K-agnostic (processes G groups
  per packet); the host controls packet count + row_base, so one kernel also
  serves down [5120,17408] (k_chunks=17, n_chunks=40) with different packing.
- **Accumulation** into the held S2MM output `o` (N_TILE=2176 fp32 = 8.7 KB, well
  under 16 KB ⇒ fast RMW): on kc==0 `o[row_base+r] = partial`, else `+= partial`.
  This is the proven held-output pattern; h itself is never RMW'd to tile memory.
- **fp32 MAC, scalar nibble extract** (the A3-proven pattern: scalar reads of a
  sole-input uint8 fifo deliver correct bytes — the shipped fused FFN relies on
  this). Dequant: `FP4_LUT[nib] * e8m0_to_f32(sc)`; e8m0 via bit-cast
  `(sc<<23)` = `2^(sc-127)`, bit-identical to host `ldexp`; sc==0 ⇒ dead block
  (skip). 8-lane `aie::mul` + `reduce_add` per 8 elements.
- Two xclbins: `fst_qwopus_ffn` (full) + `fst_qwopus_ffn_noop` (same DMA, no
  compute) to measure compute-vs-DMA.

## Measured result (Phase 2b)

`tools/qwopus_ffn_probe` (8 tiles, 69.8 MB input, host `mxfp4_matvec_f32` ref,
deterministic data with realistic e8m0 scales 120..140):

```
full matvec : 738.51 ms (0.09 GB/s)
no-op (DMA) :   2.04 ms (34.18 GB/s)
compute/DMA ratio full/noop = 361.62×   => COMPUTE-BOUND
correctness : max|Δ|=1.28  rel=2.03e-06  firstbad=0  argmax MATCH  -> PASS
```

Three findings:

### 1. Correctness — PASS (bit-correct within fp32 round-off)
`max|Δ|=1.28` on outputs of magnitude ~5e5 ⇒ relative error **2.0e-6**, argmax
**MATCH** (ref=16912=npu). The residual is fp32 summation-order round-off
(host OpenMP scalar-sequential vs NPU 8-lane `reduce_add`), not a bug. The
re-stream-h dequant + fp32 MAC reproduces `mxfp4_matvec_f32` exactly to fp32
precision. The A3 conclusion holds: `load_v`-as-uint8 + scalar nibble extract +
`FP4_LUT` is bit-correct.

### 2. DMA — UNBLOCKED (34 GB/s, the wall is gone)
No-op (DMA only) = **2.04 ms = 34 GB/s**. The chunked < 16 KB packet fixed the
prior 0.29 GB/s packet-buffer wall. (34 GB/s vs the Stage-1 51 GB/s ceiling —
the gap is per-packet BD-setup overhead at 680 pkts/tile × 8 tiles; acceptable.)
**The re-stream-h pattern works: h is streamed at 34 GB/s, never held on-tile.**

### 3. Compute — BOUND (362× the DMA floor) ← the make-or-break
The full matvec is **738 ms vs the 2 ms DMA floor = 362×**. The scalar nibble
extract + scalar fp32 dequant (build `wv[8]` from 4 scalar byte reads + 8 scalar
`FP4_LUT[nib]*scale`, per 8 elements, per group, per row, per packet) dominates.
**This makes the NPU FFN SLOWER than the host**: the host does the same matvec
(gate, 17408×5120) in ~650 ms with OpenMP; the NPU scalar-dequant kernel takes
738 ms. Moving the FFN to the NPU as-shipped is a **regression**, not a win.

The orchestrator's worry is confirmed: scalar nibble extract bottlenecks the
DMA. The 2 ms DMA floor (which would be a 325× speedup over host) is unreachable
with scalar dequant.

## Phase 2c — bf16 MMUL swap (documented, NOT implemented)

Per directive: document the fp32→bf16 MMUL change; ensure the structure does not
preclude it. The kernel structure (re-stream-h packet, held-output + kc-clear,
K-agnostic host packing) is **drop-in compatible** — only the MAC unit and
dequant precision change. Specifically:

1. **Dequant → bf16.** Replace `FP4_LUT[16]` (float) with `FP4_LUT_bf16[16]`
   and `e8m0_to_f32(sc)` with `e8m0_to_bf16(sc) = (bfloat16)(sc << 7)` (bitwise,
   both 8-bit bias-127 exponents — the shipped fused-FFN pattern,
   `fst_fused_dequant_gemm_kernel.cc:76`). The dequant target becomes a bf16
   weight tile, not scalar fp32.
2. **MAC → `aie::mmul`.** Replace the per-row scalar dot
   `acc += reduce_add(mul(wvec_fp32, hvec_fp32))` with native AIE2P
   `aie::mmul<M,K,N>` (bf16×bf16→fp32 accumulator). The natural tile for M_n=16
   outputs, 8 k's at a time is `mmul<16,8,8>` (16 rows × 8 k-cols × 8 batch) or
   the fused-FFN's `<16,64,64>` block. The mmul accumulator is fp32 (so the
   accumulation precision is fp32 — better than pure bf16; matches the bf16-S
   A/B gate, Fork D). The h operand loads as bf16 (the model's h IS bf16, so h
   streams as 2 KB chunks instead of 4 KB — halving h DMA).
3. **Inner-loop restructure.** The current "for each row r, for each group g,
   for each i4: scalar build wv[8], dot" becomes "for each tile of (M_n rows ×
   K-block): dequant a bf16 weight sub-tile into a register/short buffer, load
   a bf16 h sub-tile, mmul, accumulate." This is exactly the fused-FFN's
   `fst_fused_dequant_gemm_16x64x64` loop, adapted to re-stream h instead of
   holding A. The dequant-to-buffer step (the fused FFN's `B_buf`) stays < 16 KB
   (a 16×64×2 = 2 KB bf16 tile) — fast.
4. **What does NOT change:** the packet geometry, the re-stream-h h access, the
   held-output + kc-clear accumulation, the host packing, the K-agnostic design,
   the 8-tile N-split, the no-op/DMA measurement harness. Only the MAC unit +
   dequant precision change. The current kernel hardcodes nothing that blocks
   the swap; the `acc += reduce_add(mul(...))` line is the single MAC site.

**Expected payoff (bf16 MMUL):** the dequant+MAC moves from 8 scalar ops/element
to a vectorized mmul (~128 MACs/mmul at near-peak). If the kernel reaches the
DMA floor, the NPU FFN matrix goes from 738 ms → ~2 ms (per matrix), and the
full FFN (gate+up+down) from ~1.95 s (host, 40% of the token) → ~6 ms. That is
the only path to a real speedup; it requires the **Fork D bf16-S sign-off**
(bf16 weights/mmul is the lossless-argmax relaxation, A/B-proven 39/39).

## Decision

The fp32 re-stream-h kernel is a **correctness + DMA-infrastructure success**
but a **performance non-starter** (362× compute-bound, slower than host). Do NOT
wire `FST_Q35_FFN_NPU` with the fp32 scalar-dequant kernel. The viable
acceleration is the **bf16 MMUL swap (Phase 2c → Fork D)**: same re-stream-h
structure, native mmul, ~2 ms/matrix ceiling, gated on the bf16-S sign-off.

## Artifacts (repo-clean)
- `kernels/fst_qwopus_ffn_kernel.cc` — `ffn_matvec_restream` + `ffn_matvec_noop_stream`.
- `kernels/gen_qwopus_ffn.py` — 8-tile re-stream-h generator (full/noop/both).
- `kernels/fst_qwopus_ffn.xclbin` + `_insts.bin` (full), `fst_qwopus_ffn_noop.xclbin` + `_insts.bin`.
- `tools/qwopus_ffn_probe.cpp` — correctness + compute-vs-DMA probe (relative-error verdict, full-output snapshot before no-op).