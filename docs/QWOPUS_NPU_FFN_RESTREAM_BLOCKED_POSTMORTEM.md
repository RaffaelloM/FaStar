# NPU FFN Re-stream-h (Fork A / Phase 2) — BLOCKED POSTMORTEM

**Date:** 2026-07-13
**Verdict:** The orchestrator-specified Phase 2 design — `[h(20 KB) | RPB weight rows]` in ONE uint8 stream per tile, fp32 `load_v` for h + scalar nibble extract — is **blocked on two independent axes**, both measured. The viable alternative is a significant redesign with multiple unproven IRON elements and a marginal ~0.3 tok/s ceiling. Decision deferred to the orchestrator.

## What we built

- `kernels/fst_qwopus_ffn_kernel.cc` — `ffn_matvec_rows_stream` (uint8 packet `[h|RPB rows]`, h read via `load_v`-as-float at offset 0, weights scalar-read, fp32 8-lane MAC) + `ffn_matvec_noop_stream` (same DMA, no compute, for compute-vs-DMA).
- `kernels/gen_qwopus_ffn.py` — 8-tile, one uint8 MM2S + one fp32 S2MM. RPB=2 (RPB=4 overflows 64 KB tile memory at depth-2). Packet = 26240 B.
- `tools/qwopus_ffn_probe.cpp` — 8-tile matvec vs host `mxfp4_matvec_f32` ref + compute-vs-DMA.
- `kernels/fst_ffn_diag_*` — isolated uint8-packet correctness diagnostic.

## Measured result — BROKEN

`tools/qwopus_ffn_probe` (228 MB input, 8 tiles, 1088 packets/tile):

```
matvec(full compute) = 779.49 ms   no-op(DMA only) = 779.82 ms   compute-share = -0.0%
DMA floor = 0.29 GB/s   (input 228.4 MB)
max|Δ| = 3.6e4 (row 95)  max rel = 50   argmax DIFF   => FAIL (garbage output)
```

Two independent failures:

### Axis 1 — CORRECTNESS: the uint8 ObjectFifo delivers garbage in this configuration

Diagnostic (`kernels/fst_ffn_diag_*`): 1 tile, 16 packets, ramp fill `inp[i]=i&0xFF`, core reads `p[off]` (scalar) + `load_v<8>` at offset 0 (vector).

| pkt size | byte[0] got | byte[0] expected | verdict |
|---|---|---|---|
| 8192 B | 96 | 0 (ramp byte[0] always 0, 8192%256=0) | garbage |
| 26240 B | 224 | 128 (`p*26240&0xFF` is only 0 or 128) | garbage |

Both scalar `p[off]` and vector `load_v<8>(reinterpret_cast<float*>(p))` return garbage (vector lanes = 4e21, 1e24, 3e29 — not the ramp bytes 0..31).

**The memory note "pure-uint8 fixes scalar reads / `load_v` is byte-perfect" is INCOMPLETE.** It does NOT hold for **uint8 input + float output + `load_v`-as-float h**. The proven fused-FFN uint8 path (`fst_fused_dequant_gemm_kernel.cc:94-127`) uses a **different** configuration: ONE uint8 carrying `[A(1024 BF16)|B(2304 B)]`, **bf16 `aie::mmul`** (not fp32 `load_v`), scalar `pB[k]` reads, and **no float/bf16 fifo alongside the uint8 input**. That config is dequant-proven (M=16 batched). My config is not. Not yet replicated.

### Axis 2 — SPEED: packets ≥16 KB hit the on-tile-array wall on the buffer itself

No-op (DMA only) = 780 ms for 228 MB = **0.29 GB/s**, vs Stage-1's 51 GB/s on 11520-byte (<16 KB) packets. The 26240-byte ObjectFifo buffer is ≥16 KB → the **same on-tile-array wall** that killed held-h (1470× slow) and chunkwise-GDN S applies to the **packet buffer**. h = 20 KB > 16 KB → **h cannot live in a single packet buffer**.

(Stage-1's `f_h` 20 KB float buffer DMA'd at 11.74 GB/s only because it was **unread**; once the core reads a ≥16 KB buffer, it's slow.)

## The structural impasse

**Both h-access designs are now dead:**
- **Held h** — 1470× slow (Stage-1, the ≥16 KB read-only array wall).
- **h-in-packet** — ≥16 KB buffer wall (0.29 GB/s) + uint8 garbage.

h = 20 KB exceeds the 16 KB fast threshold. This is the **same 16 KB wall** that killed chunkwise GDN's on-tile S — it now blocks the FFN input vector too. Every q35 projection has K=5120 → 20 KB h, so the wall blocks moving ALL of them to NPU this way.

## Viable alternative (not built — major redesign, ~0.3 tok/s ceiling)

1. **FLOAT fifos** (correct vector loads). Scalar reads of a float fifo return a ramp → the weight nibble extract **MUST be vectorized**: `load_v` the 16 weight bytes as a `uint8x16`, `aie::unpack` to split low/high nibbles, vector LUT for FP4 → no scalar `pB[k]`.
2. **Chunked h**: split h=20 KB into <8 KB chunks; each packet = `[h_chunk | weight columns for those groups]`, partial-sum across chunks into a small held output (RMW, must stay <16 KB). Packets <16 KB → fast DMA.
3. fp32 MAC.

Multiple unproven IRON elements (vector nibble extract via `aie::unpack`+LUT; chunked partial-sum RMW into a held output across many packets) → high iteration risk for a **marginal ~0.3 tok/s ceiling** (the report's own estimate, vs the 10× goal).

## Decision

**Blocked — orchestrator's call.** Do NOT wire `FST_Q35_FFN_NPU` yet. The specified design is blocked by axis 1 + axis 2; the viable alternative is a significant redesign whose payoff (~0.3 tok/s) is marginal against the 10× goal.

## Forks for the orchestrator

- **A1 — Replicate the fused-FFN's uint8 config** for M=1: single uint8 `[h|W]`, bf16 `aie::mmul` (pad M=1→8/16, wasting 8-16× compute but using the proven path), no float fifo alongside. Tests whether the fused config fixes the garbage. Risk: M-padding waste + mmul is a batched op, unclear for M=1.
- **A2 — Float + chunked h + vectorized nibble extract** (the viable path above). Biggest redesign, highest risk, ~0.3 ceiling.
- **A3 — Vectorize nibble extract on a FLOAT weight fifo first** (cheap correctness probe): confirm `load_v`-as-uint8 + `aie::unpack` returns correct nibbles from a float fifo, before building the full chunked kernel. De-risks the hardest IRON element.
- **D — Constraint relaxation (bf16-S gate, already A/B-proven lossless)**: the cleanest accelerant, unlocks bf16 NPU kernels and sidesteps the fp32-precision fight. Sign-off-gated.
- **E — Accept ~0.2 tok/s as the in-constraint ceiling.** Phase 1 (passB) and Phase 2 (FFN) as-specified are both blocked; the 16 KB wall now spans GDN state AND FFN input. Without a constraint relaxation or a chunked-h + vector-nibble-extract breakthrough, ~0.2–0.3 is the ceiling.

## Artifacts (repo-clean)
- `kernels/fst_qwopus_ffn_kernel.cc`, `gen_qwopus_ffn.py`, `fst_qwopus_ffn*.xclbin` + `_insts.bin`
- `tools/qwopus_ffn_probe.cpp`
- `kernels/fst_ffn_diag_kernel.cc`, `gen_ffn_diag.py`, `fst_ffn_diag_small/large.xclbin` + `_insts.bin` (correctness diagnostic)