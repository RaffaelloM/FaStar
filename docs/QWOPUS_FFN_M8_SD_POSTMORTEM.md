# QWOPUS FFN M=8 Speculative-Decoding — MEASURED WIN (compute-bound, 1.76× over host)

**Date:** 2026-07-14
**Status:** Probe-only WIN. NOT wired. `FST_Q35_FFN_NPU` stays OFF (shipped 0.20 tok/s host path untouched).
**Verdict:** M=8 16-tile N-split = **48.5 ms for 8 tokens** vs host OpenMP M=8 85.6 ms → **1.76× faster** (6.07 ms/h vs 10.7 ms/h). 8/8 argmax MATCH both gate/up and down. **Compute-bound** (DMA floor 3.79 ms, 12.8× headroom) — NOT DMA-bound.

---

## The architectural lever

M=8 (speculative decoding, 8 h-vectors per matvec) exceeds the 64 KB tile output limit **only if one tile computes the full N=17408**: 8×17408×4 = 557 KB. Splitting N across tiles fixes it: NT=16 → N_TILE=1088 → held output 8×1088×4 = **34.8 KB** (fits a 64 KB tile). The 8 h-vectors fill the 8 A-rows of `<8,8,4>` FREE (same MMUL count as M=1), so M=8 costs the same per-tile MMUL work as M=1 but delivers 8 matvecs.

NT=16 is the ceiling: the NPU2 has 32 compute tiles but only 8 shims × (2 MM2S + 2 S2MM) = **16 DMA channels** for a 1-fifo-in + 1-fifo-out-per-tile design. NT=32 placement FAILS: *"no ShimNOCTile has sufficient DMA capacity for 0 input/1 output channels near centroid column 5."*

## Measured results (both shapes, stable across reruns)

| Kernel | up full | up noop(DMA) | down full | down noop | ratio |
|--------|---------|--------------|-----------|----------|-------|
| M=8 M_n=16 NT=16 depth=2 | 48.53 ms | 3.79 ms | 48.60 ms | 3.80 ms | 12.8× |
| M=8 M_n=32 NT=16 depth=1 (wide) | 48.64 ms | — | — | — | — |

- 8/8 argmax MATCH, up rel~3.0e-3, down rel~2.5e-3. PASS.
- Beats host M=8 OpenMP (85.6 ms = 8 × 10.7 ms/h) by **1.76×**.
- `noop` column = **trivial noop** (1 scalar read, 1 write, no load_v loop) — the TRUE DMA floor.

## The pivotal correction: it is COMPUTE-bound, not DMA-bound

The earlier "noop ≈ full (48.7 ≈ 48.5) → DMA-bound" conclusion was **wrong**. That noop did 128 `load_v`/packet (8 h-vectors × 16 groups × `load_v<8>`) — which is itself ~48 ms of real load work. A **trivial noop** (1 read of `pkt[0]`, 1 write, no load loop) measures the true dispatch+DMA+output floor:

```
trivial noop (up)   = 3.79 ms  (36.11 GB/s over 136.8 MB)
trivial noop (down) = 3.80 ms  (36.03 GB/s)
full M=8 (up)       = 48.53 ms (2.82 GB/s)
full M=8 (down)     = 48.60 ms
```

So the real compute = 48.53 − 3.79 ≈ **44.7 ms**; DMA is only 3.79 ms. The kernel is **compute-bound with 12.8× DMA headroom**. The 16-tile M=1 noop (10784 B packet, 4.4 KB output) = 2.21 ms (41 GB/s) — the 16-tile dispatch cost itself is tiny; the M=8 cost is the per-tile compute, not dispatch.

(Per-tile compute ≈ 21760 MMUL/tile + 2×-dequant. At ~2.23 µs/MMUL-including-dequant the dequant staging is the bulk.)

## Why dequant-once (2-live MMUL) was NOT used

Step 1 built `fst_qwopus_ffn_n4_live2_kernel.cc` — dequant-once-across-nt4 with 2 live `aie::mmul` accumulators via `alignas(128)` (the linchpin fix: `aie::mmul` stores `accum_type data` at offset 0, so `alignas(128)` on the mmul variable aligns the spilled accumulator).

- **alignas fix WORKS** — 2 live accumulators coexist, no crash, CORRECT (argmax MATCH, rel 3.010e-3, PASS).
- **But 7 ms SLOWER**: 91.93 ms (2-live dequant-once) vs 84.99 ms (n4v 1-live 2×-dequant), same 8-tile M=1 geometry.

The dequant-amortization premise is FALSE on AIE2P: the 2-live-MMUL register/stack overhead exceeds the saved dequant pass. **M=8 uses 2×-dequant (n4v structure) — safer AND faster.** The M=8 win comes from NT=16 parallelism (2× over the 8-tile M=1 n4v 85 ms), not dequant amortization.

## Why the wide variant (M_n=32) didn't help

`fst_qwopus_ffn_m8w_kernel.cc` (M_n=32, 4 dequant tiles, 340 pkts/tile vs 680, packet 16928 B) = **48.64 ms — identical to M_n=16 (48.53 ms)**. This was expected to halve the DMA floor, but since the kernel is **compute-bound** (not DMA-bound), halving the packet count changes nothing — compute is fixed by K×N×MMUL. Confirms the floor is compute, not packet-dispatch.

(M_n=32 packet 16928 B ≥ 16 KB is on the M=4-proven DMA-safe side of the packet wall — the wall is higher, between 16928 and 26240 B.)

## What's left (not pursued — gate stays OFF)

The compute-bound 48.5 ms is near the structural ceiling for this geometry:
- MMUL count is fixed by K×N; M=8 already fills all 8 A-rows (free parallelism maxed).
- `<8,8,4>` native mac is the fastest available (bfp16 `<8,8,8>` explodes under the S recurrence; not relevant here but same family).
- NT=16 is the shim-DMA-channel ceiling; NT=32 doesn't place. More tiles need a broadcast/shared-fifo or mem-tile design — a bigger architectural change, not in scope.
- 2×-dequant is the local optimum; dequant-once is slower; a correct vectorized dequant (faster than the n4v scalar-strided copy) is unproven and the staging bug was hard.

The 12.8× DMA headroom is real but unusable without reducing compute, and compute has no obvious remaining lever at NT=16.

## Files

- `kernels/fst_qwopus_ffn_m8_kernel.cc` — M=8 M_n=16 NT=16 depth=2 (the measured win)
- `kernels/fst_qwopus_ffn_m8w_kernel.cc` — M_n=32 wide variant (identical 48.6 ms, compute-bound)
- `kernels/fst_qwopus_ffn_m8_trivnoop_kernel.cc` — trivial noop (TRUE DMA floor = 3.79 ms)
- `kernels/fst_qwopus_ffn_n4_live2_kernel.cc` — Step 1: 2-live alignas probe (works, 7 ms slower)
- `kernels/gen_qwopus_ffn_m8.py`, `gen_qwopus_ffn_m8w.py`, `gen_qwopus_ffn_nt16m1.py`
- `tools/qwopus_ffn_m8_probe.cpp`, `tools/qwopus_ffn_m8w_probe.cpp`

## Recommendation

M=8 NT=16 is a **verified 1.76× win over host M=8** (48.5 ms vs 85.6 ms), 8/8 correct. It is compute-bound with large DMA headroom. The probe is NOT wired (gate OFF, shipped path untouched) per standing constraint. Wiring would require engine-side M=8 h-vector packing + the 16-tile dispatch integration; the per-h 6.07 ms vs host 10.7 ms (1.76×) is the realizable speedup if the orchestrator authorizes wiring.