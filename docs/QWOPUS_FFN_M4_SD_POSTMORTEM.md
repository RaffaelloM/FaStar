# QWOPUS FFN M=4 Speculative-Decoding — MEASURED DEAD-END (2026-07-14)

The M=4 SD lever (dequant the weight once, matvec 4 h-vectors → NPU M=4 beats host
M=4 = 42.8 ms) was the last un-measured QWopus FFN speedup path. **It is a measured
dead-end**: every M=4 NPU architecture built and measured loses to the 42.8 ms host
M=4 baseline (host per-h = 10.7 ms; OpenMP `mxfp4_matvec_f32`, src/fst_engine.cpp:251).

This was NOT written off on a pretext — six kernel variants were built, run on real
hardware (gate/up [17408,5120], narrow sc 120..140), and measured. The dequant staging
bug that blocked it was first FIXED and proven bit-exact.

## Prerequisite WIN: the vectorized dequant is now fast AND bit-exact

The Phase 4a vectorized dequant diverged because its `aie::vector` staging
(`interleave_zip`/`concat`/`extract<8>(8,16,24)`/`store_unaligned_v`) wrote the 4 ks
sub-blocks IDENTICAL (k=0..7 duplicated 4×, k=8..31 lost) — see
memory [[qwopus-ffn-dequant-vector-staging-bug]]. **FIX** (kernels/fst_qwopus_ffn_n4v_kernel.cc):
replace the 4× `extract<8>` strided stores with a single `aie::store_v(tmp, bits)` of
the 32-elem `bits` vector to a flat stack temp, then a scalar strided copy
`dq[ks*64+n*8+kr] = tmp[ks*8+kr]`. Dump-kernel verified: 4 ks sub-blocks now DISTINCT
and **bit-exact 0/32 vs the kernel-correct scalar reference**. Combined with native
`<8,8,4>` (n4v), **argmax MATCH both shapes** (up rel=3.0e-3, down rel=2.5e-3) — the
correctness blocker is DEAD. **dequant-only latency (dequant-once, 64 calls/pkt, no
MMUL) = 23.6 ms** — squarely in the hoped-for 23-33 ms range. The dequant is no longer
the blocker; the M=4 matvec arithmetic is.

## MEASURED M=4 matrix (gate/up, narrow; host M=4 = 42.8 ms, 10.7 ms/h)

| Kernel | MMUL shape | dequant | output | full ms | correct? | per-h ms |
|--------|-----------|---------|--------|---------|----------|----------|
| M=1 n4v (baseline) | `<8,8,4>` ×512 | 2× (47) | 1× held 8.7KB | 85.0 | PASS | 85.0 |
| **M=4 `<4,8,8>`** | `<4,8,8>` ×256 | 1× (23.6) | 4× held 34KB | 96.65 | **PASS (4/4)** | 24.2 |
| M=4 `<4,8,8>` m0-only | `<4,8,8>` ×256 | 1× | 1× held | 95.31 | (diag) | — |
| **M=4 `<8,8,4>` 2×dq** | `<8,8,4>` ×512 | 2× (47) | 4× held 34KB | 87.67 | **PASS (4/4)** | 21.9 |
| M=4 stream-output | `<4,8,8>` ×256 | 1× | stream | 762.8 | PASS (host-sum) | 190.7 |
| dequant-only | — | 1× (23.6) | — | 23.6 | — | — |

Best correct M=4 = **87.67 ms** (per-h 21.9) vs host 42.8 ms (per-h 10.7) → **NPU loses ~2×**.

## Three structural reasons (each independently fatal)

1. **The dequant pass is a fixed overhead the host avoids by FUSING dequant+MAC.**
   The host's `acc += FP4_TABLE[nib]*scale*hb[i]` fuses the dequant into the MAC — no
   separate pass. The NPU cannot fuse bitwise-MXFP4-dequant into a bf16-MMUL (different
   functional units), so it pays a SEPARATE dequant pass. Even amortized ONCE over 4 h's
   (23.6 ms) + the fast `<8,8,4>` MMULs (38 ms) = **61.6 ms > 42.8 ms host**. The
   amortization lowers per-h dequant (5.9 ms) but the host pays ~0, and the NPU MMUL
   (9.5 ms/h) only roughly ties the host MAC (10.7 ms/h) — there is no margin to absorb
   the dequant pass.

2. **dequant-once-across-nt4 (the amortization) REQUIRES 2 LIVE MMUL accumulators** —
   one C per nt4, both alive across all 32 groups (dequant runs once per (nt,g) and is
   reused by both nt4). 2-live-MMUL crashes the AIE2P stack (a vector slot lands at a
   non-64-B-aligned offset — the known issue). The 1-live-MMUL SAFE path (n4v structure:
   nt4 outside g, dequant per nt4) forces **2× dequant (47 ms)** → no amortization →
   M=4 = M=1 = 85-88 ms (the 4 h's ride free in A but the dequant is re-done). This is
   the 87.67 ms row.

3. **`<4,8,8>` achieves dequant-once (1 live MMUL, 8 N-outputs/MMUL → 1 nt4, no 2-live
   issue) BUT its MMUL is ~4× SLOWER per instruction** (0.28 ms/MMUL vs `<8,8,4>`'s
   0.074 ms/MMUL). Proven by the m0-only diagnostic: full 4-h compute + 1× output =
   95.31 ms ≈ the 4×-output M=4 (96.65 ms) → the 4× output RMW is NOT the bottleneck;
   the `<4,8,8>` MMUL is. So dequant-once (23.6) + slow `<4,8,8>` (71) = 96 ms.

The dequant-once + fast-`<8,8,4>` combo (projected 61.6 ms) is blocked by #2, and even
unblocked it loses (#1). The 4× held output (34 KB) is a secondary issue — it forces
input-fifo depth=1 (34 KB + 2×16 KB > 64 KB tile) but is NOT the latency driver (m0-only
proved 1× output = same 95 ms); stream-output makes it far worse (762 ms, per-packet
acquire/release × 680 = 0.12 GB/s).

## The one ray of hope: M=8 (not built)

`<8,8,4>` A has 8 M-rows; M=8 fills them with 8 h-vectors (no waste). dequant-once
(23.6) + 512 `<8,8,4>` MMULs (38, 8 h's in A) = **61.6 ms for 8 h's = 7.7 ms/h < host
10.7** — WINS by ~28%. But M=8 hits two walls, BOTH untested:
- **2-live-MMUL** (same as #2; dequant-once needs it) — the crash may be fixable with
  alignas/decl-reorder, never tried.
- **8× held output = 69 KB** (8×2176×4) — exceeds the 64 KB tile (worse than M=4's
  34 KB). Needs NT=32 (N_TILE=544 → 8×544×4=17 KB, fits) which needs 32 compute tiles,
  OR stream-output (762 ms fatal overhead).

M=8 is the highest-value untested path. It is NOT measured; both walls must be broken
first. Recommend: (a) attempt the 2-live-MMUL alignas fix on a tiny probe (the linchpin
for ALL amortized paths), (b) if it works, evaluate the 8× output via NT geometry.

## Decision

M=4 SD = measured dead-end. **FST_Q35_FFN_NPU stays OFF; shipped 0.20 tok/s host
path untouched.** Do NOT wire any M=4 kernel (n4/n4v/m4/m4s/m4n4). The n4v kernel (fast
+ bit-exact vectorized dequant + native `<8,8,4>`) remains the correct M=1 baseline
should the NPU FFN ever be revisited. The dequant staging fix (store_v+copy) is the
reusable artifact.

## Artifacts

- kernels/fst_qwopus_ffn_n4v_kernel.cc — FIXED dequant (store_v+copy) + `<8,8,4>`, M=1, correct.
- kernels/fst_qwopus_ffn_n4v_dump_kernel.cc — dq-dump (verified 4 ks blocks distinct, bit-exact).
- kernels/fst_qwopus_ffn_n4v_dqonly_kernel.cc — dequant-only (23.6 ms isolation).
- kernels/fst_qwopus_ffn_m4_kernel.cc — M=4 `<4,8,8>` held (96.65 ms, correct).
- kernels/fst_qwopus_ffn_m4_m0_kernel.cc — M=4 m0-only diagnostic (output not the bottleneck).
- kernels/fst_qwopus_ffn_m4s_kernel.cc — M=4 stream-output (762 ms, correct, overhead-fatal).
- kernels/fst_qwopus_ffn_m4n4_kernel.cc — M=4 `<8,8,4>` 2×-dequant (87.67 ms, correct, no amortization).
- kernels/gen_qwopus_ffn_{m4,m4s,m4n4,m4_m0}.py — env-variant builders (4 h's in, 4×/stream output).
- tools/qwopus_ffn_m4_probe.cpp, tools/qwopus_ffn_m4s_probe.cpp — M=4 probes (4 h in, 4 out).
- tools/qwopus_ffn_probe.cpp — FST_FFN_DUMP + totals discriminator (unchanged M=1 probe).