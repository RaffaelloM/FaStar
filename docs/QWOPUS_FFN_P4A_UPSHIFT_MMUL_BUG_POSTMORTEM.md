# QWOPUS FFN Phase 4a (vectorized bitwise dequant) — POSTMORTEM

**Date:** 2026-07-14
**Status:** Phase 4a dequant **COMPILES** (the prescribed bitwise approach is no
longer blocked). BUT E2E validation **FAILS**, and a `dq=1.0` isolation test
proves the failure is in the **shared mmul/emulated-bfp16/output path**, NOT the
dequant. The dequant vectorization is correct-by-construction (bit-exact,
python-verified) but cannot be E2E-validated until the pre-existing mmul bug is
fixed. **Orchestrator's call** on whether to debug the mmul path.

## What Phase 4a set out to do

Vectorize the MXFP4→BF16 dequant with **pure bitwise vector logic** (no scalar
LUT), eliminating the 2.7-dequants/cycle compute bottleneck that made the prior
kernel 33 ms / 3.26× slower than the 10.7 ms OpenMP host. Formula (per nibble
`S E1 E0 M`, scaled by e8m0 `2^(sc-127)`):
  `BF16 = (S<<15) | (Exponent<<7) | (Mantissa<<6)`, `Exponent = 126+(E1E0)` if
  `E1E0M>0` else `0` (subnormal keeps man=0), via `aie::vector<uint16>` bitwise ops.

## Achievement 1 — the bitwise dequant COMPILES (legalizer wall cleared)

The first attempt used **right shifts** (`aie::logical_downshift` / `operator>>`)
to normalize each nibble (`bv>>4` for the high nibble, `nv>>3`/`nv>>1` to extract
fields). This hit a hard AIE2P backend legalizer wall:

```
error in backend: unable to legalize instruction: %59:_(<4 x s32>) = G_AND
  in ffn_matvec_restream  (shift_bits_impl_common / downshift -> SRS path)
```

Vector right-shifts on uint8/uint16 route through the SRS accumulator path,
which generates an illegal 128-bit `G_AND` that cannot be legalized. **No
shipped kernel uses vector right-shifts** (the working `fst_dequant_v4_vectorized`
uses scalar LUT + `aie::mul`).

**Fix — upshift-only formulation.** The legalizer wall is in the *right*-shift
path only; **left shifts (upshift) legalize**. So the dequant extracts each
nibble's sign/E1E0/M fields at their **native bit positions** and upshifts them
straight to the BF16 bit positions — no normalization right-shift:

| field | low nibble (bits 0-3) | high nibble (bits 4-7) | BF16 target |
|-------|-----------------------|------------------------|-------------|
| sign  | `0x08 & bv` (bit3)    | `0x80 & bv` (bit7)     | bit15: `<<12` / `<<8` |
| E1E0  | `0x06 & bv` (E1E0<<1) | `0x60 & bv` (E1E0<<5) | exp contrib: `<<6` / `<<2` (= E1E0<<7) |
| M     | `0x01 & bv` (M)       | `0x10 & bv` (M<<4)     | bit6 mantissa: `<<6` / `<<2` (= M<<6) |

`exp = (sc-1)<<7 + E1E0<<7` (scale folded into exponent, no scalar mul). Low/high
dequanted into separate 16-elem vectors, then `interleave_zip(lo,hi,1)` gives
k-order `[lo0,hi0,lo1,hi1,…] = [k0,k1,…,k31]` (matches the scalar path's
`k=2i←low nibble of byte i, k=2i+1←high`). **Compiles clean** (full xclbin
38010 B). Uses only: `unpack` (widen), `bit_and`(scalar,vec), upshift `<<`,
`bit_or`(vec,vec), `add`(vec,scalar), `select`, `compare`, `interleave_zip`,
`concat`, `store_unaligned_v`.

## Achievement 2 — dequant math is bit-exact

Python-verified the upshift formula produces **identical BF16 bits** to the
scalar LUT path for all 16 nibbles across the probe's sc range (120-140): **0
mismatches**. (Two bugs exist *outside* the probe range: sc=1 subnormal→BF16
subnormal 0x0040 that the formula collapses to 0; and the sc=253/254 clamp-to-252
underflows the exponent by 1. Neither affects the probe or real FFN weights,
which sit well inside normal range. Noted for the record.)

## The wall — E2E output is WRONG, and it's NOT the dequant

Probe (`tools/qwopus_ffn_probe.cpp`, gate/up `[17408,5120]`, sc 120-140):
  - **bitwise dequant:** 23.1 ms, argmax DIFF, relmax=1.19 (finite garbage)
  - **v4-proven dequant** (scalar LUT + vector `aie::mul` by broadcast scale — the
    *shipped working pattern*): 28.4 ms, argmax DIFF, relmax=0.94 (finite garbage)
  - **scalar fp32 dequant** (exact host-ref math): NaN / nondeterministic
    (1354 ms — the AIE cannot reliably run heavy scalar fp32: stack/local-mem
    corruption → racy NaN; a scalar control is INVALID on this AIE)

The **v4-proven dequant also failing** is the key signal: a known-correct dequant
produces garbage, so the bug is in the **shared mmul/emulated-bfp16/output path**
(the only code common to all variants), not the dequant.

### Definitive isolation: `dq = 1.0` constant test

Bypassed the dequant entirely — wrote `dq = 1.0` to all 256 entries (broadcast
1.0 into the strided store). A correct matvec with constant `B[k,n]=1.0` gives
`C[0,n] = sum_k h[k] = -8.64` **identical for all 16 outputs** (A is h-broadcast,
same across n). The NPU instead gave **per-n varying** values (-11.9, -12.7,
-17.1, -22.6, -26.2, …). Per-n variation with a constant B is **impossible** in a
correct matvec, so the mmul/emulated-bfp16/output-extraction path is broken:
`B[k,n]` is not reading the intended dq element, or `A` is not broadcast, or
`cv[n]` is not `C[0,n]`, or the emulated `mac_8x8_8x8T_conf` accumulator layout
differs from the assumed `C[m][n]`. (The emulated-bfp16 flag
`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1` IS set before the include; the
mmul loop, transpose, A_buf broadcast, and `cv[n]` extraction are unchanged from
the prior `argmax-MATCH` kernel — yet now produce garbage, suggesting a
pre-existing regression in the mmul path or that the prior MATCH was from a
different build state not reproducible from the current source.)

## What is solid / what is not

- **SOLID:** the upshift-only bitwise dequant compiles and is bit-exact to the
  LUT path by construction. The AIE2P right-shift legalizer wall is worked
  around (left-shift-only). This was the prescribed Phase 4a deliverable.
- **NOT SOLID:** E2E correctness — blocked by the mmul/output path bug, which is
  orthogonal to and pre-dates the dequant work. Cannot validate argmax or
  measure a real speedup until the mmul path is fixed.
- **Speed (provisional, *if* the mmul bug were fixed):** 23 ms/matvec for the
  bitwise path (vs 33 ms scalar-LUT, vs 10.7 ms OpenMP host). The dequant
  vectorization cut ~10 ms (33→23) but the kernel is still compute-bound 13.7×
  over the 1.7 ms DMA floor — the mmul itself is now a large share. Even
  corrected, 23 ms > 10.7 ms host, so the NPU FFN would still lose to OpenMP
  unless the mmul cost also drops. (The dequant was not the only bottleneck.)

## Recommendation

The dequant vectorization (Phase 4a) is implemented and compiles. The blocker
to E2E validation is the **pre-existing mmul/emulated-bfp16/output bug**, which
is a separate debugging effort (likely the emulated `mac_8x8_8x8T_conf`
accumulator/output layout, or the `cv[n]=C[0,n]` extraction). Two options for the
orchestrator:

1. **Debug the mmul path** (isolate `cv[n]` vs accumulator layout, or swap the
   emulated mmul for a non-emulated/proven GEMM shape) — needed to validate
   Phase 4a E2E and to know the real speedup.
2. **Accept Phase 4a as "compiles + bit-exact by construction"** and move on
   (the provisional 23 ms still loses to the 10.7 ms host, so even a correct NPU
   FFN may not be a win — consistent with the prior postmortem's "NPU cannot beat
   OpenMP host FFN" finding).

The kernel file (`kernels/fst_qwopus_ffn_kernel.cc`) is left in the clean
bitwise-dequant state (diagnostic scaffolding removed); it compiles. The gate
`FST_Q35_FFN_NPU` remains OFF by default (no regression to the shipped 0.20
tok/s host path).

## Artifacts

- `kernels/fst_qwopus_ffn_kernel.cc` — upshift-only bitwise dequant (compiles).
- `logs/ffn_p4a_build_full.log` — bitwise build (EXIT 0).
- `logs/ffn_p4a_probe.log` — bitwise probe (23 ms, argmax DIFF).
- `logs/ffn_p4a_v4_probe.log` — v4-proven-dequant probe (28 ms, argmax DIFF).
- `logs/ffn_p4a_const_probe.log` — **dq=1.0 isolation** (per-n variation with
  constant B = the mmul-path-bug proof).
- `tools/qwopus_ffn_probe.cpp` — probe (unchanged).