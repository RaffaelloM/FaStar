# A3 Vector-Nibble Dequant Probe — PASS (bit-exact)

**Date:** 2026-07-13
**Verdict:** The hardest IRON element of the float-fifo NPU FFN path — vectorized
MXFP4 dequant from a FLOAT-typed ObjectFifo — **works bit-exactly**, using the
proven fused-FFN pattern (`aie::load_v`-as-uint8 + `vector::operator[]` register
extract + scalar `FP4[]`), NOT a hardware LUT. The probe ALSO falsifies the
postmortem's "scalar reads of a float fifo return a ramp" premise: scalar
MEMORY reads deliver the packed bytes bit-exactly too.

## The probe

`kernels/fst_ffn_nibprobe_kernel.cc` — two kernels on the SAME float fifo
(32-float = 128 B packet; first 16 B = 16 MXFP4 weight bytes; rest = 0):

- **`nib_vec`** (path under test): `load_v<32>(reinterpret_cast<uint8_t*>(pkt))`
  byte-perfect vector load, then `wb32[i]` (`vector::operator[]`, register
  extract — NOT memory-scalar) + scalar `FP4[b & 0x0F]` / `FP4[(b>>4)&0x0F]`.
  Output = 32 floats = [16 low-nibble FP4, 16 high-nibble FP4].
- **`nib_sca`** (control): scalar MEMORY reads `p[i]` of the float packet +
  scalar `FP4[]`.

`kernels/gen_ffn_nibprobe.py` — 1-tile, float32 in/out ObjectFifos (depth-2),
16 packets. Builds vec and sca in SEPARATE processes (`@iron.jit` caches the
resolved program in-process and `fn_name` is not part of the cache key → one
process yields two identical xclbins).

`tools/qwopus_ffn_nibprobe.cpp` — packs 16 deterministic weight bytes
(`(p*13 + i*37) & 0xFF`, full 0..255 range) into the first 16 B of each 128-B
float packet; host ref `FP4_LUT[nib]`; compares both NPU outputs to ref
(max|Δ|, argmax match).

## Measured result — PASS

```
[time] nib_vec = 0.32 ms
[probe] nib_vec max|Δ|=0.0000e+00 argmax ref=11 npu=11 MATCH  firstbad=-1
[time] nib_sca = 0.23 ms
[probe] nib_sca max|Δ|=0.0000e+00 argmax ref=11 npu=11 MATCH  firstbad=-1
vec_maxabs=0.0000e+00 vec_argmatch=1 sca_maxabs=0.0000e+00 sca_argmatch=1 PASS
```

**Both paths are bit-exact (max|Δ| = 0, argmax identical, firstbad = -1) across
all 16 packets.** No hardware LUT, no `bit_and`, no `aie::unpack` — just the
proven `load_v`-as-uint8 + `operator[]` + scalar LUT pattern, adapted to a float
fifo.

## Two findings

### Finding 1 — the float-fifo NPU FFN dequant is DE-RISKED

The prior postmortem (`docs/QWOPUS_NPU_FFN_RESTREAM_BLOCKED_POSTMORTEM.md`)
claimed the float-fifo path required a hardware LUT because "scalar reads of a
float fifo return a ramp → MUST vectorize the nibble extract: `load_v` the 16
weight bytes as a `uint8x16`, `aie::unpack` to extract nibbles, vector LUT for
FP4." That requirement is **false**. The proven fused-FFN pattern
(`fst_fused_dequant_gemm_kernel.cc`) — `load_v`-as-uint8 (byte-perfect) +
`vector::operator[]` (register extract) + scalar `FP4[]` — works bit-exactly on
a float fifo. No `aie::unpack`, no `aie::lut<4,float>`, no nibble-split via
`bit_and`.

The Phase-2 failure was a single, specific bug: it `load_v`'d as **FLOAT**
(`load_v<8>(reinterpret_cast<float*>(p))`) instead of as **uint8**
(`load_v<32>(reinterpret_cast<uint8_t*>(p))`). The float reinterpret reinterprets
the packed bytes as floats (garbage values like 4e21, 1e24). The uint8
reinterpret reads the raw bytes byte-perfectly. That is the entire difference.

This kills three dead-ends I chased:
- **Hardware `aie::lut<4,float,float>`** — no AIE2P reference exists
  (`rgba2hue` uses the LUT only `#ifdef __AIE2__` and falls back to scalar on
  AIE2P due to acc32-SRS differences); the bank-duplication layout I derived by
  analogy to AIE2's uint16 LUT was half-wrong (lanes {0-3,8-11} correct,
  {4-7,12-15} wrong = 2 of 4 parallel ports). The float layout uses
  `load_lut_2x_int16` + `T32_16x2_lo`, which differs from uint16's
  `load_lut_2x_int8` + `T16_16x4_lo` and is undocumented. Now MOOT.
- **`aie::bit_and` on uint8** — un-legalizable on AIE2P (`G_AND` on
  `<4 x s32>` ICE's clang exit 70). The nibble split must NOT use AND/shift.
  Now MOOT (operator[] + scalar `& 0x0F` on a register uint8 is fine — that's
  scalar on a register value, not vector `bit_and`).
- **`aie::unpack`** — widens element width (uint8→int16), does NOT split
  nibbles. Not needed.

### Finding 2 — scalar MEMORY reads work on a float fifo (postmortem premise FALSIFIED)

`nib_sca` does `uint8_t b = p[i]` — a scalar MEMORY read of a uint8 pointer
into the DMA'd float fifo element. It returns the EXACT packed bytes
(max|Δ| = 0). The prior postmortem's premise — "scalar reads of a float fifo
return a ramp" — is **false**. Scalar reads of a float fifo deliver the packed
bytes correctly, same as a uint8 fifo.

Implication: the "always vectorial" constraint (no slow scalar NPU math) is
still respected by `nib_vec` (the DMA load is vectorized; the dequant operates
on register values via `operator[]`, which is fast — not a memory stall). A
purely-scalar dequant (scalar memory reads + scalar LUT) would also be
*correct* but would risk the slow-scalar prohibition on the memory-read axis;
`nib_vec` (vectorized load + register extract) is the constraint-safe choice
and is what shipped.

## What this unblocks / does NOT unblock

**Unblocked:** the dequant stage of the float-fifo NPU FFN path. The IRON
element the orchestrator asked to de-risk first (vectorized dequant from a
float fifo) is proven bit-exact. The remaining FFN engineering (the MAC, scale
handling, h-access) can use this proven dequant block.

**NOT unblocked (still standing):**
- The **16 KB on-tile-array wall** (`docs/QWOPUS_NPU_FFN_RESTREAM_BLOCKED_POSTMORTEM.md`
  axis 2). h = 20 KB > 16 KB. The dequant working does NOT address the h-access
  wall. h still cannot live in a single ≥16 KB packet buffer; the chunked-h
  redesign (split h into <8 KB chunks, partial-sum RMW into a held output <16 KB)
  is still required and still has unproven IRON elements (chunked partial-sum
  RMW).
- The **~0.3 tok/s ceiling** of the chunked-h path (the report's own estimate,
  vs the 10× goal). The dequant being free does not raise that ceiling — the
  ceiling is set by the h-access wall + the marginal arithmetic intensity of
  M=1 decode.

## Decision

The A3 deliverable is complete: the hardest IRON element is de-risked
bit-exactly, and a false premise (scalar-ramp) is corrected. The float-fifo NPU
FFN path is less risky than the prior postmortem stated (no hardware LUT, no
`bit_and`, no `unpack` needed) — the remaining risk is the chunked-h RMW, not
the dequant.

**Orchestrator's call** remains: whether the chunked-h redesign (remaining
risk, ~0.3 tok/s ceiling) is worth pursuing vs the standing forks
(constraint-relaxation D, or accept ~0.2 tok/s as the in-constraint ceiling E).
The dequant de-risk lowers the cost of attempting the chunked-h path but does
not change its ceiling.

## Artifacts (repo-clean)
- `kernels/fst_ffn_nibprobe_kernel.cc` — `nib_vec` + `nib_sca`.
- `kernels/gen_ffn_nibprobe.py` — 1-tile float-fifo generator (vec/sca/both).
- `kernels/fst_ffn_nibprobe_vec.xclbin` + `_insts.bin` (vec build).
- `kernels/fst_ffn_nibprobe_sca.xclbin` + `_insts.bin` (sca build).
- `tools/qwopus_ffn_nibprobe.cpp` — 2-xclbin A/B probe.