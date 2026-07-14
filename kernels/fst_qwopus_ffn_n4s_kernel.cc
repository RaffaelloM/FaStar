// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_qwopus_ffn_n4_kernel.cc — NPU FFN matvec for Qwopus3.6 (Phase 4b,
// Re-stream-h + NATIVE bf16 <8,8,4> MMUL + vectorized bitwise MXFP4 dequant).
//
// WHY THIS EXISTS: the Phase 4a kernel (fst_qwopus_ffn_kernel.cc) used the
// EMULATED bfp16 aie::mmul<8,8,8> and DIVERGED on both gate/up and down
// (rel 1.18..4.1).  Root cause isolated 2026-07-14 (see memory
// qwopus-ffn-bfp16-on-b-rootcause): the emulated <8,8,8> UNCONDITIONALLY
// converts both operands to BFP16 (block-fp16, shared exponent per 8-elem
// block) via to_v64bfp16ebs8 + mac_8x8_8x8T_conf — and BFP16's shared exponent
// crushes the wide dynamic range of the e8m0-scaled dequant B operand (scales
// span 2^-7..2^13 ≈ 20 bits/block) → garbage.  Same bfp16-explosion the GDN
// investigation found.  Proven: h=1.0 (uniform A ⇒ bfp16-on-A exact) + real
// scaled dq STILL diverges (rel 3.3) ⇒ bfp16-on-B is the bug, NOT the transpose,
// NOT the layout, NOT the accumulation (h=1+dq=1 ⇒ output=K exactly ⇒ mmul/
// accum/output structurally correct; the p4a "dq=1.0 proves mmul broken" claim
// was a flawed const test).
//
// FIX: NATIVE <8,8,4> (mac_8x8_8x4_bf16) — bf16×bf16→fp32-accum, NO bfp16
// conversion, exact products (upcast bf16→fp32 is exact).  The only EXACT mmul
// path on AIE2P; proven 99.5% argmax by the GDN mmuln4 kernel
// (kernels/fst_gdn_8kslab_mmuln4_kernel.cc).  Cost: N=4 (max native N; no native
// N=8) ⇒ 4 N-tiles of 4 instead of 2 of 8 (2× the mac instructions) + scalar
// 4-copy B packing (aie_api has no 4-elem bf16 vector ⇒ no load_v<4>/extract<4>).
// For the M=1 correctness BASELINE the dequant is re-run per N-tile (2× dequant)
// so only ONE MMUL is live at a time (avoids the 2-live-MMUL stack crash noted
// in the Phase 4a kernel); speed is irrelevant for the baseline — the dequant
// amortization + speed target is the M=4 kernel (fst_qwopus_ffn_m4_kernel.cc).
//
// Everything else is byte-identical to Phase 4a: the same packet geometry, the
// same proven bit-exact vectorized bitwise dequant (dequant_group unchanged),
// the same re-stream-h BF16 h, the same held-output RMW (kc==0 store / +=).
// Only the MMUL tile shape (8→4) and the B-operand load (transpose→scalar
// 4-copy gather) change.
//
// M=1 MXFP4 matvec: out[n] = sum_k W[n,k]·h[k], W raw MXFP4 (17-byte blocks: 1
// e8m0 scale + 16 nibble bytes for 32 FP4 elements, low-nibble-first), h BF16.
// One K-agnostic kernel serves every q35 projection; host controls packet
// count (n_chunks × k_chunks) and per-packet row_base.
//
// MMUL mapping (native <8,8,4>, A[8,8]@B[8,4]->C[8,4] fp32-accum, exact):
//   A[m,k] = h[K_idx+k] broadcast 8×  (M=8 = tile minimum; 7 M-rows redundant
//                                     for the M=1 matvec, row 0 = output)
//   B[k,j] = dequant(output_{nbase+j}, K_idx+k)  k-outer n-inner, 4 cols
//   C[0,j] = sum_k h[K_idx+k]·dq(W[output_{nbase+j}, K_idx+k]) = partial[j]
// 16 outputs/packet = 2 dequant tiles (nt 0,1; 8 outputs each) × 2 N-tiles-of-4
// (nt4 0,1; outputs 0..3 / 4..7 of each dequant tile).  K-reduction across
// G=32 groups × 4 K-subs × K=8 = 1024 k per packet (full K across k_chunks).
// Held output: kc==0 ⇒ o[..]=partial, else o[..]+=partial (RMW across k_chunks).

// NO bfp16 emulation: native <8,8,4> uses mac_8x8_8x4_bf16 (exact bf16, NO
// to_v64bfp16ebs8).  Do NOT define AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16.
#include <aie_api/aie.hpp>
#include <stdint.h>
// The aie_api operator overloads (& | << >> + - == != < > etc.) live in
// aie::operators and need a using-directive to be visible (per aie_api's own
// operators.hpp).  The vectorized bitwise dequant below uses them.
using namespace aie::operators;

constexpr int M_n = 16;     // output rows per packet (the N dimension of the mmul)
constexpr int G   = 32;     // MXFP4 groups per k_chunk (h_chunk = G*32 bf16 = 2 KB)
constexpr int HDR = 32;     // [row_base(4) | kc(4) | 24 pad] (h 32 B-aligned)
constexpr int BLK_BYTES = 17;                    // 1 e8m0 scale + 16 nibble bytes (32 FP4)
constexpr int H_BYTES   = G * 32 * 2;            // 2048 (bf16)
// Weight region split so the 16 nibble bytes of every block are 16-B-aligned
// (a 17-B block can't all be 16-aligned => load_v<16> of the nibble bytes needs
// them split out from the scales).  Same 8704-B total; PKT stays 10784 B <16KB.
constexpr int SC_BYTES  = M_n * G;              // 512 e8m0 scales (1 per row/group)
constexpr int NB_BYTES  = M_n * G * 16;         // 8192 nibble bytes (16-aligned blocks)
constexpr int W_BYTES   = SC_BYTES + NB_BYTES;   // 8704 (split, same total)
constexpr int PKT_BYTES = HDR + H_BYTES + W_BYTES;  // 10784 (< 16 KB)

// ── MMUL tiling (NATIVE bf16, exact) ─────────────────────────────────────────
// aie::mmul<8,8,4,bfloat16,bfloat16,accfloat> = mac_8x8_8x4_bf16, NO bfp16.
// A=64 (M=8,K=8), B=32 (K=8,N=4), C=32 (M=8,N=4).  16 outputs/packet =
// 2 dequant tiles (nt 0,1) × 2 N-tiles-of-4 (nt4 0,1).  1 live MMUL (sequential
// nt4) to avoid the 2-live-MMUL stack crash; dequant re-run per nt4 (2× dequant,
// acceptable for the M=1 correctness baseline; the M=4 kernel amortizes it).
using MMUL = aie::mmul<8, 8, 4, bfloat16, bfloat16, accfloat>;
constexpr unsigned K_TILE   = 32;             // one MXFP4 block = 32 K elems
constexpr unsigned colA     = K_TILE / 8;     // 4 K-sub-tiles (K=8 per mmul)
constexpr unsigned N_TILES4 = 4;              // 4 N-tiles of 4 (16 outputs)

// ── Vectorized bitwise MXFP4→BF16 dequant (Phase 4a, UNCHANGED, bit-exact) ───
// Replaces the scalar FP4 LUT + scalar e8m0 scale-mul with pure vector bitwise
// ops on aie::vector<uint16_t,32>.  The e8m0 scale is FOLDED into the exponent
// (no scalar mul): sb = e8m0<<7 is exponent-only, scale×fp4_mag = exp-field add.
//
// BIT-EXACT to the prior scalar LUT path: every fp4 magnitude {0.5,1,1.5,2,3,4,
// 6} is exactly representable in BF16, and × a power-of-two scale stays exact,
// so both paths produce identical BF16 dq bits.
//
// FP4 (E2M1) nibble v=[S E1 E0 M], magnitude m, scaled by e8m0=2^(sc-127):
//   zero      (E1E0==0, M==0) : 0
//   subnormal (E1E0==0, M==1) : 0.5·2^(sc-127) = 2^(sc-128) -> exp=sc-1, man=0
//   normal    (E1E0>0)        : 2^(E1E0-1)·(1+0.5M)·2^(sc-127)
//                              -> exp=sc+E1E0-1, man=M<<6
// The subnormal case MUST clear the mantissa bit: naive "man=M<<6 always" gives
// 2^(126-127)·1.5 = 0.75, not 0.5, for nibble 1.  Here man=(M<<6) only where
// E1E0>0 (select).  sc clamped to 252 so exp_field=sc-1+E1E0<=254.  Dead blocks
// (sc==0/255) and zero nibbles zeroed via the nonzero mask.
//
// AIE2P legalizer constraint: vector RIGHT shifts (aie::logical_downshift /
// operator>>) on uint8/uint16 do NOT compile.  So this dequant uses ONLY left
// shifts (upshift, which legalize) + AND + OR + add + select + compare.  Each
// nibble's sign/E1E0/M fields are extracted at their NATIVE bit positions and
// upshifted straight to the bf16 bit positions — no normalization right-shift.
// SCALAR bit-exact MXFP4->BF16 dequant (drop-in replacement for the vectorized
// Phase 4a dequant_group).  The vectorized version (interleave_zip/concat/extract<8>/
// store_unaligned_v) produces a WRONG dq layout: the 4 ks sub-blocks (k=0..7, 8..15,
// 16..23, 24..31) come out IDENTICAL (= k=0..7) => k=8..31 lost, k=0..7 duplicated 4x
// per group => the matvec sums 4x(partial k=0..7) vs the host's full k=0..31 => the
// rel~1.2..3.3 magnitude/sign divergence on BOTH <8,8,8> and <8,8,4> (the dq=1.0 const
// test could NOT see it — uniform values read identically from any slot; only a dq
// dump revealed it).  The per-nibble bitwise MATH is bit-exact (host-verified); the
// bug was purely in the aie::vector staging ops.  This scalar version uses NONE of
// them — pure scalar bitwise => bit-exact by construction.  Slower (the 2.7-dequants/
// cycle path) but CORRECT; correctness first, re-vectorize later once argmax MATCHes.
static inline void dequant_group(const uint8_t *scales, const uint8_t *nibbles,
                                 int nt, int g, bfloat16 *dq) {
    for (int n = 0; n < 8; ++n) {
        const int rg = (nt * 8 + n) * G + g;
        const uint8_t sc = scales[rg];
        if (sc == 0 || sc == 255) {                           // dead block -> 32 zeros
            for (int ks = 0; ks < 4; ++ks)
                for (int kr = 0; kr < 8; ++kr)
                    *reinterpret_cast<uint16_t*>(dq + ks*64 + n*8 + kr) = 0;
            continue;
        }
        const uint8_t sc_eff = (sc > 252) ? (uint8_t)252 : sc;
        const uint16_t scm1_sh7 = (uint16_t)((sc_eff - 1) << 7);
        const uint8_t *nb = nibbles + (size_t)((nt * 8 + n) * G + g) * 16;
        for (int ks = 0; ks < 4; ++ks) {
            for (int kr = 0; kr < 8; ++kr) {
                const int k = ks * 8 + kr;
                const uint8_t byte = nb[k >> 1];
                const uint8_t nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                const uint16_t sign = (nib & 0x08) ? 0x8000 : 0;
                const int E1E0 = (nib >> 1) & 0x3;             // bits 1-2
                const int M    = nib & 0x1;
                uint16_t bits;
                if (E1E0 == 0) {
                    // subnormal/zero: 0.5*M * 2^(sc-127) = M ? 2^(sc-128) : 0
                    bits = M ? (uint16_t)(sign | scm1_sh7) : (uint16_t)0;  // exp=sc-1, man=0
                } else {
                    // normal: 2^(E1E0-1)*(1+0.5M)*2^(sc-127) -> exp=sc-1+E1E0, man=M<<6
                    bits = sign | (uint16_t)((E1E0 << 7) + scm1_sh7) | (uint16_t)(M << 6);
                }
                *reinterpret_cast<uint16_t*>(dq + ks*64 + n*8 + kr) = bits;
            }
        }
    }
}

// Re-stream-h NATIVE-<8,8,4> matvec.  Dequant is the Phase 4a vectorized bitwise
// path (unchanged, bit-exact); the MMUL is now native exact bf16 (no bfp16).
//
// Layout (vectorized dequant): per (dequant-tile nt, group g) the 8 outputs ×
// 32 k are dequanted ONCE into dq[ks*64 + n*8 + k_rel] = dequant(output_n,
// ks*8+k_rel) (n=0..7 of the nt tile).  Each native <8,8,4> N-tile-of-4 gathers
// 4 of those 8 outputs into B_buf[k*4 + j] = dq[ks*64 + (nbase+j)*8 + k]
// (scalar 4-copy — aie_api has no 4-elem bf16 vector) for j=0..3, k=0..7, the
// exact B[k,j] = dequant(output_{nbase+j}, K_idx+k) the GDN mmuln4 kernel uses.
//   A[m,k] = h[K_idx+k] for all m (broadcast M=8 — 7 M-rows wasted, the r=8 tile
//     minimum for an M=1 matvec).
//   C.mac(A,B) => C[0,j] = sum_k h[K_idx+k]*dq(W[output_{nbase+j},K_idx+k]) = partial[j].
// 2 dequant tiles (nt 0,1) × 2 N-tiles-of-4 (nt4 0,1) = 16 outputs/packet.
// The 2 N-tiles-of-4 are processed SEQUENTIALLY (1 live MMUL — the 2-live-MMUL
// stack crash: a vector slot lands at a non-64-B-aligned offset); dequant is
// re-run per nt4 (2× dequant, acceptable for the M=1 correctness baseline; the
// M=4 kernel amortizes the dequant once across 4 h-vectors).
// Held output: kc==0 => o[..]=partial, else o[..]+=partial (RMW across k_chunks).
extern "C" void ffn_matvec_restream(
    const uint8_t *__restrict pkt,
    float          *__restrict o)
{
    const int row_base = *reinterpret_cast<const int *>(pkt);
    const int kc       = *reinterpret_cast<const int *>(pkt + 4);
    const bfloat16 *__restrict h = reinterpret_cast<const bfloat16 *>(pkt + HDR);
    const uint8_t  *__restrict scales  = pkt + HDR + H_BYTES;
    const uint8_t  *__restrict nibbles = scales + SC_BYTES;
    ::aie::set_rounding(aie::rounding_mode::floor);
    event0();

    alignas(128) bfloat16 A_buf[64];    // [m=8, k=8] m-fast (h broadcast)
    alignas(128) bfloat16 B_buf[32];    // [k=8, n=4] k-outer n-inner (scalar 4-copy)
    alignas(128) bfloat16 dq[256];      // [ks*64 + n*8 + kr] dequant of 8 outputs (1 nt tile)

    for (int nt = 0; nt < 2; ++nt) {                 // dequant tile 0 (rows 0..7), 1 (8..15)
        for (int nt4 = 0; nt4 < 2; ++nt4) {          // N-tile-of-4: 0..3 / 4..7 of the nt tile
            const int nbase = nt4 * 4;
            MMUL C;
            for (int g = 0; g < G; ++g) {
                dequant_group(scales, nibbles, nt, g, dq);   // 8 outputs × 32 k -> dq
                for (int ks = 0; ks < 4; ++ks) {          // 4 K-subs of 8 per group
                    #pragma unroll
                    for (int m = 0; m < 8; ++m)
                        #pragma unroll
                        for (int k = 0; k < 8; ++k)
                            A_buf[m * 8 + k] = h[g * 32 + ks * 8 + k];   // broadcast h
                    aie::vector<bfloat16, 64> A_op = aie::load_v<64>(A_buf);
                    // B_buf[k*4 + j] = dq[ks*64 + (nbase+j)*8 + k] = dequant(output_{nbase+j}, k)
                    #pragma unroll
                    for (int k = 0; k < 8; ++k)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            B_buf[k * 4 + j] = dq[ks * 64 + (nbase + j) * 8 + k];
                    aie::vector<bfloat16, 32> B_op = aie::load_v<32>(B_buf);
                    if (g == 0 && ks == 0) C.mul(A_op, B_op);
                    else                   C.mac(A_op, B_op);
                }
            }
            aie::vector<float, 32> cv = C.template to_vector<float>();  // cv[m*4+j]
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v = cv[j];                          // C row 0 = the matvec output
                int idx = row_base + nt * 8 + nbase + j;
                if (kc == 0) o[idx] = v; else o[idx] += v;
            }
        }
    }
    event1();
}

// No-op diagnostic: same uint8 packet footprint + DMA, NO dequant/mmul compute
// (one vector load of h per K-tile to force the read, constant output).  Isolates
// the DMA floor from the compute cost so the probe measures compute-vs-DMA.
extern "C" void ffn_matvec_noop_stream(
    const uint8_t *__restrict pkt,
    float          *__restrict o)
{
    const int row_base = *reinterpret_cast<const int *>(pkt);
    const bfloat16 *h = reinterpret_cast<const bfloat16 *>(pkt + HDR);
    volatile float sink = 0.0f;
    for (int g = 0; g < G; ++g) {
        aie::vector<bfloat16, 8> hv = aie::load_v<8>(h + g * 32);   // force h DMA read
        sink = (float)hv[0] + (float)hv[4];                          // use it (register extract)
    }
    for (int r = 0; r < M_n; r++) o[row_base + r] = 1.0f;          // constant
    (void)sink;
}