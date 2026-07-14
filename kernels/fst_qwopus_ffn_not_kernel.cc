// @variant fst_qwopus_ffn_not_kernel.cc
// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_qwopus_ffn_kernel.cc — NPU FFN matvec for Qwopus3.6 (Fork D / Phase 4a,
// Re-stream-h + emulated BF16 MMUL + VECTORIZED bitwise MXFP4 dequant,
// lossless-argmax relaxation signed off).  Phase 4a swaps the scalar FP4 LUT +
// scalar e8m0 scale-mul (the 2.7-dequants/cycle compute bottleneck) for pure
// vector bitwise uint16 ops — the dequant is now bit-exact AND vectorized.
//
// M=1 MXFP4 matvec: out[n] = sum_k W[n,k]·h[k], W raw MXFP4 (17-byte blocks: 1
// e8m0 scale + 16 nibble bytes for 32 FP4 elements, low-nibble-first), h BF16.
// One K-agnostic kernel serves every q35 projection (FFN gate/up [17408,5120],
// down [5120,17408]; SSM/attn projections); the host controls packet count
// (n_chunks × k_chunks) and per-packet row_base.
//
// DESIGN (Re-stream-h + BF16 MMUL — do NOT hold h on-tile; do NOT chunk-h RMW):
//   * ONE uint8 stream per tile.  Packet = [32 B header | h_chunk(2 KB BF16) |
//     M_n*G*17 weight bytes] = 10784 B (< 16 KB => 51 GB/s DMA side of the wall).
//       - h_chunk = G*32 bf16 (G=32 => 1024 bf16 = 2048 B = one k_chunk of K).
//         h streams as BF16 (the model's h IS bf16; halved vs the fp32 path).
//       - weight run = M_n rows × G groups × 17 B (M_n=16, G=32 => 8704 B).
//       - header = [row_base(int32) | kc(int32) | 24 B pad]; kc lets the kernel
//         CLEAR the held-output slot on the first k_chunk (kc==0) — the held
//         output is a tile-memory fifo buffer the host does NOT pre-zero.
//   * h is RE-STREAMED every packet (never held on-tile; a 20 KB held array is
//     1470× slow).  The kernel `load_v`s h from the stream at offset HDR.
//   * Accumulation into the held S2MM output `o` (N_TILE fp32 = 8.7 KB < 16 KB,
//     fast RMW): on kc==0 `o[row_base+r] = partial`, else `+= partial`.
//   * 8-tile N-split of N (gate/up: N=17408 => N_TILE=2176; down: N=5120 =>
//     N_TILE=640).  The kernel processes M_n=16 output rows per packet.
//
// MMUL (native AIE2P, reuse fst_fused_dequant_gemm_16x64x64):
//   aie::mmul<4,8,8, bfloat16, bfloat16, accauto>  (A[4×8] @ B[8×8] -> C[4×8],
//   fp32 accumulator).  For the M=1 matvec we map:
//     A = h broadcast 4×  (M=4 = minimum for the r=4 tile; 3 rows redundant)
//     B = W dequanted to a col-major B_buf (K=32 fast, N=16 slow; the 16 output
//         rows of this packet are the N dimension)
//     C row 0 = y[0..15]  (all 4 M-rows of C are identical since A rows = h)
//   so each packet does ONE M-tile (M=4) × TWO N-tiles (N=16) and K-reduces
//   across the G=32 K-tiles (one MXFP4 block each) in-kernel, accumulating two
//   MMUL objects (C0 for N-tile 0, C1 for N-tile 1).  4× M-waste (3 redundant
//   rows) is the minimum for the r=4 tile on a matvec.
//
// Packet (uint8, host-packed): [32 B header | h_chunk(2048 B) | M_n*G*17 B].

// Emulate bf16 mmul with bfp16 (block-fp16, 8-element blocks): selects the
// mmul_bf16_bf16<8,8,8> specialization that uses mac_8x8_8x8T_conf + transpose
// (NOT mac_4x8_8x8_bf16 => tests whether the -4 register underflow is avoided).
#define AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 1
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
//   scales : M_n*G e8m0 bytes            (scales[r*G+g] = block (r,g) scale)
//   nibbles: M_n*G*16 nibble bytes        (nibbles[(r*G+g)*16+i] = block's 16 bytes,
//                                        16-aligned => aie::load_v<16> works)
constexpr int SC_BYTES  = M_n * G;              // 512 e8m0 scales (1 per row/group)
constexpr int NB_BYTES  = M_n * G * 16;         // 8192 nibble bytes (16-aligned blocks)
constexpr int W_BYTES   = SC_BYTES + NB_BYTES;  // 8704 (split, same total)
constexpr int PKT_BYTES = HDR + H_BYTES + W_BYTES;  // 10784 (< 16 KB)

// ── MMUL tiling ─────────────────────────────────────────────────────────────
// DIAG: aie::mmul<8,8,8> EMULATED bfp16 (M=8,K=8,N=8).  With the emulate flag
// above, this selects mac_8x8_8x8T_conf + transpose(b,8,8) + to_v64bfp16ebs8 (NOT
// mac_4x8_8x8_bf16).  A=64 (M=8,K=8), B=64 (K=8,N=8), C=64 (M=8,N=8).  K=8 =>
// contiguous-k dequant (vectorizable, the fast <4,8,8> pattern).  Test: does
// this path have the -4 register underflow?  (bfp16 ~7-bit mantissa = precision
// risk for argmax, tested separately.)
using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
constexpr unsigned K_TILE   = 32;             // one MXFP4 block = 32 K elems
constexpr unsigned N_TILE   = M_n;            // 16 output rows
constexpr unsigned colA     = K_TILE / 8;    // 4 K-sub-tiles (K=8 per mmul)
constexpr unsigned colB     = N_TILE / 8;    // 2 N-tiles
constexpr unsigned N_COLS   = N_TILE;        // 16

// ── Vectorized bitwise MXFP4→BF16 dequant (Phase 4a) ───────────────────────
// Replaces the scalar FP4 LUT + scalar e8m0 scale-mul (the 2.7-dequants/cycle
// compute bottleneck that made the kernel 33 ms / 3.26× slower than the OpenMP
// host) with pure vector bitwise ops on aie::vector<uint16_t, 32> (32 lanes/
// cycle).  The e8m0 scale is FOLDED into the exponent (no scalar mul): since
// sb = e8m0<<7 is exponent-only, scale×fp4_mag = exponent-field add.
//
// BIT-EXACT to the prior scalar LUT path: every fp4 magnitude {0.5,1,1.5,2,3,4,
// 6} is exactly representable in BF16, and × a power-of-two scale stays exact,
// so both paths produce identical BF16 dq bits => the MMUL input is unchanged =>
// argmax preserved by construction (only the dequant SPEED changes).
//
// FP4 (E2M1) nibble v=[S E1 E0 M], magnitude m, scaled by e8m0=2^(sc-127):
//   zero      (E1E0==0, M==0) : 0
//   subnormal (E1E0==0, M==1) : 0.5·2^(sc-127) = 2^(sc-128) -> exp=sc-1, man=0
//   normal    (E1E0>0)        : 2^(E1E0-1)·(1+0.5M)·2^(sc-127)
//                              -> exp=sc+E1E0-1, man=M<<6
// The subnormal case MUST clear the mantissa bit: the naive "man=M<<6 always"
// gives 2^(126-127)·1.5 = 0.75, not 0.5, for nibble 1.  Here man=(M<<6) only where
// E1E0>0 (select).  sc is clamped to 252 (scalar) so exp_field=sc-1+E1E0<=254 and
// the assembled exp bits stay below bit 15 (no sign clobber).  Dead blocks
// (sc==0/255) and zero nibbles are zeroed via the nonzero mask.
//
// AIE2P legalizer constraint: vector RIGHT shifts (aie::logical_downshift /
// operator>>) on uint8/uint16 hit an illegal <4 x s32> G_AND in the SRS path and
// do NOT compile.  So this dequant uses ONLY left shifts (upshift, which legalize)
// + AND + OR + add + select + compare.  Each nibble's sign/E1E0/M fields are
// extracted at their NATIVE bit positions and upshifted straight to the bf16 bit
// positions — no normalization right-shift.  Low nibble (bits 0-3 of the byte):
//   sign_l = bit3 (0/8)         -> <<12 -> bit15  (8<<12 = 0x8000)
//   e_l    = bits1-2 (E1E0<<1)  -> <<6  -> E1E0<<7 (the exp contribution)
//   m_l    = bit0 (M)            -> <<6  -> bit6   (M<<6 mantissa)
// High nibble (bits 4-7 of the byte), same fields shifted +4:
//   sign_h = bit7 (0/128)       -> <<8  -> bit15  (128<<8 = 0x8000)
//   e_h    = bits5-6 (E1E0<<5)  -> <<2  -> E1E0<<7
//   m_h    = bit4 (M<<4)         -> <<2  -> bit6   (M<<6)
// exp = (sc-1)<<7 + E1E0<<7  (E1E0<<7 = e_l<<6 = e_h<<2).  low/high dequant
// separately into 16-elem vectors, then interleave [lo0,hi0,lo1,hi1,..] = [k0,k1,..].
static inline void dequant_group(const uint8_t *scales, const uint8_t *nibbles,
                                 int nt, int g, bfloat16 *dq) {
    for (int n = 0; n < 8; ++n) {
        const int rg = (nt * 8 + n) * G + g;
        const uint8_t sc = scales[rg];                       // e8m0 scale byte
        bfloat16 *dqrow = dq + n * 8;                        // dqrow[ks*64 + kr]
        if (sc == 0 || sc == 255) {                           // dead block -> 32 zeros
            for (int ks = 0; ks < 4; ++ks)
                for (int kr = 0; kr < 8; ++kr) dqrow[ks * 64 + kr] = (bfloat16)0;
            continue;
        }
        const uint8_t *nb = nibbles + (size_t)((nt * 8 + n) * G + g) * 16;  // 16-aligned
        // widen to uint16 FIRST so the upshifted field values fit (uint8<<12 overflows).
        const aie::vector<uint16_t, 16> bv =
            aie::load_v<16>(nb).template unpack<uint16_t>(); // each byte 0..255 in low 8 bits
        // low-nibble fields (bits 0-3): bit_and has only (scalar,vec) -> scalar first.
        const aie::vector<uint16_t, 16> sign_l = (uint16_t)0x08 & bv;   // bit3  (0/8)
        const aie::vector<uint16_t, 16> e_l    = (uint16_t)0x06 & bv;  // E1E0<<1 (0/2/4/6)
        const aie::vector<uint16_t, 16> m_l    = (uint16_t)0x01 & bv;   // M       (0/1)
        // high-nibble fields (bits 4-7):
        const aie::vector<uint16_t, 16> sign_h = (uint16_t)0x80 & bv;  // bit7   (0/128)
        const aie::vector<uint16_t, 16> e_h    = (uint16_t)0x60 & bv;  // E1E0<<5 (0/32/64/96)
        const aie::vector<uint16_t, 16> m_h    = (uint16_t)0x10 & bv;  // M<<4    (0/16)

        const uint8_t sc_eff = (sc > 252) ? (uint8_t)252 : sc;  // clamp (scalar)
        const uint16_t scm1_sh7 = (uint16_t)((sc_eff - 1) << 7);  // (sc-1)<<7, broadcast

        // low nibbles -> 16 bf16 bits.  exp=(sc-1)<<7 + E1E0<<7 ; E1E0<<7 = e_l<<6.
        const aie::vector<uint16_t, 16> exp_l = (e_l << (unsigned)6) + scm1_sh7; // vec+scalar
        const aie::mask<16> e_nz_l = e_l != (uint16_t)0;       // E1E0>0 (subnormal clears man)
        const aie::vector<uint16_t, 16> man_l =
            aie::select((uint16_t)0, m_l << (unsigned)6, e_nz_l); // e_nz ? (m_l<<6) : 0
        aie::vector<uint16_t, 16> bits_l =
            (sign_l << (unsigned)12) | exp_l | man_l;
        const aie::mask<16> nz_l = (e_l | m_l) != (uint16_t)0;  // zero-nibble mask
        bits_l = aie::select((uint16_t)0, bits_l, nz_l);        // nz_l ? bits_l : 0

        // high nibbles -> 16 bf16 bits.  E1E0<<7 = e_h<<2 ; M<<6 = m_h<<2.
        const aie::vector<uint16_t, 16> exp_h = (e_h << (unsigned)2) + scm1_sh7;
        const aie::mask<16> e_nz_h = e_h != (uint16_t)0;
        const aie::vector<uint16_t, 16> man_h =
            aie::select((uint16_t)0, m_h << (unsigned)2, e_nz_h);
        aie::vector<uint16_t, 16> bits_h =
            (sign_h << (unsigned)8) | exp_h | man_h;
        const aie::mask<16> nz_h = (e_h | m_h) != (uint16_t)0;
        bits_h = aie::select((uint16_t)0, bits_h, nz_h);

        // interleave: bits[2i]=bits_l[i] (k=2i, low nibble), bits[2i+1]=bits_h[i] (k=2i+1).
        const aie::vector<uint16_t, 32> bits =
            aie::concat(aie::interleave_zip(bits_l, bits_h, 1));   // [k0,k1,..,k31]

        // store the uint16 bits into dq (reinterpreted as bf16 by the MMUL load);
        // 4 strided 8-bf16 chunks at dqrow + ks*64 (8-B-aligned => unaligned store)
        uint16_t *dqrow_u = reinterpret_cast<uint16_t *>(dqrow);
        aie::store_unaligned_v(dqrow_u + 0 * 64, bits.template extract<8>(0));
        aie::store_unaligned_v(dqrow_u + 1 * 64, bits.template extract<8>(8));
        aie::store_unaligned_v(dqrow_u + 2 * 64, bits.template extract<8>(16));
        aie::store_unaligned_v(dqrow_u + 3 * 64, bits.template extract<8>(24));
    }
}

// Re-stream-h BF16-MMUL matvec (aie::mmul<8,8,8> EMULATED bfp16 — Phase 4a:
// dequant is now VECTOR bitwise (see dequant_group above), bit-exact to the
// prior scalar LUT path; the MMUL path below is unchanged).
// The emulate flag (top of file) routes <8,8,8> to mac_8x8_8x8T_conf + transpose +
// to_v64bfp16ebs8 (NOT mac_4x8_8x8_bf16 => NO -4 register underflow, proven by the
// perm diag: C[0,n]=n*8+k_imp, clean at k_imp=0).  K=8 per mmul => contiguous-k
// dequant (vectorizable, the fast <4,8,8> pattern) WITHOUT the -4 that killed
// native <4,8,8>/<8,8,8>.  Risk: bfp16ebs8 ~7-bit mantissa (validated by argmax).
//
// Layout (vectorized dequant): per (N-tile, group) the 8 outputs × 32 k are
// dequanted ONCE (amortized scale, vector nibble load) into the [ks][n][k_rel]
// staging dq[ks*64 + n*8 + k_rel] = dequant(output_n, ks*8+k_rel).  Each K-sub
// then reads its 8×8 = 64 elements with a SINGLE contiguous load_v<64>(dq +
// ks*64) + the proven external transpose (B_op[n*8+k_rel] = dequant(output_n,
// ks*8+k_rel) = the diag's B_buf layout), so the B operand costs 1 vector load
// + 1 transpose per K-sub (no per-K-sub scalar dequant, no per-element
// gather).  The emulated mmul's internal transpose + the external transpose
// pair to read dq as B_read[k,n] = dequant(output_n, K_idx+k) (the diag).
//   A[m,k] = h[K_idx+k] for all m (broadcast M=8 — 7 M-rows wasted, the r=8 tile
//     minimum for an M=1 matvec).
//   C.mac(A,B_op) => C[0,n] = sum_k h[K_idx+k]*dq(W[output_n,K_idx+k]) = partial.
// One packet = one (n_chunk,k_chunk): G=32 groups × 4 K-subs × K=8 = 1024 k.
// Two N-tiles (outputs 0..7, 8..15) processed sequentially (1 live MMUL — the
// 2-live-MMUL stack crash: a vector slot lands at a non-64-B-aligned offset).
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
    alignas(128) bfloat16 dq[256];      // [k=32, n=8] k-slow n-fast (1 N-tile group)

    for (int nt = 0; nt < 2; ++nt) {                 // N-tile 0 (rows 0..7), 1 (8..15)
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
                aie::vector<bfloat16, 64> B_op = aie::load_v<64>(dq + ks * 64);   // NO external transpose (T-variant handles it)
                C.mac(A_op, B_op);
            }
        }
        aie::vector<float, 64> cv = C.template to_vector<float>();
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            float v = cv[n];                          // C row 0 = the matvec output
            int idx = row_base + nt * 8 + n;
            if (kc == 0) o[idx] = v; else o[idx] += v;
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