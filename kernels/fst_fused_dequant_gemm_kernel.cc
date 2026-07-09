// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_fused_dequant_gemm_kernel.cc — Fused MXFP4 (.fst 17-byte block) dequant + BF16 GEMM
//
// AIE2P (NPU2) target. Per call (one M×K×N tile):
//   A : BF16[M, K]           row-major, M=16, K=64  -> 1024 BF16 (ObjectFifo element)
//   B : uint8[]              32-byte-padded .fst blocks for the B[K,N] tile
//                           N=64 columns × (K/32)=2 blocks/col = 128 blocks × 32 B = 4096 bytes
//   C : BF16[M, N]           row-major, M=16, N=64  -> 1024 BF16 (read-accumulate-write)
//
// MMUL: aie::mmul<r=4, s=8, t=8, bf16, bf16, accauto>  (native accfloat on AIE2P)
//   A tile: 4×8 (r×s), B tile: 8×8 (s×t), C tile: 4×8 (r×t)
//   rowA = M/r = 4, colA = K/s = 8, colB = N/t = 8
//
// ── MEMORY-ACCESS CONTRACT (the hard-won part) ──────────────────────────────
// On the IRON/AIE2P path, SCALAR loads from a DMA-filled uint8 ObjectFifo element
// do NOT return the DMA'd data (they read a fixed ramp for the first ~256 B),
// while VECTOR loads (aie::load_v) from the SAME element return it byte-perfect
// (proven by FST_FUSED_PROBE).  aie::vector::operator[] reads the vector REGISTER
// (::extract_elem(data, idx)), not memory, so it reliably sees vector-loaded data.
// => This kernel NEVER scalar-reads B.  It vector-loads 64 B (2 padded blocks) per
//    N-column from pB and extracts the scale/nibble bytes via operator[].
//
// .fst dense block layout (per 32 FP4 elements, one N column):
//   byte 0     : e8m0 block scale
//   bytes 1..16: 16 packed nibble bytes; byte i -> elems 2i (low nibble) & 2i+1 (high)
//   FP4 LUT: {0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6}
//   e8m0->bf16: scale_u8 << 7 (bitwise; both use 8-bit bias-127 exponent)
//
// ── B PACKED LAYOUT CONTRACT (host → NPU) ───────────────────────────────────
// The host pads each 17-byte block to 32 bytes (17 data + 15 pad) so a 64-B vector
// load holds exactly 2 aligned blocks (no cross-chunk boundary).  Block index
//   block_idx = n_col * 2 + k_block      (n_col in 0..63, k_block in 0..1)
// sits at byte block_idx * 32.  Block (n_col, k_block) holds the 32 K-elements
// K = k_block*32 .. k_block*32+31 at fixed N = n_col.  Two consecutive k_blocks
// for the same n_col are contiguous (64 B), so one load_v<64> per n_col fetches
// both.  Tile = 128 blocks × 32 B = 4096 B.
//
// ── B_buf scatter layout (col-major, matches kernels.mm b_col_maj) ──────────
// aie::mmul<4,8,8> reads its 64-BF16 B operand with s (K) FAST, t (N) SLOW
// (kernels.mm(b_col_maj=True) streams innermost (s,1)).  So within each 8×8
// sub-tile the element (rr=K%8, n=N%8) lives at  rr + n*8  (rr fast).  Sub-tile
// (ii, nt) base = (ii*colB + nt)*64 BF16.
//
// C: each call produces ONE fresh K-tile partial C[M,N] = A[M,K] @ dequant(B[K,N])
//    (zero-init accumulators, no K-accumulation in the kernel).  The host sums the
//    64 K-tile partials per N-tile to form the full GEMM output.

#include <aie_api/aie.hpp>
#include <stdint.h>

using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16, accauto>;
constexpr unsigned M_TILE = 16, K_TILE = 64, N_TILE = 32;
constexpr unsigned r = 4, s = 8, t = 8;
constexpr unsigned rowA = M_TILE / r;   // 4
constexpr unsigned colA = K_TILE / s;   // 8
constexpr unsigned colB = N_TILE / t;   // 8

constexpr int BLOCK_BYTES  = 17;       // real .fst block
constexpr int BLOCK_ELEMS  = 32;
constexpr int PAD_BYTES    = 18;       // host-padded block stride (17 + 1)
constexpr int K_BLOCKS     = K_TILE / BLOCK_ELEMS;  // 2 blocks per N column
constexpr int N_COLS       = N_TILE;                // 64
constexpr int B_TILE_BYTES = N_COLS * K_BLOCKS * PAD_BYTES;  // 2304

// FP4 (E2M1) -> BF16 lookup
static const bfloat16 FP4_LUT[16] = {
    (bfloat16)0.0f,   (bfloat16)0.5f,   (bfloat16)1.0f,   (bfloat16)1.5f,
    (bfloat16)2.0f,   (bfloat16)3.0f,   (bfloat16)4.0f,   (bfloat16)6.0f,
    (bfloat16)0.0f,   (bfloat16)-0.5f,  (bfloat16)-1.0f,  (bfloat16)-1.5f,
    (bfloat16)-2.0f,  (bfloat16)-3.0f,  (bfloat16)-4.0f,  (bfloat16)-6.0f
};

// E8M0 -> BF16 bitwise (scale_u8 << 7)
static inline bfloat16 e8m0_to_bf16(uint8_t e) {
    if (e == 0 || e == 255) return (bfloat16)0.0f;
    union { uint16_t u; bfloat16 f; } cvt;
    cvt.u = (uint16_t)e << 7;
    return cvt.f;
}

// Scalar BF16 multiply via float (matches the proven dequant kernel's precision:
// (bfloat16)(float_lut * float_scale)).
static inline bfloat16 mul_bf16(bfloat16 a, bfloat16 b) {
    return (bfloat16)((float)a * (float)b);
}

// B_buf: dequanted B[K=64,N=64] tile in the MMUL's native col-major B layout.
// Sub-tile (ii, nt) base = (ii*colB + nt)*64 BF16; within it (rr,n) at rr + n*8.
// 64 sub-tiles × 64 = 4096 BF16 = 8 KB.
static bfloat16 B_buf[colA * colB * 64];

// SOLE combined uint8 input (dequant-proven path: a single uint8 ObjectFifo via
// direct shim->core DOES DMA its BO — the dequant kernel relies on exactly this).
// uint8 as a 2nd input (alongside a bf16 fifo) never DMAs, relay or not.  So both
// A and B travel in ONE uint8 element: A (1024 BF16 = 2048 B) then B (2304 B).
constexpr int A_BYTES = M_TILE * K_TILE * sizeof(bfloat16);   // 2048
constexpr int AB_TILE_BYTES = A_BYTES + B_TILE_BYTES;         // 4352

extern "C" void fst_fused_dequant_gemm_16x64x64(
    const uint8_t  *__restrict pAB,
    bfloat16       *__restrict pC)
{
    const bfloat16 *__restrict pA = reinterpret_cast<const bfloat16 *>(pAB);
    const uint8_t  *__restrict pB = pAB + A_BYTES;
    ::aie::set_rounding(aie::rounding_mode::floor);
    event0();

    // ── Dequant the whole B[K=64,N=64] tile into B_buf (col-major) ───────────
    // SCALAR reads of pB (exactly like the proven fst_dequant_q4k_block kernel,
    // which reads its uint8 ObjectFifo element with scalar pB[k] accesses).
    // Each .fst block is 17 bytes, padded to 18 (PAD_BYTES).  Within the 2304-B
    // tile, block bidx = n_col*K_BLOCKS + k_block sits at byte bidx*18.  All
    // reads stay within [0, 2304) (max byte 2302).  n_col (64) not unrolled.
    for (int n_col = 0; n_col < N_COLS; n_col++) {
        const int nt = n_col >> 3;   // n_col / 8  (N sub-tile index)
        const int n  = n_col & 7;    // n_col % 8  (N position within sub-tile)

        #pragma unroll
        for (int k_block = 0; k_block < K_BLOCKS; k_block++) {
            const int boff = (n_col * K_BLOCKS + k_block) * PAD_BYTES;  // byte in tile
            const bfloat16 scale = e8m0_to_bf16(pB[boff + 0]);

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const uint8_t byte = pB[boff + 1 + i];
                const int k0 = k_block * 32 + i * 2;
                const int k1 = k0 + 1;

                const int ii0 = k0 >> 3, rr0 = k0 & 7;   // K sub-tile, K-within
                const int ii1 = k1 >> 3, rr1 = k1 & 7;

                // col-major scatter: (rr, n) at rr + n*8 (rr=K fast, n=N slow)
                B_buf[(ii0 * colB + nt) * 64 + rr0 + n * 8] =
                    mul_bf16(FP4_LUT[byte & 0x0F], scale);
                B_buf[(ii1 * colB + nt) * 64 + rr1 + n * 8] =
                    mul_bf16(FP4_LUT[(byte >> 4) & 0x0F], scale);
            }
        }
    }

    // Drain scalar B_buf stores before the MMUL's vector reads.
    chess_separator_scheduler_local();

    // ── 2×2-blocked MMUL loop ───────────────────────────────────────────────
    // A is the row-major 16×64 ObjectFifo element.  The MMUL needs each r×s
    // (4×8) A operand in s-fast order (element (r,s) at s + r*8).  We gather it
    // with 4 vector loads of 8 BF16 (one per r-row, 8 contiguous K cols) and
    // concat into a 32-BF16 s-fast register — no scalar reads, no local buffer.
    for (unsigned z = 0; z < rowA; z += 2)
        chess_prepare_for_pipelining chess_loop_range(rowA / 2, )
    {
        bfloat16 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
        bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2)
            chess_flatten_loop
        {
            const bfloat16 *__restrict pB1 = B_buf + j * MMUL::size_B;
            const bfloat16 *__restrict pB2 = B_buf + (j + 1) * MMUL::size_B;

            // Fresh (zero-init) accumulators: each call is independent — the host
            // sums the 64 K-tile partials per N-tile.  No K-accumulation here.
            MMUL C00, C01, C10, C11;

            for (unsigned i = 0; i < colA; ++i)
                chess_flatten_loop
            {
                // A sub-tile (z, i): rows z*4+0..3, cols i*8..i*8+7.  4× load_v<8>
                // (8 contiguous K BF16 per row) -> concat = s-fast 32-BF16 operand.
                const bfloat16 *__restrict pa0 = pA + (z * 4 + 0) * 64 + i * 8;
                const bfloat16 *__restrict pa1 = pA + (z * 4 + 1) * 64 + i * 8;
                const bfloat16 *__restrict pa2 = pA + (z * 4 + 2) * 64 + i * 8;
                const bfloat16 *__restrict pa3 = pA + (z * 4 + 3) * 64 + i * 8;
                aie::vector<bfloat16, MMUL::size_A> A0 =
                    aie::concat(aie::load_v<8>(pa0), aie::load_v<8>(pa1),
                                aie::load_v<8>(pa2), aie::load_v<8>(pa3));
                const bfloat16 *__restrict pa4 = pA + ((z + 1) * 4 + 0) * 64 + i * 8;
                const bfloat16 *__restrict pa5 = pA + ((z + 1) * 4 + 1) * 64 + i * 8;
                const bfloat16 *__restrict pa6 = pA + ((z + 1) * 4 + 2) * 64 + i * 8;
                const bfloat16 *__restrict pa7 = pA + ((z + 1) * 4 + 3) * 64 + i * 8;
                aie::vector<bfloat16, MMUL::size_A> A1 =
                    aie::concat(aie::load_v<8>(pa4), aie::load_v<8>(pa5),
                                aie::load_v<8>(pa6), aie::load_v<8>(pa7));

                aie::vector<bfloat16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * colB;
                aie::vector<bfloat16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * colB;
                C00.mac(A0, B0); C01.mac(A0, B1);
                C10.mac(A1, B0); C11.mac(A1, B1);
            }

            aie::store_v(pC1, C00.template to_vector<bfloat16>()); pC1 += MMUL::size_C;
            aie::store_v(pC1, C01.template to_vector<bfloat16>()); pC1 += MMUL::size_C;
            aie::store_v(pC2, C10.template to_vector<bfloat16>()); pC2 += MMUL::size_C;
            aie::store_v(pC2, C11.template to_vector<bfloat16>()); pC2 += MMUL::size_C;
        }
    }

    event1();
}