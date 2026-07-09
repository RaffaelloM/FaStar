// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_dequant_v4_vectorized_kernel.cc — DS4 dense block FP4 → BF16 dequant
//
// AIE2P (NPU2) target. Per call: in[68] uint8, out[128] BF16.
// Processes 4 consecutive dense blocks (4 × 17 = 68 bytes, DMA-aligned).
// Each dense block layout:
//   byte 0:  e8m0 scale (one per 32-element group)
//   bytes 1..16: 16 packed FP4 nibbles (low nibble first, 2 per byte)
// Output: 128 contiguous BF16 values (4 × 32).
//
// Bitwise E8M0→BF16: scale_u8 << 7 gives correct BF16 bit pattern
// (both use 8-bit bias-127 exponent, mantissa=0 for E8M0).

#include <aie_api/aie.hpp>
#include <stdint.h>

// FP4 to BF16 LUT (16 values, 32 bytes)
// Standard values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
static const bfloat16 FP4_LUT[16] = {
    (bfloat16)0.0f,   (bfloat16)0.5f,   (bfloat16)1.0f,   (bfloat16)1.5f,
    (bfloat16)2.0f,   (bfloat16)3.0f,   (bfloat16)4.0f,   (bfloat16)6.0f,
    (bfloat16)0.0f,   (bfloat16)-0.5f,  (bfloat16)-1.0f,  (bfloat16)-1.5f,
    (bfloat16)-2.0f,  (bfloat16)-3.0f,  (bfloat16)-4.0f,  (bfloat16)-6.0f
};

// E8M0 → BF16 via bitwise integer shift (NO float division)
// Both use 8-bit bias-127 exponent. Mantissa = 0 for E8M0.
// bf16_bits = scale_u8 << 7 (aligns exponent to BF16 bits 14-7)
static inline bfloat16 e8m0_to_bf16(uint8_t scale_u8) {
    if (scale_u8 == 0 || scale_u8 == 255)
        return (bfloat16)0.0f;
    union { uint16_t u; bfloat16 f; } cvt;
    cvt.u = (uint16_t)scale_u8 << 7;
    return cvt.f;
}

constexpr int BLOCK_BYTES = 17;
constexpr int BLOCK_ELEMS = 32;
constexpr int BLOCKS_PER_CALL = 4;

// fst_dequant_v4_4096 — dequantize 4 dense FP4 blocks → 128 BF16
// Called 12,288 times per core by the IRON orchestrator.
//
// IMPORTANT: the 4 dense blocks live at offsets 0, 17, 34, 51 within the
// 68-byte call buffer.  Because 17 is not a power of 2, the per-block nibble
// field (offset 1, 18, 35, 52) is NEVER vector-aligned.  A misaligned
// aie::load_v<16> there reads shifted bytes and corrupts the high bit of
// nibbles (visible as 6.0 ↔ -6.0 sign flips).  We therefore read the 16
// nibble bytes with scalar uint8 loads (no alignment requirement) and only
// vectorize the scale × nibble multiply, which operates on the 32-byte
// aligned `vals` buffer below.
extern "C" void fst_dequant_v4_4096(
    const uint8_t *__restrict in,
    bfloat16 *__restrict out
) {
    event0();

    // 32 BF16 = 64 B; 32-byte aligned so both load_v<16> halves are aligned.
    alignas(32) bfloat16 vals[32];

    #pragma unroll
    for (int b = 0; b < BLOCKS_PER_CALL; b++) {
        const uint8_t *blk = in + b * BLOCK_BYTES;
        bfloat16 *dst = out + b * BLOCK_ELEMS;

        // ── Load scale (byte 0) ─────────────────────────────────
        bfloat16 scale = e8m0_to_bf16(blk[0]);

        // ── Read 16 packed bytes (32 nibbles) SCALARLY ───────────
        // Scalar uint8 loads have no alignment constraint, so the 17-byte
        // block stride is safe.  Each byte → 2 FP4 nibbles (low first).
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte = blk[1 + i];
            vals[i * 2]     = FP4_LUT[byte & 0x0F];
            vals[i * 2 + 1] = FP4_LUT[(byte >> 4) & 0x0F];
        }

        // ── Vectorized multiply by scale (two 16-element chunks) ─
        aie::vector<bfloat16, 16> d0 = aie::load_v<16>((const bfloat16 *)vals);
        aie::vector<bfloat16, 16> d1 = aie::load_v<16>((const bfloat16 *)vals + 16);
        aie::vector<bfloat16, 16> s0 = aie::broadcast<bfloat16, 16>(scale);

        aie::vector<bfloat16, 16> r0 = aie::mul(d0, s0).to_vector<bfloat16>();
        aie::vector<bfloat16, 16> r1 = aie::mul(d1, s0).to_vector<bfloat16>();

        // ── Store 32 BF16 output ────────────────────────────────
        aie::store_v(dst, r0);
        aie::store_v(dst + 16, r1);
    }

    event1();
}
