// fst_dequant_q4k_kernel.cc — Vectorized Q4_K dequant for AIE2P (NPU2)
//
// Processes 16 Q4_K super-blocks per call. Nibble extraction is scalar
// (32 bytes → 32 nibbles), but the multiply+subtract is fully vectorized
// using aie::mul and aie::sub on 32-element BF16 vectors.
//
// Dequant: val = d * sc_j * L[i] - dmin * m_j

#include <aie_api/aie.hpp>
#include <stdint.h>

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        uint32_t e = 0;
        uint32_t m = mant;
        while (!(m & 0x400)) { m <<= 1; e++; }
        m &= 0x3ff;
        uint32_t bits = sign | ((e + 112) << 23) | (mant << 13);
        float f; __builtin_memcpy(&f, &bits, 4); return f;
    }
    if (exp == 31) return 0.0f;
    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}

extern "C" void dequant_q4k_block(const uint8_t *__restrict in,
                                   bfloat16 *__restrict out)
{
    event0();
    for (int b = 0; b < 16; b++) {
        const uint8_t *blk = in + b * 144;
        bfloat16 *dst = out + b * 256;

        uint16_t d_u16 = blk[0] | (blk[1] << 8);
        uint16_t dmin_u16 = blk[2] | (blk[3] << 8);
        float d = fp16_to_f32(d_u16);
        float dmin = fp16_to_f32(dmin_u16);

        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;

        // Extract all 256 nibbles into a temp BF16 buffer.
        // Nibble layout: byte[i] = L[i] | (L[i+32] << 4) within each 64-element group.
        // Group g (g=0..3): qs[g*32 .. g*32+31]
        //   low nibbles  → elements [2*g*32 .. 2*g*32+31]
        //   high nibbles → elements [(2*g+1)*32 .. (2*g+1)*32+31]
        bfloat16 nib_bf[256];
        for (int g = 0; g < 4; g++) {
            for (int i = 0; i < 32; i++) {
                uint8_t byte = qs[g * 32 + i];
                nib_bf[(2*g) * 32 + i]     = (bfloat16)(float)(byte & 0x0F);
                nib_bf[(2*g+1) * 32 + i]   = (bfloat16)(float)((byte >> 4) & 0x0F);
            }
        }

        // Vectorized scale+min: process 16 BF16 at a time with aie::mul + aie::add
        for (int j = 0; j < 8; j++) {
            uint8_t sc, m;
            get_scale_min_k4(j, scales, &sc, &m);
            bfloat16 dd_bf = (bfloat16)(d * sc);
            bfloat16 dm_bf = (bfloat16)(dmin * m);
            bfloat16 neg_dm_bf = (bfloat16)(-(dmin * m));

            // Process 32 elements as two 16-element halves
            aie::vector<bfloat16, 16> nib0 = aie::load_v<16>(nib_bf + j * 32);
            aie::vector<bfloat16, 16> nib1 = aie::load_v<16>(nib_bf + j * 32 + 16);

            aie::vector<bfloat16, 16> dd_v = aie::broadcast<bfloat16, 16>(dd_bf);
            aie::vector<bfloat16, 16> neg_dm_v = aie::broadcast<bfloat16, 16>(neg_dm_bf);

            // Vector multiply: nib × scale → accfloat → bfloat16
            aie::accum<accfloat, 16> acc0 = aie::mul(nib0, dd_v);
            aie::accum<accfloat, 16> acc1 = aie::mul(nib1, dd_v);
            aie::vector<bfloat16, 16> res0 = acc0.to_vector<bfloat16>();
            aie::vector<bfloat16, 16> res1 = acc1.to_vector<bfloat16>();

            // Vector add negated min (scaled + (-dm) = scaled - dm)
            res0 = aie::add(res0, neg_dm_v);
            res1 = aie::add(res1, neg_dm_v);

            // Store 32 BF16
            aie::store_v(dst + j * 32, res0);
            aie::store_v(dst + j * 32 + 16, res1);
        }
    }
    event1();
}