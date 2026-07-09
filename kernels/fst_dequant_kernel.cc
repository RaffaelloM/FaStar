// fst_dequant_kernel.cc — V4 FP4 dequant kernel for AIE2P (NPU2)
//
// Reads a 2560-byte MXFP4 tile and produces 4096 BF16 values.
// Tile layout: [0:128) e8m0 scales, [128:192) bf16 bias (unused),
// [192:512) pad, [512:2560) packed nibbles (2 FP4 per byte, lo first).
//
// Compiled by IRON's ExternalFunction at design time.
// Compile flags: -DTILE_BYTES=2560 -DTILE_ELEMS=4096

#include <aie_api/aie.hpp>
#include <stdint.h>

// FP4 level table: [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
static const float fp4_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" void fst_dequant_v4_4096(const uint8 *__restrict in,
                                    bfloat16 *__restrict out)
{
    const uint8 *scales  = in;          // [0:128)  e8m0 exponent bytes
    const uint8 *nibbles = in + 512;    // [512:2560) packed nibbles

    event0();

    // Process columns outer-loop so each e8m0 scale is read once.
    for (int col = 0; col < 128; col++) {
        int e = scales[col];
        float sf = 0.0f;
        if (e > 0 && e < 255) {
            int exp = (int)e - 127;
            if (exp >= 0)
                sf = (float)(1ULL << exp);
            else
                sf = 1.0f / (float)(1ULL << (-exp));
        }

        for (int row = 0; row < 32; row++) {
            int byte_idx = row * 64 + col / 2;
            uint8 b = nibbles[byte_idx];
            uint8 nibble = (col & 1) ? (b >> 4) : (b & 0x0F);
            float val = fp4_table[nibble] * sf;
            out[row * 128 + col] = (bfloat16)val;
        }
    }

    event1();
}
