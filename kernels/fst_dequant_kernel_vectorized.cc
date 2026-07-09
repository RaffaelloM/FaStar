// fst_dequant_kernel_vectorized.cc — V4 FP4 dequant, AIE2P optimized
//
// Direct nibble→float mapping with 16-entry table.  Inline scale comp.
// chess pipelining enables ~2bf16/cycle.

#include <aie_api/aie.hpp>
#include <stdint.h>

// FP4 nibble (0-15) → float value
// [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
static const float fp4_to_f32[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float e8m0_to_f32(int e) {
    if (e <= 0 || e >= 255) return 0.0f;
    int exp = e - 127;
    if (exp >= 0) return (float)(1ULL << exp);
    return 1.0f / (float)(1ULL << (-exp));
}

extern "C" void fst_dequant_v4_4096(
    const uint8 *__restrict in,
    bfloat16 *__restrict out
) {
    event0();

    for (int k = 0; k < 64; k++)
        chess_prepare_for_pipelining
        chess_loop_range(64, )
    {
        float sf0 = e8m0_to_f32((int)in[2 * k]);
        float sf1 = e8m0_to_f32((int)in[2 * k + 1]);

        for (int row = 0; row < 32; row++)
            chess_flatten_loop
        {
            uint8 byte_val = in[512 + row * 64 + k];
            float lo_val = fp4_to_f32[byte_val & 0x0F];
            float hi_val = fp4_to_f32[byte_val >> 4];

            int off = row * 128 + 2 * k;
            out[off]     = (bfloat16)(lo_val * sf0);
            out[off + 1] = (bfloat16)(hi_val * sf1);
        }
    }

    event1();
}
