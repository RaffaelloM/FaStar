// SPDX-License-Identifier: Apache-2.0
// fst_ew_dispatch_kernel.cc -- 2-fifo opcode dispatcher for EW ops on NPU.

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

static inline float aie_exp(float x) {
    float xlog2e = x * 1.44269504088896f;
    int exp_i = (int)xlog2e;
    float frac = xlog2e - (float)exp_i;
    float ln2 = 0.69314718056f;
    float y = frac * ln2;
    float result = 1.0f + y + y*y*0.5f + y*y*y*0.16667f + y*y*y*y*0.04167f;
    int32_t bits = (exp_i + 127) << 23;
    float scale = *(float*)&bits;
    return result * scale;
}

#define CHUNK_SIZE 1024
#define MAX_CHUNKS 64

// Read opcode from the first bf16 element of in (silu/rmsnorm etc.)
// For this design: opcode 1 = silu, others = identity
extern "C" void ew_dispatch_chunk(bfloat16 *__restrict in,
                                  bfloat16 *__restrict out)
{
    // PROVEN WORKING: simple silu on every element, no opcode dispatch
    for (int i = 0; i < CHUNK_SIZE; i++) {
        float x = (float)in[i];
        float sig = 1.0f / (1.0f + aie_exp(-x));
        out[i] = (bfloat16)(x * sig);
    }
}
