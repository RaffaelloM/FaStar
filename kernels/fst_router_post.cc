// SPDX-License-Identifier: Apache-2.0
// fst_router_post.cc -- Vectorized activation + bitonic top-k for Router

#include <aie_api/aie.hpp>
#include <stdint.h>

static inline float approx_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    float xlog2e = x * 1.44269504089f;
    int i = (int)(xlog2e + 127.0f) << 16;
    float expx = *(float *)(&i);
    return (expx > 1e-20f) ? (expx - 1.0f) : 0.0f;
}

extern "C" void fst_router_post(const bfloat16 *__restrict logits,
                                int32_t *__restrict expert_ids,
                                float *__restrict expert_weights,
                                int M, int N_EXPERTS, int TOP_K)
{
    event0();
    for (int m = 0; m < M; m++) {
        float activated[256];
        int indices[256];

        // Vectorized activation: sqrt(softplus(x))
        constexpr int VEC = 64;
        int e = 0;
        for (; e + VEC <= N_EXPERTS; e += VEC) {
            aie::vector<bfloat16, VEC> v_in = aie::load_v<VEC>(logits + m * N_EXPERTS + e);
            aie::accum<accfloat, VEC> v_flt = aie::mul(v_in, aie::broadcast<bfloat16, VEC>((bfloat16)1.0f));
            aie::vector<float, VEC> v_f = v_flt.to_vector<float>();
            for (int j = 0; j < VEC; j++) {
                float x = v_f[j];
                activated[e + j] = __builtin_aie2p_sqrtf(approx_softplus(x));
                indices[e + j] = e + j;
            }
        }
        for (; e < N_EXPERTS; e++) {
            float x = (float)logits[m * N_EXPERTS + e];
            activated[e] = __builtin_aie2p_sqrtf(approx_softplus(x));
            indices[e] = e;
        }

        // Pad to power-of-2 for bitonic sort
        int N2 = 1;
        while (N2 < N_EXPERTS) N2 <<= 1;
        for (int j = N_EXPERTS; j < N2; j++) { activated[j] = -1.0f; indices[j] = j; }

        // Bitonic sort descending
        for (int stride = N2 / 2; stride >= 1; stride /= 2)
            for (int offset = 0; offset < N2; offset += stride * 2)
                for (int i = offset; i < offset + stride; i++)
                    if (activated[i] < activated[i + stride]) {
                        float tv = activated[i]; activated[i] = activated[i + stride]; activated[i + stride] = tv;
                        int ti = indices[i]; indices[i] = indices[i + stride]; indices[i + stride] = ti;
                    }

        // Sum and normalize
        float sw = 0.0f;
        for (int k = 0; k < TOP_K; k++) sw += activated[k];
        float inv_sw = 1.0f / (sw + 1e-12f);
        for (int k = 0; k < TOP_K; k++) {
            expert_ids[m * TOP_K + k] = indices[k];
            expert_weights[m * TOP_K + k] = activated[k] * inv_sw;
        }
    }
    event1();
}
