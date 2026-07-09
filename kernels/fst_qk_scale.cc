// SPDX-License-Identifier: Apache-2.0
// fst_qk_scale.cc -- Vectorized scale for QK attention scores

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void fst_qk_scale(bfloat16 *__restrict data, int32_t K_full_dim)
{
    event0();
    constexpr int VEC = 64;
    constexpr int TOTAL = 8 * 64;
    float scale_f = 1.0f / __builtin_aie2p_sqrtf((float)K_full_dim);
    aie::vector<bfloat16, VEC> scale_vec = aie::broadcast<bfloat16, VEC>((bfloat16)scale_f);
    for (int i = 0; i < TOTAL; i += VEC) {
        aie::vector<bfloat16, VEC> v = aie::load_v<VEC>(data + i);
        v = aie::mul(v, scale_vec).to_vector<bfloat16>();
        aie::store_v(data + i, v);
    }
    event1();
}
