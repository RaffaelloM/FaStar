// SPDX-License-Identifier: Apache-2.0
// fst_router_zero.cc -- Zero accumulator for Router GEMM

#include <aie_api/aie.hpp>

extern "C" void fst_router_zero(bfloat16 *c)
{
    constexpr int N = 8 * 64;
    constexpr int r = 512 / (sizeof(bfloat16) * 8);
    const aie::vector<bfloat16, r> zeros = aie::zeros<bfloat16, r>();
    bfloat16 *c_end = c + N;
    for (; c < c_end; c += r)
        aie::store_v(c, zeros);
}
