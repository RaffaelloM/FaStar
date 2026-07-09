// SPDX-License-Identifier: Apache-2.0
// fst_qk_gemm.cc -- Vectorized bf16 GEMM tile for QK attention

#include <aie_api/aie.hpp>

template <unsigned m, unsigned k, unsigned n>
static inline void
fst_matmul_bf16(const bfloat16 *__restrict pA,
                const bfloat16 *__restrict pB,
                bfloat16 *__restrict pC)
{
    constexpr int r = 4, s = 8, t = 8;
    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
    constexpr unsigned rowA = m / r, colA = k / s, colB = n / t;

    ::aie::set_rounding(aie::rounding_mode::floor);
    event0();

    for (unsigned z = 0; z < rowA; z += 2)
        chess_prepare_for_pipelining chess_loop_range(4, )
        {
            bfloat16 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
            bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

            for (unsigned j = 0; j < colB; j += 2)
                chess_flatten_loop
            {
                const bfloat16 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                const bfloat16 *__restrict pB1 = pB + j * MMUL::size_B;
                const bfloat16 *__restrict pB2 = pB + (j + 1) * MMUL::size_B;

                aie::vector<bfloat16, MMUL::size_C> acc_C00 = aie::zeros<bfloat16, MMUL::size_C>();
                aie::vector<bfloat16, MMUL::size_C> acc_C01 = aie::zeros<bfloat16, MMUL::size_C>();
                aie::vector<bfloat16, MMUL::size_C> acc_C10 = aie::zeros<bfloat16, MMUL::size_C>();
                aie::vector<bfloat16, MMUL::size_C> acc_C11 = aie::zeros<bfloat16, MMUL::size_C>();

                MMUL C00(acc_C00), C01(acc_C01), C10(acc_C10), C11(acc_C11);

                for (unsigned i = 0; i < colA; ++i)
                    chess_flatten_loop
                {
                    aie::vector<bfloat16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
                    aie::vector<bfloat16, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
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

extern "C" void fst_qk_gemm(bfloat16 *q, bfloat16 *k_t, bfloat16 *scores)
{
    fst_matmul_bf16<8, 64, 64>(q, k_t, scores);
}
