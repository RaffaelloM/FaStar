// SPDX-License-Identifier: Apache-2.0
//
// fst_qk_kernel.cc -- Vectorized QK attention: GEMM tile + scale
//
// AIE2P (NPU2) target.
// GEMM tile: Q[m, k] × K_T[k, n] → scores[m, n]
// Uses aie::mmul<4,8,8,bf16,bf16,accauto> with 2x2 expansion.
// Post-GEMM: vectorized scale by 1/sqrt(K_full_dim).

#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Vectorized mmul tile: bf16 GEMM with 2x2 expansion ────────────────────

template <unsigned m, unsigned k, unsigned n>
static inline void
fst_matmul_bf16(const bfloat16 *__restrict pA,
                const bfloat16 *__restrict pB,
                bfloat16 *__restrict pC)
{
    constexpr int r = 4;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;

    constexpr unsigned rowA = m / r;
    constexpr unsigned colA = k / s;
    constexpr unsigned colB = n / t;

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
                const bfloat16 *__restrict pB1 = pB + (j) * MMUL::size_B;
                const bfloat16 *__restrict pB2 = pB + (j + 1) * MMUL::size_B;

                aie::vector<bfloat16, MMUL::size_A> A0;
                aie::vector<bfloat16, MMUL::size_A> A1;
                aie::vector<bfloat16, MMUL::size_B> B0;
                aie::vector<bfloat16, MMUL::size_B> B1;

                aie::vector<bfloat16, MMUL::size_C> acc_C00 = aie::zeros<bfloat16, MMUL::size_C>();
                aie::vector<bfloat16, MMUL::size_C> acc_C01 = aie::zeros<bfloat16, MMUL::size_C>();
                aie::vector<bfloat16, MMUL::size_C> acc_C10 = aie::zeros<bfloat16, MMUL::size_C>();
                aie::vector<bfloat16, MMUL::size_C> acc_C11 = aie::zeros<bfloat16, MMUL::size_C>();

                MMUL C00(acc_C00);
                MMUL C01(acc_C01);
                MMUL C10(acc_C10);
                MMUL C11(acc_C11);

                for (unsigned i = 0; i < colA; ++i)
                    chess_flatten_loop
                {
                    A0 = aie::load_v<MMUL::size_A>(pA1);
                    pA1 += MMUL::size_A;
                    A1 = aie::load_v<MMUL::size_A>(pA2);
                    pA2 += MMUL::size_A;
                    B0 = aie::load_v<MMUL::size_B>(pB1);
                    pB1 += MMUL::size_B * colB;
                    B1 = aie::load_v<MMUL::size_B>(pB2);
                    pB2 += MMUL::size_B * colB;

                    C00.mac(A0, B0);
                    C01.mac(A0, B1);
                    C10.mac(A1, B0);
                    C11.mac(A1, B1);
                }

                aie::store_v(pC1, C00.template to_vector<bfloat16>());
                pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<bfloat16>());
                pC1 += MMUL::size_C;
                aie::store_v(pC2, C10.template to_vector<bfloat16>());
                pC2 += MMUL::size_C;
                aie::store_v(pC2, C11.template to_vector<bfloat16>());
                pC2 += MMUL::size_C;
            }
        }

    event1();
}

// ── Vectorized zero ────────────────────────────────────────────────────────
template <typename T, unsigned N>
static inline void fst_zero_vec(T *__restrict c)
{
    constexpr int r = 512 / (sizeof(T) * 8);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    T *__restrict c_end = c + N;
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
}

// ── Entry points ───────────────────────────────────────────────────────────

extern "C" {

// GEMM tile: Q[8,64] × K_T[64,64] → scores[8,64]
void fst_qk_gemm(bfloat16 *q, bfloat16 *k_t, bfloat16 *scores)
{
    fst_matmul_bf16<8, 64, 64>(q, k_t, scores);
}

// Zero C accumulator: scores[8,64] = 0
void fst_qk_zero(bfloat16 *scores)
{
    fst_zero_vec<bfloat16, 8 * 64>(scores);
}

// Vectorized scale: multiply all elements by 1/sqrt(K_full_dim)
// Input: scores[8, n] bf16 where n = 64 * N_tiles
// Processes 64 bf16 elements per vector iteration.
void fst_qk_scale(bfloat16 *__restrict data, int32_t K_full_dim)
{
    event0();

    constexpr int VEC = 64;
    float scale_f = 1.0f / __builtin_aie2p_sqrtf((float)K_full_dim);
    aie::vector<bfloat16, VEC> scale_vec = aie::broadcast<bfloat16, VEC>((bfloat16)scale_f);

    // Process 8 rows × N columns. Total elements = 8 * N.
    // The caller passes the full tile (8 * 64 = 512 elements).
    constexpr int TOTAL = 8 * 64;
    for (int i = 0; i < TOTAL; i += VEC) {
        aie::vector<bfloat16, VEC> v = aie::load_v<VEC>(data + i);
        v = aie::mul(v, scale_vec).to_vector<bfloat16>();
        aie::store_v(data + i, v);
    }

    event1();
}

} // extern "C"
