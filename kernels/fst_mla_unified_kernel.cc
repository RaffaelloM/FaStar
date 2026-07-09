// SPDX-License-Identifier: Apache-2.0
// fst_mla_unified_kernel.cc -- Single C++ source for all 8 MLA tile kernels.
//
// This file is linked into the unified MLA xclbin.  Every function is a
// *tile* kernel: it operates on one AIE DMA tile at a time.  The host-side
// IRON runtime_sequence loops over the output tiles and reduction tiles.
//
// Tile shape used by every kernel:
//   A: [8, 64]  (M_tile=8, K_tile=64)
//   B: [64, 64] (K_tile=64, N_tile=64)
//   C: [8, 64]  (M_tile=8, N_tile=64)

#include <aie_api/aie.hpp>
#include <stdint.h>

// ============================================================================
// Shared 8x64x64 bf16 GEMM accumulating into C (no zeroing)
// ============================================================================
template <unsigned m, unsigned k, unsigned n>
static inline void
mla_matmul_bf16(const bfloat16 *__restrict pA,
                const bfloat16 *__restrict pB,
                bfloat16 *__restrict pC)
{
    constexpr unsigned r = 4, s = 8, t = 8;
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

                // Load partial accumulator from C buffer (zeroed by mla_zero before first MAC)
                aie::vector<bfloat16, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
                aie::vector<bfloat16, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
                aie::vector<bfloat16, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
                aie::vector<bfloat16, MMUL::size_C> acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

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

// ============================================================================
// Zero one C tile (8*64 = 512 bf16)
// ============================================================================
extern "C" void mla_zero(bfloat16 *c)
{
    constexpr int N = 8 * 64;
    constexpr int r = 512 / (sizeof(bfloat16) * 8);
    const aie::vector<bfloat16, r> zeros = aie::zeros<bfloat16, r>();
    bfloat16 *c_end = c + N;
    for (; c < c_end; c += r)
        aie::store_v(c, zeros);
}

// ============================================================================
// Shared bf16 GEMM tile kernel used by qc, kvc, oa, ob, wq_b, k_pe, and
// internally by qk/sv before scaling.
// ============================================================================
extern "C" void mla_gemm(bfloat16 *A, bfloat16 *B, bfloat16 *C)
{
    mla_matmul_bf16<8, 64, 64>(A, B, C);
}

// ============================================================================
// QK scale: multiply each of the 512 output bf16s by 1/sqrt(K_full_dim)
// ============================================================================
extern "C" void mla_qk_scale(bfloat16 *__restrict data, int32_t K_full_dim)
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

// ============================================================================
// SV scale: identity (kept for symmetry with the qk pipeline)
// ============================================================================
extern "C" void mla_sv_scale(bfloat16 *__restrict data, int32_t D)
{
    (void)data;
    (void)D;
}
