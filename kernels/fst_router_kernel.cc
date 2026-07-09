// SPDX-License-Identifier: Apache-2.0
//
// fst_router_kernel.cc -- Vectorized Router: GEMM tile + activation + top-k
//
// AIE2P (NPU2) target.
// GEMM tile: hidden[m, k] × weights[k, n] → logits[m, n]
// Uses aie::mmul<4,8,8,bf16,bf16,accauto> with 2x2 expansion.
// Post-GEMM: sqrt(softplus(x)) activation + bitonic top-k.

#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Vectorized mmul tile: bf16 GEMM with 2x2 expansion ────────────────────
// Tile: A[m_tile, k_tile], B[k_tile, n_tile], C[m_tile, n_tile]
// m_tile must be multiple of 8, k_tile and n_tile multiples of 8.
// Uses aie::mmul<4,8,8,bf16,bf16,accauto> for native bf16 multiply-accumulate.

template <unsigned m, unsigned k, unsigned n>
static inline void
fst_matmul_bf16(const bfloat16 *__restrict pA,
                const bfloat16 *__restrict pB,
                bfloat16 *__restrict pC)
{
    constexpr int r = 4;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0, "m must be multiple of 2*r=8");
    static_assert(k % s == 0,       "k must be multiple of s=8");
    static_assert(n % (2 * t) == 0, "n must be multiple of 2*t=16");

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

// ── Entry points for MLIR-AIE ExternalFunction ─────────────────────────────

extern "C" {

// GEMM tile: A[8,64] × B[64,64] → C[8,64]
void fst_router_gemm(bfloat16 *a, bfloat16 *b, bfloat16 *c)
{
    fst_matmul_bf16<8, 64, 64>(a, b, c);
}

// Zero C accumulator: C[8,64] = 0
void fst_router_zero(bfloat16 *c)
{
    fst_zero_vec<bfloat16, 8 * 64>(c);
}

// ── Post-GEMM: activation + bitonic top-k ──────────────────────────────────
// Vectorized sqrt(softplus(x)) using AIE2P exp2 + LUT.
// Bitonic sort for top-k (inherently sequential, but activation is vectorized).

static inline float approx_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    float xlog2e = x * 1.44269504089f;
    int i = (int)(xlog2e + 127.0f) << 16;
    float expx = *(float *)(&i);
    return (expx > 1e-20f) ? (expx - 1.0f) : 0.0f;
}

void fst_router_post(const bfloat16 *__restrict logits,
                     int32_t *__restrict expert_ids,
                     float *__restrict expert_weights,
                     int M, int N_EXPERTS, int TOP_K)
{
    event0();

    constexpr int VEC = 64;

    for (int m = 0; m < M; m++) {
        float activated[256];
        int indices[256];

        // Vectorized activation: sqrt(softplus(x))
        int e = 0;
        for (; e + VEC <= N_EXPERTS; e += VEC) {
            aie::vector<bfloat16, VEC> v_in = aie::load_v<VEC>(logits + m * N_EXPERTS + e);
            aie::accum<accfloat, VEC> v_flt = aie::mul(v_in, aie::broadcast<bfloat16, VEC>((bfloat16)1.0f));
            aie::vector<float, VEC> v_f = v_flt.to_vector<float>();
            for (int j = 0; j < VEC; j++) {
                float x = v_f[j];
                float sp = approx_softplus(x);
                activated[e + j] = __builtin_aie2p_sqrtf(sp);
                indices[e + j] = e + j;
            }
        }
        // Scalar tail
        for (; e < N_EXPERTS; e++) {
            float x = (float)logits[m * N_EXPERTS + e];
            float sp = approx_softplus(x);
            activated[e] = __builtin_aie2p_sqrtf(sp);
            indices[e] = e;
        }

        // Pad to power-of-2 for bitonic sort
        int N2 = 1;
        while (N2 < N_EXPERTS) N2 <<= 1;
        for (int j = N_EXPERTS; j < N2; j++) {
            activated[j] = -1.0f;
            indices[j] = j;
        }

        // Bitonic sort descending
        for (int stride = N2 / 2; stride >= 1; stride /= 2) {
            for (int offset = 0; offset < N2; offset += stride * 2) {
                for (int i = offset; i < offset + stride; i++) {
                    if (activated[i] < activated[i + stride]) {
                        float tv = activated[i];
                        activated[i] = activated[i + stride];
                        activated[i + stride] = tv;
                        int ti = indices[i];
                        indices[i] = indices[i + stride];
                        indices[i + stride] = ti;
                    }
                }
            }
        }

        // Vectorized weight normalization
        float sw = 0.0f;
        int k = 0;
        for (; k + 4 <= TOP_K; k += 4) {
            aie::vector<float, 4> w = aie::load_v<4>(activated + k);
            aie::accum<accfloat, 4> acc = aie::add(aie::zeros<accfloat, 4>(),
                                                    aie::broadcast<float, 4>(sw));
            // Scalar reduction (4 elements)
            sw += activated[k] + activated[k+1] + activated[k+2] + activated[k+3];
        }
        for (; k < TOP_K; k++) sw += activated[k];

        float inv_sw = 1.0f / (sw + 1e-12f);
        aie::vector<float, VEC> inv_sw_vec = aie::broadcast<float, VEC>(inv_sw);

        k = 0;
        for (; k + VEC <= TOP_K; k += VEC) {
            aie::vector<float, VEC> w = aie::load_v<VEC>(activated + k);
            aie::accum<accfloat, VEC> scaled = aie::mul(w, inv_sw_vec);
            aie::vector<float, VEC> result = scaled.to_vector<float>();
            for (int j = 0; j < VEC && (k+j) < TOP_K; j++) {
                expert_weights[m * TOP_K + k + j] = result[j];
            }
        }
        for (; k < TOP_K; k++) {
            expert_weights[m * TOP_K + k] = activated[k] * inv_sw;
        }
        for (k = 0; k < TOP_K; k++) {
            expert_ids[m * TOP_K + k] = indices[k];
        }
    }

    event1();
}

} // extern "C"
