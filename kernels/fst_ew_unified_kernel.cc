// SPDX-License-Identifier: Apache-2.0
// fst_ew_unified_kernel.cc -- Single source for all elementwise/router/lm-head
// kernels that live in the unified EW xclbin (hw_context #5).
//
// Every function is a *tile* kernel: it operates on one AIE DMA tile at a
// time.  The host-side IRON runtime_sequence loops over output tiles and
// reduction tiles.  All functions use direct pointer math (aie::load_v /
// aie::store_v) on the memref pointers handed to them by the runtime.
//
// Tile shape used by the GEMM kernels:
//   A: [8, 64]  (M_tile=8, K_tile=64)
//   B: [64, 64] (K_tile=64, N_tile=64)
//   C: [8, 64]  (M_tile=8, N_tile=64)

#include <aie_api/aie.hpp>
#include <stdint.h>

// ============================================================================
// Shared scalar helpers (no libm on AIE)
// ============================================================================
static inline float aie_sqrt(float x) {
    int32_t i = *(int32_t*)&x;
    i = 0x5f3759df - (i >> 1);
    float y = *(float*)&i;
    y = y * (1.5f - 0.5f * x * y * y);
    return x * y;
}

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

// ============================================================================
// 1. RMSNorm  --  out = in * weight / sqrt(mean(in^2) + eps)
//    Operates on ONE row of N elements.  The runtime feeds rows one at a
//    time; N is baked at compile time (4096 for hidden-dim norms).
//    Vectorized: sum_sq via aie::reduce_add on float vectors; the
//    scale+weight+output pass is a vectorized aie::mul + aie::add.
// ============================================================================
extern "C" void ew_rmsnorm(bfloat16 *__restrict out,
                           const bfloat16 *__restrict in,
                           const bfloat16 *__restrict weight,
                           int32_t N)
{
    event0();
    constexpr int VEC = 32;
    /* Pass 1: vectorized sum of squares.
     * Load bf16, widen to float via accfloat, square into acc, reduce. */
    float sum_sq = 0.0f;
    int i = 0;
    for (; i + VEC <= N; i += VEC) {
        aie::vector<bfloat16, VEC> vin = aie::load_v<VEC>(in + i);
        aie::accum<accfloat, VEC> vfp(vin);
        aie::vector<float, VEC> vf = vfp.to_vector<float>();
        aie::accum<accfloat, VEC> sq = aie::mul(vf, vf);
        sum_sq += aie::reduce_add(sq.to_vector<float>());
    }
    for (; i < N; i++) { float v = (float)in[i]; sum_sq += v * v; }

    float rms = sum_sq / (float)N;
    /* aie::sqrt has a scalar specialization for float. */
    float inv_rms = 1.0f / aie::sqrt(rms + 1e-6f);

    /* Pass 2: vectorized out = in * weight * inv_rms.
     * Use aie::mul(bf16,bf16)->accfloat, scale by inv_rms via broadcast bf16 mul. */
    const bfloat16 inv_rms_bf = (bfloat16)inv_rms;
    aie::vector<bfloat16, VEC> invrms_v = aie::broadcast<bfloat16, VEC>(inv_rms_bf);
    i = 0;
    for (; i + VEC <= N; i += VEC) {
        aie::vector<bfloat16, VEC> vin = aie::load_v<VEC>(in + i);
        aie::vector<bfloat16, VEC> vw  = aie::load_v<VEC>(weight + i);
        aie::accum<accfloat, VEC> prod = aie::mul(vin, vw);
        aie::accum<accfloat, VEC> scaled = aie::mul(prod.to_vector<bfloat16>(), invrms_v);
        aie::store_v(out + i, scaled.to_vector<bfloat16>());
    }
    for (; i < N; i++)
        out[i] = (bfloat16)((float)in[i] * (float)weight[i] * inv_rms);
    event1();
}

// ============================================================================
// 2. SiLU  --  out = in * sigmoid(in) = in * (0.5 + 0.5*tanh(0.5*in))
//    Vectorized using the native AIE2P aie::tanh intrinsic (16 float lanes).
//    Each 32-bf16 load is split into two 16-float halves, tanh'd, and the
//    identity silu(x) = x * (0.5 + 0.5*tanh(x/2)) is applied with aie::mul.
// ============================================================================
extern "C" void ew_silu(bfloat16 *__restrict out,
                        const bfloat16 *__restrict in,
                        int32_t N)
{
    event0();
    constexpr int VEC = 32;          /* bf16 native load/store width */
    constexpr int HALF = 16;         /* float/tanh native lane width  */
    const float half_f = 0.5f;
    const bfloat16 half_bf = (bfloat16)0.5f;
    /* Broadcast 0.5 as a bf16 vector for the vectorized sigmoid mul. */
    aie::vector<bfloat16, VEC> half_v = aie::broadcast<bfloat16, VEC>(half_bf);

    int i = 0;
    for (; i + VEC <= N; i += VEC) {
        aie::vector<bfloat16, VEC> x = aie::load_v<VEC>(in + i);

        /* x/2  -> float vector (two 16-lane halves) */
        aie::accum<accfloat, VEC> xfp(x);
        aie::vector<float, VEC> xf = xfp.to_vector<float>();
        aie::vector<float, HALF> xh0 = xf.extract<HALF>(0);
        aie::vector<float, HALF> xh1 = xf.extract<HALF>(1);
        /* scale by 0.5 in float domain */
        aie::vector<float, HALF> xh0s = aie::mul(xh0, half_f).to_vector<float>();
        aie::vector<float, HALF> xh1s = aie::mul(xh1, half_f).to_vector<float>();

        /* tanh -> bf16 (native AIE2P intrinsic, 16 float lanes -> 16 bf16) */
        aie::vector<bfloat16, HALF> t0 = aie::tanh(xh0s);
        aie::vector<bfloat16, HALF> t1 = aie::tanh(xh1s);

        /* sigmoid = 0.5 + 0.5*tanh  (bf16 domain, vectorized add + mul) */
        aie::vector<bfloat16, HALF> s0 = aie::add(half_v.extract<HALF>(0),
                                                 aie::mul(half_v.extract<HALF>(0), t0).to_vector<bfloat16>());
        aie::vector<bfloat16, HALF> s1 = aie::add(half_v.extract<HALF>(1),
                                                 aie::mul(half_v.extract<HALF>(1), t1).to_vector<bfloat16>());

        /* silu = x * sigmoid : bf16 mul, 16 lanes each half */
        aie::vector<bfloat16, HALF> y0 = aie::mul(x.extract<HALF>(0), s0).to_vector<bfloat16>();
        aie::vector<bfloat16, HALF> y1 = aie::mul(x.extract<HALF>(1), s1).to_vector<bfloat16>();

        /* repack two 16-lane bf16 halves into one 32-lane store */
        aie::vector<bfloat16, VEC> y;
        y.insert(0, y0);
        y.insert(1, y1);
        aie::store_v(out + i, y);
    }
    /* Remainder: scalar fallback (N is always a multiple of 1024 here, but
     * keep the tail for correctness on any N). */
    for (; i < N; i++) {
        float x = (float)in[i];
        float sig = 1.0f / (1.0f + aie_exp(-x));
        out[i] = (bfloat16)(x * sig);
    }
    event1();
}

// ============================================================================
// 3. Elementwise Mul  --  out = a * b   (already vectorized; kept as-is)
// ============================================================================
extern "C" void ew_mul(bfloat16 *__restrict out,
                      const bfloat16 *__restrict a,
                      const bfloat16 *__restrict b,
                      int32_t N)
{
    event0();
    constexpr int VEC = 32;
    int i = 0;
    for (; i + VEC <= N; i += VEC) {
        aie::vector<bfloat16, VEC> va = aie::load_v<VEC>(a + i);
        aie::vector<bfloat16, VEC> vb = aie::load_v<VEC>(b + i);
        aie::accum<accfloat, VEC> prod = aie::mul(va, vb);
        aie::store_v(out + i, prod.to_vector<bfloat16>());
    }
    for (; i < N; i++) out[i] = (bfloat16)((float)a[i] * (float)b[i]);
    event1();
}

// ============================================================================
// 4. Softmax (row-wise, single row of N)  --  out = softmax(in)
// ============================================================================
extern "C" void ew_softmax(bfloat16 *__restrict out,
                           const bfloat16 *__restrict in,
                           int32_t N)
{
    event0();
    float max_val = -1e30f;
    for (int i = 0; i < N; i++) {
        float v = (float)in[i];
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float v = aie_exp((float)in[i] - max_val);
        out[i] = (bfloat16)v;
        sum += v;
    }
    float inv_sum = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < N; i++)
        out[i] = (bfloat16)((float)out[i] * inv_sum);
    event1();
}

// ============================================================================
// 5. RoPE  --  apply rotation to (q_pe) using (cos,sin) lookup table (lut)
//    Operates on ROWS rows of COLS columns (pairs of dims).  COLS is even.
// ============================================================================
extern "C" void ew_rope(bfloat16 *__restrict out,
                        const bfloat16 *__restrict in,
                        const bfloat16 *__restrict lut,
                        int32_t ROWS, int32_t COLS)
{
    event0();
    for (int r = 0; r < ROWS; r++) {
        for (int d = 0; d < COLS; d += 2) {
            float c = (float)lut[r * COLS + d];
            float s = (float)lut[r * COLS + d + 1];
            float x0 = (float)in[r * COLS + d];
            float x1 = (float)in[r * COLS + d + 1];
            out[r * COLS + d]     = (bfloat16)(x0 * c - x1 * s);
            out[r * COLS + d + 1] = (bfloat16)(x0 * s + x1 * c);
        }
    }
    event1();
}

// ============================================================================
// Shared 8x64x64 bf16 GEMM tile (aie::mmul<r=4,s=8,t=8>).
// Accumulates into C (C must be zeroed by ew_gemm_zero before the first MAC).
// Used by both router_gemm and lm_head_gemm; the N dimension is tiled by
// the host runtime_sequence, so the tile kernel is identical for both.
// ============================================================================
template <unsigned m, unsigned k, unsigned n>
static inline void
ew_matmul_bf16(const bfloat16 *__restrict pA,
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

                /* SET-then-ACC: seed the accumulators from the EXISTING C tile
                 * (cleared by ew_gemm_zero on the first K-tile, holding the
                 * running partial on subsequent K-tiles) so the mac calls below
                 * ACCUMULATE across the host K-loop instead of overwriting.
                 * Without this each of the 64 mm calls overwrites C and only the
                 * last K-tile survives -> 1/64 of the correct dot product, which
                 * is the lm_head/router_gemm garbage-argmax root cause. */
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

extern "C" void ew_gemm_zero(bfloat16 *c)
{
    constexpr int N = 16 * 64;   // M=16 standardization (tile m=16, n=64)
    constexpr int r = 512 / (sizeof(bfloat16) * 8);
    const aie::vector<bfloat16, r> zeros = aie::zeros<bfloat16, r>();
    bfloat16 *c_end = c + N;
    for (; c < c_end; c += r) aie::store_v(c, zeros);
}

// 6. Router GEMM tile (A[16,64] x B[64,64] -> C[16,64])  [M=16 standardization]
extern "C" void ew_router_gemm(bfloat16 *a, bfloat16 *b, bfloat16 *c)
{
    ew_matmul_bf16<16, 64, 64>(a, b, c);
}

// 7. LM Head GEMM tile (A[16,64] x B[64,64] -> C[16,64])  [M=16 standardization]
extern "C" void ew_lm_head_gemm(bfloat16 *a, bfloat16 *b, bfloat16 *c)
{
    ew_matmul_bf16<16, 64, 64>(a, b, c);
}
