// SPDX-License-Identifier: Apache-2.0
// fst_elementwise_unified_kernels.cc
// Unified elementwise kernels: rmsnorm, silu, mul, softmax, rope

#include <aie_api/aie.hpp>
#include <stdint.h>

static inline float aie_sqrt(float x) {
    // Use AIE2P hardware sqrt
    int32_t bits;
    memcpy(&bits, &x, 4);
    // Fast inverse sqrt approximation, then Newton step
    float xhalf = 0.5f * x;
    int32_t i = *(int32_t*)&x;
    i = 0x5f3759df - (i >> 1);
    float y = *(float*)&i;
    y = y * (1.5f - xhalf * y * y);
    return x * y;  // x * (1/sqrt(x)) = sqrt(x)
}

static inline float aie_exp(float x) {
    // exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950)
    float xlog2e = x * 1.44269504088896f;
    // ldexp: 2^floor(xlog2e) * 2^frac(xlog2e)
    int exp_i = (int)xlog2e;
    float frac = xlog2e - (float)exp_i;
    // 2^frac via Taylor: 1 + frac*ln2 + (frac*ln2)^2/2 + ...
    float ln2 = 0.69314718056f;
    float y = frac * ln2;
    float result = 1.0f + y + y*y*0.5f + y*y*y*0.16667f + y*y*y*y*0.04167f;
    // Multiply by 2^exp_i using bit manipulation
    int32_t bits = (exp_i + 127) << 23;
    float scale = *(float*)&bits;
    return result * scale;
}

// ── RMSNorm: out = in * rsqrt(mean(in^2) + eps) * weight ──────────────────
extern "C" void ew_rmsnorm(bfloat16 *__restrict out,
                           const bfloat16 *__restrict in,
                           const bfloat16 *__restrict weight,
                           int32_t N)
{
    event0();
    float sum_sq = 0.0f;
    for (int i = 0; i < N; i++) {
        float v = (float)in[i];
        sum_sq += v * v;
    }
    float rms = sum_sq / (float)N;
    float inv_rms = 1.0f / aie_sqrt(rms + 1e-6f);

    for (int i = 0; i < N; i++) {
        float v = (float)in[i] * (float)weight[i] * inv_rms;
        out[i] = (bfloat16)v;
    }
    event1();
}

// ── SiLU: out = x * sigmoid(x) ───────────────────────────────────────────
extern "C" void ew_silu(bfloat16 *__restrict out,
                        const bfloat16 *__restrict in,
                        int32_t N)
{
    event0();
    for (int i = 0; i < N; i++) {
        float x = (float)in[i];
        float sig = 1.0f / (1.0f + aie_exp(-x));
        out[i] = (bfloat16)(x * sig);
    }
    event1();
}

// ── Mul: out = a * b (elementwise) ────────────────────────────────────────
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
        aie::vector<bfloat16, VEC> out_v = prod.to_vector<bfloat16>();
        aie::store_v(out + i, out_v);
    }
    for (; i < N; i++) {
        out[i] = (bfloat16)((float)a[i] * (float)b[i]);
    }
    event1();
}

// ── Softmax: row-wise softmax over N elements ─────────────────────────────
extern "C" void ew_softmax(bfloat16 *__restrict data, int32_t N)
{
    event0();
    float max_val = -1e30f;
    for (int i = 0; i < N; i++) {
        float v = (float)data[i];
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float v = aie_exp((float)data[i] - max_val);
        data[i] = (bfloat16)v;
        sum += v;
    }
    float inv_sum = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < N; i++) {
        data[i] = (bfloat16)((float)data[i] * inv_sum);
    }
    event1();
}

// ── RoPE: apply rotary positional embedding ─────────────────────────────
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