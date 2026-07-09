// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_mla_attention_kernel.cc -- MLA GQA attention kernels for AIE2P
//
// 1. fst_mla_softmax_32x128 -- Row-wise softmax with attn_sinks bias.
//    Per row: max-subtract, LUT-based exp (x in [-10, 0]), normalize.
//
// 2. fst_mla_softmax_32x64  -- Smaller tile variant.
//
// Match ds4.c:7045 layer_attention_rows_one exactly:
//   denom = exp(sinks[h]-max) + Σ exp(score[r]-max)
//   output = Σ exp(score[r]-max) * V[r] / denom
// Only the score normalization is done here (output = softmax @ V is
// handled by a separate scalar matmul).
//
// Uses safe AIE2P patterns: chess_prepare_for_pipelining, scalar float ops.
// No uint4, no vectorized division, no fabricated APIs.

#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Exp LUT for x in [-10, 0] ────────────────────────────────────────────
static const float EXP_LUT[83] = {
    1.00000000e+00f, 8.82496903e-01f, 7.78800786e-01f, 6.87289250e-01f,
    6.06530678e-01f, 5.35261428e-01f, 4.72366566e-01f, 4.16862082e-01f,
    3.67879450e-01f, 3.24652469e-01f, 2.86504817e-01f, 2.52837539e-01f,
    2.23130164e-01f, 1.96911687e-01f, 1.73773944e-01f, 1.53355175e-01f,
    1.35335284e-01f, 1.19432926e-01f, 1.05399225e-01f, 9.30144414e-02f,
    8.20849979e-02f, 7.24397677e-02f, 6.39277706e-02f, 5.64161146e-02f,
    4.97870684e-02f, 4.39369326e-02f, 3.87742078e-02f, 3.42181160e-02f,
    3.01973834e-02f, 2.66490990e-02f, 2.35177445e-02f, 2.07543779e-02f,
    1.83156385e-02f, 1.61635082e-02f, 1.42642165e-02f, 1.25881047e-02f,
    1.11089140e-02f, 9.80356349e-03f, 8.65130059e-03f, 7.63471940e-03f,
    6.73762272e-03f, 5.94593764e-03f, 5.24728560e-03f, 4.63069403e-03f,
    4.08655833e-03f, 3.60636813e-03f, 3.18259209e-03f, 2.80861803e-03f,
    2.47858397e-03f, 2.18733958e-03f, 1.93031617e-03f, 1.70349821e-03f,
    1.50338207e-03f, 1.32678039e-03f, 1.17087924e-03f, 1.03328991e-03f,
    9.11901427e-04f, 8.04737652e-04f, 7.10164462e-04f, 6.26714931e-04f,
    5.53069976e-04f, 4.88077533e-04f, 4.30723945e-04f, 3.80113146e-04f,
    3.35448806e-04f, 2.96034338e-04f, 2.61252395e-04f, 2.30555726e-04f,
    2.03463057e-04f, 1.79554734e-04f, 1.58460441e-04f, 1.39846087e-04f,
    1.23410109e-04f, 1.08906448e-04f, 9.61079345e-05f, 8.48164404e-05f,
    7.48495552e-05f, 6.60495456e-05f, 5.82887347e-05f, 5.14402275e-05f,
    4.53971661e-05f, 4.00649978e-05f, 3.53567735e-05f
};

static inline float exp_neg_bf16(float x) {
    if (x < -10.0f) return 0.0f;
    if (x > 0.0f) return 1.0f;
    float idx = -x * 8.0f;
    int i0 = (int)idx;
    int i1 = i0 + 1;
    if (i1 > 80) { i0 = 80; i1 = 80; }
    float frac = idx - (float)i0;
    float v0 = EXP_LUT[i0];
    float v1 = EXP_LUT[i1];
    return v0 + frac * (v1 - v0);
}

// ── Softmax: scores[m,n] and sinks[m] ────────────────────────────────────
extern "C" void fst_mla_softmax_32x128(
    bfloat16       *__restrict scores,
    const bfloat16 *__restrict sinks)
{
    const unsigned M = 32;
    const unsigned N = 128;

    event0();

    for (unsigned r = 0; r < M; r++)
        chess_prepare_for_pipelining chess_loop_range(M, )
    {
        float max_s = (float)sinks[r];

        for (unsigned c = 0; c < N; c++) {
            float s = (float)scores[r * N + c];
            if (s > max_s) max_s = s;
        }

        float denom = exp_neg_bf16((float)sinks[r] - max_s);

        for (unsigned c = 0; c < N; c++) {
            float s = (float)scores[r * N + c];
            float e = exp_neg_bf16(s - max_s);
            scores[r * N + c] = (bfloat16)e;
            denom += e;
        }

        float inv_denom = 1.0f / denom;
        for (unsigned c = 0; c < N; c++) {
            float v = (float)scores[r * N + c];
            scores[r * N + c] = (bfloat16)(v * inv_denom);
        }
    }

    event1();
}

// ── 32 x 64 variant (for larger KV windows with 4 groups per head) ──────
extern "C" void fst_mla_softmax_32x64(
    bfloat16       *__restrict scores,
    const bfloat16 *__restrict sinks)
{
    const unsigned M = 32;
    const unsigned N = 64;

    event0();

    for (unsigned r = 0; r < M; r++)
        chess_prepare_for_pipelining chess_loop_range(M, )
    {
        float max_s = (float)sinks[r];

        for (unsigned c = 0; c < N; c++) {
            float s = (float)scores[r * N + c];
            if (s > max_s) max_s = s;
        }

        float denom = exp_neg_bf16((float)sinks[r] - max_s);

        for (unsigned c = 0; c < N; c++) {
            float s = (float)scores[r * N + c];
            float e = exp_neg_bf16(s - max_s);
            scores[r * N + c] = (bfloat16)e;
            denom += e;
        }

        float inv_denom = 1.0f / denom;
        for (unsigned c = 0; c < N; c++) {
            float v = (float)scores[r * N + c];
            scores[r * N + c] = (bfloat16)(v * inv_denom);
        }
    }

    event1();
}
