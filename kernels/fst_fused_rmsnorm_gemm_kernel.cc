// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_fused_rmsnorm_gemm_kernel.cc — Fused RMSNorm + Q8_0 Dequant + GEMM
//
// AIE2P (NPU2) target. Eliminates the intermediate attn_norm[4096] buffer
// by absorbing RMSNorm into the GEMM tile loop.
//
// Tile dimensions (M=8 tokens, K_tile=32, N_tile=32):
//   A_scaled   [M, K_tile] BF16  — pre-multiplied by per-row rms_scale
//   NormW      [K_tile]    BF16  — RMSNorm weight slice
//   B_q8       [N_tile*34] uint8 — Q8_0 blocks: per column: [scale_u16, q8_0..q8_31]
//   C          [M, N_tile] BF16  — accumulator (zeroed by caller)
//
// RMSNorm math:  sum_sq = Σ x[i]^2, rms = 1/√(sum_sq/N + ε), x_norm = x * rms * weight
// Fused form:    x_norm[k] = x_scaled[row][k] * NormW[k]
// where x_scaled = x_raw[row][k] * rms_scale[row] (pre-computed on host)

#include <aie_api/aie.hpp>
#include <stdint.h>

// Q8_0 scale decode: first 2 bytes are BF16 scale
static inline float q8_0_scale_f32(const uint8_t *blk) {
    uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    union { uint16_t u; float f; } cvt;
    cvt.u = (uint32_t)bits << 16;
    return cvt.f;
}

// Q8_0 block size (matches GGUF Q8_0: 32 int8 + 2 bytes BF16 scale)
#define QK8_0 32

// RMSNorm eps (matches DS4_DEFAULT_RMS_EPS)
#define RMS_EPS 1.0e-6f

extern "C" void fst_fused_rmsnorm_gemm_q8_0_8x32x32(
    const bfloat16 *__restrict pAW,    //  packed: [M=8, K_tile=32] A_scaled
                                       //  followed by [K_tile=32] NormW
    const uint8_t  *__restrict pB,     //  [N_tile=32 * 34] uint8, Q8_0 blocks
    bfloat16       *__restrict pC)     //  [M=8, N_tile=32] BF16, accumulator
{
    constexpr unsigned M = 8;
    constexpr unsigned K = 32;
    constexpr unsigned N = 32;

    const bfloat16 *pA = pAW;               //  first  M*K elements
    const bfloat16 *pW = pAW + M * K;       //  next   K   elements

    // ── Stage 1: Apply RMSNorm to A tile ──────────────────────────────
    // a_normed[row][k] = a_scaled[row][k] * normW[k]
    bfloat16 A_normed[M * K];
    for (unsigned r = 0; r < M; r++)
        chess_prepare_for_pipelining chess_loop_range(M, )
    {
        for (unsigned k = 0; k < K; k++)
            chess_prepare_for_pipelining chess_loop_range(K, )
        {
            A_normed[r * K + k] = (bfloat16)((float)pA[r * K + k] * (float)pW[k]);
        }
    }

    // ── Stage 2: Dequant B and accumulate into C ──────────────────────
    // B_q8 layout: for each of N columns, 34 bytes [scale_u16, q8_0..q8_31]
    // For this K_tile (1 Q8_0 block of 32 K rows):
    //   B[col][row] = scale_bf16 * (float)q8_value
    for (unsigned n = 0; n < N; n++)
        chess_prepare_for_pipelining chess_loop_range(N, )
    {
        const uint8_t *blk = pB + n * 34;
        float scale = q8_0_scale_f32(blk);
        const int8_t *q8 = (const int8_t *)(blk + 2);

        for (unsigned r = 0; r < M; r++)
        {
            float acc = (float)pC[r * N + n];
            for (unsigned k = 0; k < K; k++)
            {
                float a = (float)A_normed[r * K + k];
                float b = scale * (float)q8[k];
                acc += a * b;
            }
            pC[r * N + n] = (bfloat16)acc;
        }
    }
}
