// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_fused_zero_kernel.cc — zero a 16×64 BF16 C tile (1024 BF16).
// Called by the fused-FFN wrapper once per output tile before the K-loop.
// Separate translation unit so the fused gemm and zero ExternalFunctions each
// own a distinct object file (no duplicate symbols at aiecc link time).

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void fst_fused_zero_16x64(bfloat16 *c)
{
    constexpr int N = 16 * 64;                       // 1024
    constexpr int V = 512 / (sizeof(bfloat16) * 8);  // 32
    const aie::vector<bfloat16, V> zeros = aie::zeros<bfloat16, V>();
    bfloat16 *end = c + N;
    for (; c < end; c += V) aie::store_v(c, zeros);
}