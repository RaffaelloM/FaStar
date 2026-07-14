// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_ffn_nibprobe_kernel.cc — A3 vector-nibble correctness probe (Fork A).
//
// De-risk the hardest IRON element of the float-fifo NPU FFN path: VECTORIZED
// MXFP4 dequant from a FLOAT-typed ObjectFifo.  The probe establishes whether
// the proven fused-FFN dequant pattern (fst_fused_dequant_gemm_kernel.cc) —
// `aie::load_v`-as-uint8 (byte-perfect vector load) + `vector::operator[]`
// (register extract, NOT memory-scalar) + scalar FP4_LUT[] — works on a FLOAT
// fifo, where the Phase-2 attempt failed only because it `load_v`-as-FLOAT
// (wrong reinterpret, garbage) instead of `load_v`-as-uint8.
//
// Two A/B variants on the SAME float fifo:
//   nib_vec — load_v<32>-as-uint8 + operator[] register nibble extract + FP4[].
//             The path under test (vectorized DMA load, register dequant).
//   nib_sca — scalar MEMORY reads p[i] + scalar FP4[].  Control: confirms
//             whether scalar memory reads of a float fifo deliver the bytes
//             (the postmortem claimed they return a ramp — tested here).
//
// Output: 32 floats = [16 low-nibble FP4, 16 high-nibble FP4].

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int NBYTE = 16;   // 16 weight bytes -> 32 nibbles

static const float FP4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

// load_v-as-uint8 (byte-perfect on a float fifo) + operator[] register extract.
extern "C" void nib_vec(const float *__restrict pkt, float *__restrict out)
{
    const uint8_t *p = reinterpret_cast<const uint8_t *>(pkt);
    aie::vector<uint8_t, 32> wb32 = aie::load_v<32>(p);     // byte-perfect vector load
    for (int i = 0; i < NBYTE; ++i) {
        uint8_t b = wb32[i];                                // operator[] reads the REGISTER
        out[i]       = FP4[b & 0x0F];
        out[16 + i]  = FP4[(b >> 4) & 0x0F];
    }
}

// Scalar control: scalar MEMORY reads p[i] of the float packet + scalar FP4[].
// Tests whether scalar memory reads of a float fifo deliver the packed bytes
// (the postmortem claimed they return a ramp on a float fifo).
extern "C" void nib_sca(const float *__restrict pkt, float *__restrict out)
{
    const uint8_t *p = reinterpret_cast<const uint8_t *>(pkt);
    for (int i = 0; i < NBYTE; ++i) {
        uint8_t b = p[i];                                   // scalar MEMORY read
        out[i]       = FP4[b & 0x0F];
        out[16 + i]  = FP4[(b >> 4) & 0x0F];
    }
}