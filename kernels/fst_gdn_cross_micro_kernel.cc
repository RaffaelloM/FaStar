// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_cross_micro_kernel.cc — MINIMAL core↔core ObjectFifo micro-test.
//
// Isolates the 2-tile split's #1 risk (a worker→worker ObjectFifo actually
// transferring on this NPU2 / IRON build) from the 16 KB stack-array risk,
// which on its own hits an AIE2P bank-allocation wall (16 KB + frame spans 2
// data-memory banks and collides with the ObjectFifo buffer placed at bank 1).
//
// This kernel uses NO large stack array — each worker computes its partial
// straight from the input scalars (gdec, row_offset) in a tiny register loop —
// so it compiles with default ~1 KB stacks.  The ONLY thing under test is the
// core↔core stream f_cross (W0.prod -> W1.cons) plus the per-tile shim pattern
// (1 MM2S + <=1 S2MM, no 2-S2MM race).
//
//   W0 (tile 0): in0 = [gdec, row_offset=0, pad..]; partial0 = gdec*Σ_{1..64} r ;
//               SEND partial0 via f_cross.prod   (0 shim S2MM)
//   W1 (tile 1): in1 = [gdec, row_offset=64, pad..]; partial1 = gdec*Σ_{65..128} r;
//               RECV partial0 via f_cross.cons; total = partial0+partial1;
//               DRAIN total (1 shim S2MM)
//
// Expected total (gdec=0.5, bit-exact in fp32):
//   0.5 * (1+..+128) = 0.5 * 8256 = 4128.0
// partial0 = 0.5*Σ_{1..64} = 0.5*2080 = 1040.0 ; partial1 = 0.5*6176 = 3088.0
// If f_cross transfers cleanly, total == 4128.0 for all 48 v-heads.
// A silent f_cross miss (W1 reads 0 / stale) or a 2-S2MM race makes total ≠ 4128.0.

#include <aie_api/aie.hpp>
#include <stdint.h>

// partial = gdec * Σ_{r=1..HALF} (row_offset + r).  Closed-form, tiny stack.
static inline float half_partial(const float *__restrict in) {
    const float gdec       = in[0];
    const int   row_offset = (int)in[1];
    // Σ_{r=1..HALF} (row_offset + r) = HALF*row_offset + HALF*(HALF+1)/2
    constexpr int HALF = 64;
    float s = 0.0f;
    for (int r = 1; r <= HALF; ++r) s += (float)(row_offset + r);
    return gdec * s;          // fp32, bit-exact
}

extern "C" void gdn_cross_w0(const float *__restrict in, float *__restrict xout) {
    xout[0] = half_partial(in);          // partial0 -> f_cross
}

extern "C" void gdn_cross_w1(const float *__restrict in,
                              const float *__restrict xin,
                              float *__restrict out) {
    float partial1 = half_partial(in);
    out[0] = xin[0] + partial1;          // total = partial0 + partial1 -> shim
}