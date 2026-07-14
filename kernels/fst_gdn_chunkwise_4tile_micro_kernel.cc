// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_chunkwise_4tile_micro_kernel.cc — Stage 1.1c 4-TILE 8 KB-stack
// INTERNAL-LOOP micro-test (decisive 4-tile reduction test).
//
// Resolves the chunkwise premise's two remaining risks together:
//   (a) Is a stack-local bfloat16 S array RMW FAST + CLEAN across 48 v-heads
//       when it FITS one data-memory bank (8 KB)?  (PROVEN for 1 tile in the
//       8kstack single-call internal-loop test: 1.6 ms, all 48 == 264.0.)
//   (b) Does the 3-hop core↔core ObjectFifo chain (T0->T1->T2->T3) transfer
//       cleanly when each tile is a SINGLE-CALL INTERNAL-LOOP worker?
//       (1 hop PROVEN PASS in the minimal cross micro-test; the previous
//       PER-V-HEAD 3-hop chain broke, but that was the repeated-8KB-frame
//       call pattern, now eliminated.)
//
// KEY STRUCTURE (the load-bearing IRON pattern, proven in the 8kstack test):
// each tile's TOP-LEVEL external function declares the 8 KB bfloat16 S array
// ONCE and loops the 48 v-heads INTERNALLY.  Helpers take S* (tiny frames) and
// are pure functions of (S, gdec, row_offset).  The worker core calls the top
// function EXACTLY ONCE per dispatch — so the 8 KB frame is entered only once
// per tile and the SP does not restore between v-heads (the repeated-large-
// frame-call breakage is avoided).
//
// Why 4 tiles x 8 KB: single-tile 32 KB hit the load/store immediate limit
// [-32768,-64]; 2-tile 16 KB hit a bank-allocation wall (16 KB + frame spans
// bank 0 into bank 1, where the ObjectFifo buffer is placed).  8 KB fits bank 0
// with frame room, leaving banks 1-3 for ObjectFifo buffers, and is far inside
// the immediate limit.
//
// Cross-tile reduction: ONE 96-float packet per tile pair carries
//   [ gdec(48) | partials(48) ]
// gdec rides the chain (each tile reads gdec[v] for v-head v); the running
// partial sum accumulates one tile's contribution per hop.  T0 seeds from the
// shim input (gdec per v-head); T3 drains the 48 totals to the shim output.
//
// Shim discipline (PROVEN minimal-cross pattern): ONE shim MM2S (f_in0 -> T0,
// 384 floats = 48 v-heads x [gdec, pad x7]) + ONE shim S2MM (T3 -> f_out3, 384
// floats).  T1/T2 have no shim (only core↔core fifos).  Total 1 MM2S + 1 S2MM.
//
//   f_in0  shim MM2S -> T0 : in[v*8+0] = gdec_v   (rows are HARDCODED per tile:
//                                                   T0 rows 1..32, T1 33..64,
//                                                   T2 65..96, T3 97..128)
//   f_c01 T0.prod -> T1.cons : [ gdec(48) | partial0(48) ]
//   f_c12 T1.prod -> T2.cons : [ gdec(48) | partial0+partial1(48) ]
//   f_c23 T2.prod -> T3.cons : [ gdec(48) | partial0+1+2(48) ]
//   f_out3 T3.prod -> shim S2MM : out[v*8+0] = Σ partials = total_v
//
//   Expected total = 0.5 * Σ_{1..128} r = 4128.0 for all 48 v-heads
//   (gdec=0.5, K=8 RMW; partial0=264, +776, +1288, +1800 = 4128.0).

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV    = 128;
constexpr int QUART = 32;          // rows per tile (8 KB bf16: 32*128*2 = 8192)
constexpr int VB    = 16;
constexpr int NVB   = HV / VB;      // 8
constexpr int KST   = 8;
constexpr int NVH   = 48;
constexpr int INPKT = 8;            // shim input stride per v-head
constexpr int OUTPKT= 8;           // shim output stride per v-head
constexpr int CROSS = 96;          // [gdec(48) | partials(48)]

// Pure helper: fill S[QUART*HV] (rows row_offset+1 .. +32), RMW K=8 (gdec),
// return Σ over all 128 cols.  S is the CALLER's 8 KB frame (passed in); this
// function has a tiny stack frame.  Bit-identical to the 8kstack test math.
static inline float gdn_quart_partial(bfloat16 *__restrict S, float gdec, int row_offset) {
    for (int r = 0; r < QUART; ++r) {
        bfloat16 *row = S + r * HV;
        bfloat16 val_bf = (bfloat16)(float)(row_offset + r + 1);
        aie::vector<bfloat16, VB> bv = aie::broadcast<bfloat16, VB>(val_bf);
        for (int n = 0; n < NVB; ++n) aie::store_v(row + n * VB, bv);
    }
    {
        const bfloat16 gdec_bf = (bfloat16)gdec;
        aie::vector<bfloat16, VB> gv = aie::broadcast<bfloat16, VB>(gdec_bf);
        for (int k = 0; k < KST; ++k)
            for (int r = 0; r < QUART; ++r) {
                bfloat16 *row = S + r * HV;
                for (int n = 0; n < NVB; ++n) {
                    aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
                    aie::accum<accfloat, VB> acc = aie::mul(sv, gv);
                    aie::store_v(row + n * VB, acc.to_vector<bfloat16>());
                }
            }
    }
    float sum = 0.0f;
    aie::vector<bfloat16, VB> ones = aie::broadcast<bfloat16, VB>((bfloat16)1.0f);
    for (int r = 0; r < QUART; ++r) {
        bfloat16 *row = S + r * HV;
        for (int n = 0; n < NVB; ++n) {
            aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
            aie::accum<accfloat, VB> acc = aie::mul(sv, ones);
            sum += aie::reduce_add(acc.to_vector<float>());
        }
    }
    return sum;
}

// T0: seed.  Reads gdec per v-head from the shim input; writes the chain
// packet [gdec(48) | partial0(48)].  rows 1..32 (row_offset=0).
extern "C" void gdn_micro_t0(const float *__restrict in, float *__restrict xout) {
    bfloat16 S[QUART * HV];                 // 8 KB — ONE frame for all 48 (no re-call)
    for (int v = 0; v < NVH; ++v) {
        const float gdec = in[v * INPKT + 0];
        xout[v]            = gdec;                              // propagate gdec
        xout[NVH + v]      = gdn_quart_partial(S, gdec, 0);    // rows 1..32
    }
}
// T1: rows 33..64 (row_offset=32).  Reads [gdec(48)|run(48)]; adds partial1.
extern "C" void gdn_micro_t1(const float *__restrict xc, float *__restrict xout) {
    bfloat16 S[QUART * HV];
    for (int v = 0; v < NVH; ++v) {
        const float gdec = xc[v];
        const float run  = xc[NVH + v];
        xout[v]        = gdec;
        xout[NVH + v]  = run + gdn_quart_partial(S, gdec, 32);
    }
}
// T2: rows 65..96 (row_offset=64).
extern "C" void gdn_micro_t2(const float *__restrict xc, float *__restrict xout) {
    bfloat16 S[QUART * HV];
    for (int v = 0; v < NVH; ++v) {
        const float gdec = xc[v];
        const float run  = xc[NVH + v];
        xout[v]        = gdec;
        xout[NVH + v]  = run + gdn_quart_partial(S, gdec, 64);
    }
}
// T3: rows 97..128 (row_offset=96).  Drains the 48 totals to the shim output.
extern "C" void gdn_micro_t3(const float *__restrict xc, float *__restrict out) {
    bfloat16 S[QUART * HV];
    for (int v = 0; v < NVH; ++v) {
        const float gdec = xc[v];
        const float run  = xc[NVH + v];
        out[v * OUTPKT + 0] = run + gdn_quart_partial(S, gdec, 96);
    }
}