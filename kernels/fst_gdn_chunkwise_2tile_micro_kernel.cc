// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_chunkwise_2tile_micro_kernel.cc — Stage 1.1b MICRO-TEST for the
// 2-TILE bf16 S split (the path the user chose after the single-tile 32 KB
// stack proved infeasible).  Resolves the 2-tile split's #1 risk cheaply:
//
//   Is a core↔core worker-to-worker ObjectFifo (one Worker's .prod() feeding
//   another Worker's .cons()) ACTUALLY functional on this NPU2 / IRON build?
//
// Why this is the crux: the M=1 2-S2MM race that killed fusion was a SHIM DMA
// race (core↔DDR).  In a 2-tile split each tile's shim stays at 1 MM2S + 1 S2MM
// (the cross-tile partial-sum + delta traffic routes over the STREAM-SWITCH,
// a separate resource) — so the shim race need not occur.  BUT every proven
// multi-worker design reachable here (repo compile_ffn_multicore.py, IRON's own
// algorithms/transform.py) uses INDEPENDENT workers each with their own shim
// DMA — never a worker→worker core↔core stream.  The 2-tile GDN recurrence
// REQUIRES core↔core exchange every token (the Sᵀ@kn / Sᵀ@qn reductions span
// all 128 rows across both tiles; a shim-mediated DDR round-trip would
// reintroduce the 2-S2MM race).  So the whole split hinges on core↔core
// ObjectFifo working here.  This micro-test proves/disproves it in a handful
// of compile+run attempts, before the full 2-tile kernel.
//
// What it does (one v-head, K=8, gdec=0.5, bf16-exact → rounding-mode-independent):
//
//   Worker 0 (tile 0): 16 KB stack bfloat16 Shalf0[64*128]  (rows 0..63)
//       fill row r <- bf16(r+1); RMW K=8: Shalf0 = 0.5*Shalf0; partial0 = Σ fp32
//       SEND partial0 to W1 via core↔core ObjectFifo (f_cross.prod)
//   Worker 1 (tile 1): 16 KB stack bfloat16 Shalf1[64*128]  (rows 64..127)
//       fill row r <- bf16(r+65); RMW K=8; partial1 = Σ fp32
//       RECV partial0 from W1 via f_cross.cons; total = partial0 + partial1
//       DRAIN total to DDR (shim S2MM)
//
//   Per-tile shim: W0 = 1 MM2S (gdec+rowoff) + 0 S2MM.  W1 = 1 MM2S + 1 S2MM.
//   No 2-S2MM anywhere.  f_cross is the core↔core stream under test.
//
// Expected total (host reference, bit-exact in bf16):
//   128 cols × Σ_{r=1..128} (r * 0.5^8) = 128 × (128·129/2)/256 = 4128.0
// partial0 = 128 × Σ_{1..64} r/256  = 1040.0 ; partial1 = 128 × Σ_{65..128} r/256 = 3088.0
// If f_cross transfers cleanly AND both 16 KB stacks are clean, total == 4128.0.
// A garbage .bss-style failure, a silent f_cross miss, or a 2-S2MM race would
// all make total ≠ 4128.0 (or hang).

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV    = 128;
constexpr int HALF  = 64;          // rows per tile (16 KB bf16: 64*128*2 = 16384)
constexpr int VB    = 16;          // bf16 native lanes (AIE2P)
constexpr int NVB   = HV / VB;     // 8 bf16-vecs per 128-element row
constexpr int KST   = 8;           // recurrence steps (K)

// Shared fill+RMW+partial over a 16 KB stack Shalf[HALF*HV].  row_offset is the
// global row index of this tile's first row (W0: 0, W1: 64) so the two halves
// cover distinct rows 1..128.  Returns the fp32 partial checksum.
static inline float gdn_half_partial(const float *__restrict in) {
    const float gdec = in[0];          // 0.5 (bit-exact in bf16)
    const int   row_offset = (int)in[1];

    bfloat16 Shalf[HALF * HV];         // 16 KB stack — the thing under test (fits the
                                      // [-32768,-64] immediate; far smaller than the
                                      // 32 KB single-tile array that crashed)
    // fill: row r <- bf16(row_offset + r + 1).  Integers exact in bf16.
    for (int r = 0; r < HALF; ++r) {
        bfloat16 *row = Shalf + r * HV;
        bfloat16 val_bf = (bfloat16)(float)(row_offset + r + 1);
        aie::vector<bfloat16, VB> bv = aie::broadcast<bfloat16, VB>(val_bf);
        for (int n = 0; n < NVB; ++n)
            aie::store_v(row + n * VB, bv);
    }
    // RMW K=8: Shalf = gdec*Shalf (bf16 storage, fp32 compute).  With gdec=0.5
    // this is bit-exact in bf16 (exponent only).
    {
        const bfloat16 gdec_bf = (bfloat16)gdec;
        aie::vector<bfloat16, VB> gv = aie::broadcast<bfloat16, VB>(gdec_bf);
        for (int k = 0; k < KST; ++k) {
            for (int r = 0; r < HALF; ++r) {
                bfloat16 *row = Shalf + r * HV;
                for (int n = 0; n < NVB; ++n) {
                    aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
                    aie::accum<accfloat, VB> acc = aie::mul(sv, gv);
                    aie::store_v(row + n * VB, acc.to_vector<bfloat16>());
                }
            }
        }
    }
    // partial checksum: sum all Shalf as fp32 (widen bf16 -> accfloat -> float).
    float sum = 0.0f;
    aie::vector<bfloat16, VB> ones = aie::broadcast<bfloat16, VB>((bfloat16)1.0f);
    for (int r = 0; r < HALF; ++r) {
        bfloat16 *row = Shalf + r * HV;
        for (int n = 0; n < NVB; ++n) {
            aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
            aie::accum<accfloat, VB> acc = aie::mul(sv, ones);
            sum += aie::reduce_add(acc.to_vector<float>());
        }
    }
    return sum;
}

// Worker 0 (tile 0): compute partial0, SEND it via the core↔core ObjectFifo.
//   in[8]        = [gdec, row_offset=0, pad...]
//   xout[8]     = [partial0, pad...]   (f_cross PROD packet, consumed by W1)
extern "C" void gdn_micro_w0(const float *__restrict in,
                              float *__restrict xout) {
    xout[0] = gdn_half_partial(in);   // partial0
}

// Worker 1 (tile 1): compute partial1, RECV partial0 via f_cross, drain total.
//   in[8]       = [gdec, row_offset=64, pad...]
//   xin[8]      = [partial0, pad...]   (f_cross CONS packet, produced by W0)
//   out[8]      = [total, pad...]      (shim S2MM to DDR)
extern "C" void gdn_micro_w1(const float *__restrict in,
                              const float *__restrict xin,
                              float *__restrict out) {
    float partial1 = gdn_half_partial(in);
    out[0] = xin[0] + partial1;        // total = partial0 + partial1
}