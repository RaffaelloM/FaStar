// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_passB_delta_kernel.cc — passB with DELTA BROADCAST (Fork B / Phase 1).
//
// WHY: the shipped passB (gdn_passB_block in fst_gdn_scan_kernel.cc) packs the
// 128-float delta into EVERY input row: 48 v-heads × 128 rows × 128 fp32 =
// 3.1 MB of replicated delta streamed through the shim per SSM layer, on top of
// the 3.1 MB of S0 read and 3.1 MB of S2 written.  passB is 33.6 ms/layer = 33%
// of the token, dominated by this bidirectional shim DMA at ~280 MB/s/dir.
// The delta is IDENTICAL across all 128 rows of one v-head, so replicating it
// 128× is pure waste.
//
// FIX: hold the per-v-head delta (128 fp32 = 512 B, WELL under the 8 KB fast
// on-tile-array threshold — the ≥16 KB wall that killed chunkwise GDN and the
// held-h FFN matvec does NOT apply here) via a SECOND MM2S shim, fed ONCE per
// v-head (48 small packets), and DROP delta from the per-row packet.  This
// removes the 3.1 MB replicated delta read → passB DMA drops from 9.4 MB to
// 6.3 MB (S0 read + S2 write only).
//
// Per-row packet shrinks 264 -> 136 fp32:
//   [S0(128) | kn_i@128 | gdec@129 | pad(6)]   (128+2=130, padded to 136 for %8==0
//   row stride — aie::load_v silently rounds to the nearest 8-float-aligned
//   address, so a 130-stride would corrupt the S0 vector loads of row r>0).
//
// Math is BYTE-IDENTICAL to gdn_passB_block: s2[r][j] = gdec*S[r][j] + kn_i[r]*delta[j].
// Only delta's source changes (a held per-v-head pointer vs row+HV).  8 rows/block,
// 1024-float output packet (the proven silu drain geometry).  2 MM2S (f_in + f_delta)
// + 1 S2MM (f_out) — within the 2+2 shim budget.

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV = 128;
constexpr int V = 8;
constexpr int NVEC = HV / V;          // 16
constexpr int PKT_BD = 136;           // [S0(128)|kn_i@128|gdec@129|pad(6)]  (%8==0)
constexpr int ROWS_PER_PKT = 16;      // 16 rows/block (delta-bcast shrinks the row
                                       //  264->136 so 16×136=2176 fits the 4096-word BD
                                       //  cap; shipped 264×16=4224 > 4096 caps at 8).
                                       //  Halves the packet count 768->384.

// Process 8 rows of one v-head.  p carries [S0|kn_i|gdec|pad] per row (136 fp32);
// delta is the HELD per-v-head vector (128 fp32, acquired once per v-head by the
// IRON worker and passed in).  s2_out holds 8 updated S rows (1024 fp32).
extern "C" void gdn_passB_block_delta(
    const float *__restrict p,        // ROWS_PER_PKT × PKT_BD
    float *__restrict s2_out,         // ROWS_PER_PKT × HV
    const float *__restrict delta)    // HV (held per v-head)
{
    for (int r = 0; r < ROWS_PER_PKT; ++r) {
        const float *row   = p + r * PKT_BD;
        const float *s_row = row;                 // [0,128)  8-aligned
        const float kn_i = row[HV + 0];           // 128 (scalar, no align need)
        const float gdec = row[HV + 1];           // 129 (scalar)
        float *out = s2_out + r * HV;
        for (int n = 0; n < NVEC; ++n) {
            auto s = aie::load_v<V>(s_row + n * V);
            auto d = aie::load_v<V>(delta + n * V);
            auto s2 = aie::add(aie::mul(s, gdec).template to_vector<float>(),
                               aie::mul(d, kn_i).template to_vector<float>());
            aie::store_v(out + n * V, s2);
        }
    }
}