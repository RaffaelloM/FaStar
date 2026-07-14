// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_3pass_kernel.cc — Qwen3.5-Next GDN scan, ONE-XCLBIN 3-PASS-SEQUENTIAL
// design (the "no-state / dispatch-amortization without on-tile RMW" path).
//
// THE GOAL: cut the shipped 3-xclbin 3-dispatch/SSM-layer GDN (passA/delta/passB
// each its own xrt::run, 144 dispatches/token) to ONE xrt::run/SSM-layer (48
// dispatches/token) ≈ ~3x, by running all 3 passes inside ONE worker in ONE
// xclbin.  This sidesteps BOTH proven dead-ends:
//   - the M=1 fused kernel (fst_gdn_fused) holds S ON-CHIP across passes →
//     aie.iron.Buffer RMW 600x slow (docs/QWOPUS_FUSED_GDN_RACE_ROOTCAUSE.md);
//   - the chunkwise M=K on-tile-state kernel holds S on-tile across K tokens →
//     ~340x slow at >=16KB in every storage class (the 2-tile/8-tile dead-ends).
// HERE S is NEVER held on-tile: it is streamed ONE ROW at a time from DDR
// (read twice — passA then passB — exactly as the shipped 3-pass already does),
// and discarded.  Only the SMALL on-tile state stays: ab[a|b]=1KB, delta=512B,
// par=1.5KB ≈ 3KB total — well under the ~8KB fast-stack threshold.  No
// aie.iron.Buffer; no >=16KB on-tile array.
//
// MATH: byte-identical to the shipped 3-pass (fst_gdn_scan_kernel.cc).
//   passA row:  a[c]+=S[i,c]*kn[i]; b[c]+=S[i,c]*qn[i]   (accumulate into held ab)
//   delta:      c=kn·qn; kvm=gdec*a; delta=(v-kvm)*beta; y=gdec*b+delta*c
//   passB row:  S2[i,c]=gdec*S[i,c]+kn[i]*delta[c]
// gdn_3p_passA_row is gdn_passA_row verbatim.  gdn_3p_delta is gdn_delta_full
// with a/b read from the held ab fifo and v/kn/qn/gdec/beta from the par fifo
// (instead of all packed in one 642 packet).  gdn_3p_passB_block is
// gdn_passB_block with delta read from the held delta fifo and gdec from par
// (instead of packed in the 264 row).  The arithmetic is identical.
//
// FIFOS (one worker; 2 MM2S + 2 S2MM shim = within the 2+2 budget; ab/delta
// are CORE-LOCAL held ObjectFifos, not shim streams):
//   f_in_S  (shim MM2S, 130/pkt, 256 pkts/v-head): [S_row(128)|kn_i|qn_i]
//           — first 128 rows feed passA, next 128 feed passB (qn_i ignored there)
//   f_in_par(shim MM2S, 386/pkt, 1/v-head): [v(128)|kn(128)|qn(128)|gdec|beta]
//   f_y     (shim S2MM, 128/pkt, 1/v-head): y
//   f_out   (shim S2MM, 1024/pkt, 16/v-head): S2 rows (8 rows/pkt, proven silu drain)
//   f_ab    (core-local held, 256): [a(128)|b(128)] — acquired once/v-head, RMW'd
//   f_delta (core-local held, 128): delta — written by delta, read by passB

#include <aie_api/aie.hpp>

constexpr int HV   = 128;
constexpr int V    = 8;
constexpr int NVEC = HV / V;          // 16

// ── zero the held ab [a|b] = 256 fp32 in place.
extern "C" void gdn_3p_zero(float *__restrict ab) {
    auto z = aie::zeros<float, V>();
    for (int n = 0; n < 2 * NVEC; ++n) aie::store_v(ab + n * V, z);
}

// ── passA one 8-row block.  s_in = 8 rows × 130 = [S_row(128)|kn_i|qn_i] ×8;
//    ab = [a(128)|b(128)] (held).  VERBATIM gdn_passA_row math, looped over 8 rows
//    (the unified f_in_S delivers 8-row blocks for both passA and passB phases).
extern "C" void gdn_3p_passA_block(
    const float *__restrict s_in,
    float *__restrict ab
) {
    constexpr int ROWS_PER_PKT = 8;
    constexpr int SIN_STRIDE = 130;     // [S_row(128)|kn_i(1)|qn_i(1)]
    float *a_acc = ab;
    float *b_acc = ab + HV;
    for (int r = 0; r < ROWS_PER_PKT; ++r) {
        const float *s_row = s_in + (size_t)r * SIN_STRIDE;
        const float kn_i  = s_in[(size_t)r * SIN_STRIDE + HV + 0];
        const float qn_i  = s_in[(size_t)r * SIN_STRIDE + HV + 1];
        for (int n = 0; n < NVEC; ++n) {
            auto s = aie::load_v<V>(s_row + n * V);
            auto a = aie::load_v<V>(a_acc + n * V);
            aie::store_v(a_acc + n * V, aie::add(a, aie::mul(s, kn_i).template to_vector<float>()));
            auto b = aie::load_v<V>(b_acc + n * V);
            aie::store_v(b_acc + n * V, aie::add(b, aie::mul(s, qn_i).template to_vector<float>()));
        }
    }
}

// ── delta.  ab=[a(128)|b(128)] (held); par=[v(128)|kn(128)|qn(128)|gdec|beta]=386;
//    delta_out=[delta(128)] (held, read by passB); y_out=[y(128)] (drained to f_y).
//    VERBATIM gdn_delta_full math, with a/b from ab and v/kn/qn/gdec/beta from par.
extern "C" void gdn_3p_delta(
    const float *__restrict ab,
    const float *__restrict par,
    float *__restrict delta_out,
    float *__restrict y_out
) {
    const float *a    = ab;
    const float *b    = ab + HV;
    const float *v    = par;
    const float *kn   = par + HV;
    const float *qn   = par + 2 * HV;
    const float gdec  = par[3 * HV + 0];
    const float beta  = par[3 * HV + 1];

    float c = 0.0f;
    for (int n = 0; n < NVEC; ++n) {
        auto kk = aie::load_v<V>(kn + n * V);
        auto qq = aie::load_v<V>(qn + n * V);
        c += aie::reduce_add(aie::mul(kk, qq).template to_vector<float>());
    }
    for (int n = 0; n < NVEC; ++n) {
        auto av = aie::load_v<V>(a + n * V);
        auto bv = aie::load_v<V>(b + n * V);
        auto vv = aie::load_v<V>(v + n * V);
        auto kvm = aie::mul(av, gdec).template to_vector<float>();
        auto dv  = aie::mul(aie::sub(vv, kvm), beta).template to_vector<float>();
        aie::store_v(delta_out + n * V, dv);
        auto yv = aie::add(aie::mul(bv, gdec).template to_vector<float>(),
                          aie::mul(dv, c).template to_vector<float>());
        aie::store_v(y_out + n * V, yv);
    }
}

// ── passB one 8-row block.  s_in = 8 rows × 130 = [S_row(128)|kn_i(1)|qn_i(1)]
//    (qn_i ignored); delta = held [128] (same for all 8 rows); gdec from par[3*HV];
//    s2_out = 8 rows × 128 (1024).  VERBATIM gdn_passB_block math, delta+gdec
//    sourced from held fifos instead of packed in the 264 row.
extern "C" void gdn_3p_passB_block(
    const float *__restrict s_in,
    const float *__restrict delta,
    const float *__restrict par,
    float *__restrict s2_out
) {
    constexpr int ROWS_PER_PKT = 8;
    constexpr int SIN_STRIDE = 130;     // [S_row(128)|kn_i(1)|qn_i(1)]
    const float gdec = par[3 * HV + 0];
    for (int r = 0; r < ROWS_PER_PKT; ++r) {
        const float *s_row = s_in + (size_t)r * SIN_STRIDE;
        const float kn_i   = s_in[(size_t)r * SIN_STRIDE + HV];
        float *out = s2_out + (size_t)r * HV;
        for (int n = 0; n < NVEC; ++n) {
            auto s = aie::load_v<V>(s_row + n * V);
            auto d = aie::load_v<V>(delta + n * V);
            auto s2 = aie::add(aie::mul(s, gdec).template to_vector<float>(),
                               aie::mul(d, kn_i).template to_vector<float>());
            aie::store_v(out + n * V, s2);
        }
    }
}