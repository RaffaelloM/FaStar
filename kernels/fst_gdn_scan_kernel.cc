// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_scan_kernel.cc — Qwen3.5-Next Gated Delta Net (GDN) scan, ROW-STREAMING.
//
// WHY ROW-STREAMING: the GDN state S[128,128] is 64 KB fp32 — too big for a 64 KB
// AIE2P tile as a single ObjectFifo packet.  So the state lives in a DDR BO and is
// streamed ONE ROW (128 fp32 = 512 B) at a time through small depth-2 objectfifos
// (= aie::dmabuf<2> double-buffering).  The small accumulators (a, b, delta, y) are
// held as tiny tile packets (512 B each) acquired once and updated in place across
// the 128-row loop — the same held-accumulator pattern MLA uses for its C matrix.
//
// The scan is split across 3 xclbins (chained into ONE run_blob dispatch = 1 xrt::run):
//
//   passA (per row i): a += S[i]*kn[i] ; b += S[i]*qn[i]            # a=Sᵀ@kn, b=Sᵀ@qn
//   delta  (once):     kvm = gdec*a ; delta=(v-kvm)*beta ; c=kn·qn ; y = gdec*b + delta*c
//   passB (per row i): S2[i] = gdec*S[i] + kn[i]*delta              # writes updated state
//
// Derived from the validated recurrence (scripts/qwopus_ssm_ref.py, bit-correct vs
// transformers): S=gdec*S ; kvm=Sᵀ@kn ; delta=(v-kvm)*beta ; S+=outer(kn,delta) ;
// y=Sᵀ@qn.  Splitting kvm=Sᵀ@kn and y=Sᵀ@qn into passA (over raw S0) and folding the
// decay into delta/passB avoids any 64 KB intermediate buffer — S0 is streamed twice
// straight from DDR.  y = gdec*b + delta*c follows from S2ᵀ@qn = gdec·(S0ᵀ@qn) +
// delta·(knᵀ@qn) = gdec·b + delta·c.
//
// qn=l2norm(q)*(1/sqrt(128)), kn=l2norm(k), gdec=exp(g_logit), beta=sigmoid(b) are
// computed UPSTREAM on ew_unified (the AIE scalar unit has no sqrt/exp).  This kernel
// receives qn, kn, gdec, beta ready.  All fp32 (recurrent_state is fp32 in the model
// -> bit-correct-capable).  VEC = 8 fp32 lanes; 128/8 = 16 vectors per row.

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV = 128;
constexpr int V = 8;
constexpr int NVEC = HV / V;   // 16

// ── passA: one row.  Packed input p = [S_row(128) | kn_i(1) | qn_i(1)] = 130 fp32
//    (kn_i/qn_i folded into the row packet to keep the core's DMA fanin ≤ 2 in / 2 out).
//    ab is a [256] HELD accumulator laid out [a(128)|b(128)] (acquired once, RMW in
//    place) — ONE held output fifo, mirroring the proven FFN GEMM C-matrix pattern
//    (two simultaneous held outputs broke the shim drain flush on this IRON build).
extern "C" void gdn_passA_row(
    const float *__restrict p,
    float *__restrict ab
) {
    const float *s_row = p;
    const float kn_i  = p[HV + 0];
    const float qn_i  = p[HV + 1];
    float *a_acc = ab;
    float *b_acc = ab + HV;
    for (int n = 0; n < NVEC; ++n) {
        auto s = aie::load_v<V>(s_row + n * V);
        auto a = aie::load_v<V>(a_acc + n * V);
        aie::store_v(a_acc + n * V, aie::add(a, aie::mul(s, kn_i).template to_vector<float>()));
        auto b = aie::load_v<V>(b_acc + n * V);
        aie::store_v(b_acc + n * V, aie::add(b, aie::mul(s, qn_i).template to_vector<float>()));
    }
}

// ── delta: once.  Packed input d = [a(128)|b(128)|v(128)|kn(128)|qn(128)|gdec(1)|beta(1)]
//    = 642 fp32 (single small packet, no streaming).  Output o = [delta(128)|y(128)] = 256.
//    kvm = gdec*a ; delta = (v-kvm)*beta ; c = kn·qn ; y = gdec*b + delta*c.
extern "C" void gdn_delta_full(
    const float *__restrict d,
    float *__restrict o
) {
    const float *a    = d;
    const float *b    = d + 128;
    const float *v    = d + 256;
    const float *kn   = d + 384;
    const float *qn   = d + 512;
    const float gdec  = d[640];
    const float beta  = d[641];

    // c = kn · qn  (scalar dot over 16 vectors; reduce_add sums a vector's lanes,
    // accumulate the 16 per-vector partials into the stack scalar c).
    float c = 0.0f;
    for (int n = 0; n < NVEC; ++n) {
        auto kk = aie::load_v<V>(kn + n * V);
        auto qq = aie::load_v<V>(qn + n * V);
        c += aie::reduce_add(aie::mul(kk, qq).template to_vector<float>());
    }

    float *delta = o;
    float *y     = o + 128;
    for (int n = 0; n < NVEC; ++n) {
        auto av = aie::load_v<V>(a + n * V);
        auto bv = aie::load_v<V>(b + n * V);
        auto vv = aie::load_v<V>(v + n * V);
        auto kvm = aie::mul(av, gdec).template to_vector<float>();      // gdec*a
        auto dv  = aie::mul(aie::sub(vv, kvm), beta).template to_vector<float>(); // (v-kvm)*beta
        aie::store_v(delta + n * V, dv);
        auto yv = aie::add(aie::mul(bv, gdec).template to_vector<float>(),
                          aie::mul(dv, c).template to_vector<float>()); // gdec*b + delta*c
        aie::store_v(y + n * V, yv);
    }
}

// ── passB: a BLOCK of ROWS_PER_PKT rows.  Row layout (264 fp32, 8-aligned fields):
//      [S_row(128) | delta(128) | kn_i(1) | gdec(1) | pad(5)]
//    delta at offset 128 (mult of 8) and row stride 264 (mult of 8) are REQUIRED:
//    aie::load_v<V> silently reads from the nearest 8-float-aligned address, so any
//    misaligned load offset/stride corrupts the data (the original 258-wide packet put
//    delta at 130 and row stride 258, both %8 != 0).  kn_i/gdec are scalar reads (no
//    alignment need).  One 1024-float output packet holds 8 rows (silu-proven drain).
//    s2[r][j] = gdec*S_row[r][j] + kn_i[r]*delta[j].
extern "C" void gdn_passB_block(
    const float *__restrict p,
    float *__restrict s2_out
) {
    constexpr int ROWS_PER_PKT = 8;
    constexpr int PKT_B = 264;                   // [S0(128)|delta(128)|kn_i(1)|gdec(1)|pad(5)]
    for (int r = 0; r < ROWS_PER_PKT; ++r) {
        const float *row   = p + r * PKT_B;
        const float *s_row = row;                 // [0,128)  8-aligned
        const float *delta = row + HV;            // [128,256) 8-aligned
        const float kn_i = row[2 * HV + 0];       // 256
        const float gdec = row[2 * HV + 1];       // 257
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

// ── zero a [256] accumulator (a|b) in place before the row loop.
extern "C" void gdn_zero(float *__restrict acc) {
    auto z = aie::zeros<float, V>();
    for (int n = 0; n < 2 * NVEC; ++n) aie::store_v(acc + n * V, z);
}