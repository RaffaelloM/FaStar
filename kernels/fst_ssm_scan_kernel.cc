// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_ssm_scan_kernel.cc — Qwen3.5-Next Gated Delta Net (GDN) decode step on AIE2P.
//
// WHY: the "SSM" layers of qwen35 are NOT Mamba2 vector selective scans.  They are
// Gated Delta Net linear attention with a MATRIX state per value head.  The first
// version of this kernel implemented `s = a*s + b*x; y = c*s` (1-D vector scan,
// 16 groups) — structurally wrong.  This is the faithful rewrite.
//
// VALIDATION: the recurrence below is the pure-delta-rule core of
// scripts/qwopus_ssm_ref.py::gdn_delta_rule, validated bit-correct (max|Δy|<1e-8)
// against transformers' torch_recurrent_gated_delta_rule (which the reference calls
// after l2norm+scale+exp — those are done UPSTREAM here, see "Division of labor").
//
// CONTRACT (per v-head, per decode token), persistent state S [HK, HV] = [128,128]
// stored ROW-MAJOR fp32 (S[i*HV + j] = S[i][j], i = key index, j = value index):
//     S    = gdec * S
//     kvm  = Sᵀ @ kn        (= sum_i S[i,:] * kn[i])      [HV]
//     delta= (v - kvm) * beta                            [HV]
//     S    = S + outer(kn, delta)                         (S[i,j] += kn[i]*delta[j])
//     y    = Sᵀ @ qn        (= sum_i S[i,:] * qn[i])      [HV]
//
// DIVISION OF LABOR (zero-CPU): the AIE scan kernel runs ONLY this pure delta rule
// — no transcendentals (the AIE scalar unit has no sqrt/exp).  The elementwise
//   qn = l2norm(q)*(1/sqrt(HK)),  kn = l2norm(k),  gdec = exp(g_logit),  beta = sigmoid(b)
// are computed UPSTREAM on the ew_unified NPU kernel (which already implements
// RMSNorm = sum-of-squares + rsqrt + scale, and SiLU = x*sigmoid(x) — so l2norm,
// exp and sigmoid are within its capability).  The projections (qkv, gate, out)
// and conv1d/SiLU are separate NPU GEMM / ew_unified ops.
//
// GEOMETRY: HK=HV=128, n_v_heads=48, n_k_heads=16 (q,k repeated 3x).  48 v-heads /
// 16 AIE cores = 3 v-heads/core, processed SEQUENTIALLY per dispatch (keeps ONE
// [128,128] fp32 state = 64 KB in tile memory).  State is fp32 (the model keeps the
// recurrent state in fp32) so this is bit-correct-capable.
//
// PACKED LAYOUT, ONE v-head, ONE token, ALL FP32 (qn,kn already normed/scaled,
// gdec already exp'd, beta already sigmoid'd by upstream ew_unified):
//   in  = [S_in (HK*HV) | qn (HK) | kn (HK) | v (HV) | gdec (1) | beta (1)]
//   out = [S_out (HK*HV) | y (HV)]
// The host keeps the fp32 state in a DDR BO and feeds S_out back as S_in next token.

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HK = 128;   // head_k_dim
constexpr int HV = 128;   // head_v_dim
constexpr int V = 8;      // AIE2 fp32 lanes per vector
constexpr int NVEC = HV / V;   // 16 vectors per row

// All running state is aie::vector<float,V>; aie::mul(...).to_vector<float>()
// materialises the transient accumulator so every aie::add is vector+vector.
extern "C" void fst_ssm_scan_step(
    const float *__restrict in,
    float *__restrict out
) {
    event0();
    const float *Sin = in;
    const float *qn = in + HK * HV;
    const float *kn = qn + HK;
    const float *v  = kn + HK;
    const float gdec = v[HV];
    const float beta = v[HV + 1];
    float *Sout = out;
    float *y    = out + HK * HV;

    // ── Pass 1: S = gdec*S ; kvm = Sᵀ @ kn ──────────────────────────────────
    aie::vector<float, V> kvm[NVEC];
    for (int n = 0; n < NVEC; ++n) kvm[n] = aie::zeros<float, V>();
    for (int i = 0; i < HK; ++i) {
        const float ki = kn[i];
        for (int n = 0; n < NVEC; ++n) {
            auto s  = aie::load_v<V>(Sin + i * HV + n * V);     // S[i, 8n:8n+8]
            auto sv = aie::mul(s, gdec).template to_vector<float>();   // gdec*S
            aie::store_v(Sout + i * HV + n * V, sv);            // decayed S
            kvm[n] = aie::add(kvm[n], aie::mul(sv, ki).template to_vector<float>()); // += S*kn[i]
        }
    }

    // ── delta = (v - kvm) * beta ───────────────────────────────────────────
    aie::vector<float, V> delta[NVEC];
    for (int n = 0; n < NVEC; ++n) {
        auto vv = aie::load_v<V>(v + n * V);
        auto dv = aie::sub(vv, kvm[n]);
        delta[n] = aie::mul(dv, beta).template to_vector<float>();
    }

    // ── Pass 2: S += outer(kn, delta) ; y = Sᵀ @ qn ─────────────────────────
    aie::vector<float, V> yacc[NVEC];
    for (int n = 0; n < NVEC; ++n) yacc[n] = aie::zeros<float, V>();
    for (int i = 0; i < HK; ++i) {
        const float ki = kn[i];
        const float qi = qn[i];
        for (int n = 0; n < NVEC; ++n) {
            auto s   = aie::load_v<V>(Sout + i * HV + n * V);
            auto su  = aie::add(s, aie::mul(delta[n], ki).template to_vector<float>()); // S + kn[i]*delta
            aie::store_v(Sout + i * HV + n * V, su);
            yacc[n] = aie::add(yacc[n], aie::mul(su, qi).template to_vector<float>());  // += S*qn[i]
        }
    }
    for (int n = 0; n < NVEC; ++n) aie::store_v(y + n * V, yacc[n]);

    event1();
}