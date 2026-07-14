// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_8kslab_full_kernel.cc — FULL-RECURRENCE 8 KB column-slab stack probe.
//
// Fills the empty cell in the chunkwise-GDN test matrix (see
// docs/QWOPUS_GDN_8KB_SLAB_ANALYSIS.md): is ≤8 KB stack RMW FAST under the REAL
// GDN 3-pass recurrence (K=8 × 48 v-heads), or does it degrade to the ≥16 KB
// ~1.3µs/vec rate?
//   * The prior ≤8 KB "fast" result (4-tile 8 KB micro, 5.87 ms) was a TOY
//     one-step decay-mul, NOT the real recurrence.
//   * Every prior FULL-recurrence test used ≥16 KB stack (2-tile 16 KB,
//     22 ms/v-head, slow) or a persistent Buffer (8-tile 4 KB, slow at any size).
//   * The ≤8 KB-stack × full-recurrence cell was NEVER measured.  This fills it.
//
// Scaffold = the proven 8kstack single-call internal-loop (tiny I/O, S on one 8 KB
// stack frame entered ONCE for all 48 v-heads so the SP does not restore between
// v-heads — avoids the repeated-large-frame-call breakage).  Body = the REAL GDN
// 3-pass recurrence (reused verbatim from the bit-correct 8-tile / proven 3-pass
// kernel — same idiom as kernels/fst_gdn_chunkwise_2tile_kernel.cc::recur_core):
//   passA: a[c] = Σ_i S[i,c]·kn[i] ; b[c] = Σ_i S[i,c]·qn[i]   (row stream)
//   delta: delta[c] = (v[c] − gdec·a[c])·beta ; c_scalar = kn·qn ; y[c] = gdec·b[c] + delta[c]·c_scalar
//   passB: S[i,c] = gdec·S[i,c] + kn[i]·delta[c]   (RMW the 8 KB stack S in place, store bf16)
// run K=8 sequential tokens (delta_t depends on S_t — the real M=K recurrence).
//
// PROBE INPUT SOURCING (transparent, NOT a toy recurrence): kn/qn/v/gdec/beta and
// the initial S0 are DETERMINISTIC in-kernel values (functions of v-head + token +
// index), so I/O stays tiny and DMA does not confound the stack-RMW latency.  The
// recurrence STRUCTURE is the real GDN 3-pass; only the input VALUES are
// synthesized.  The stack-access PATTERN (and thus the measured latency) is
// identical to a real-data engine kernel.  Output = a per-v-head y-checksum
// (prevents DCE).
//
// Geometry: 1 tile, 32-col slab, S ROW-MAJOR [128 rows][32 cols] bf16 = 8 KB
// stack (HV*COLS = 128*32 = 4096 bf16).  passA/passB stream the 128 rows (each
// row = 32 bf16 = 4 vecs of 8); a[c] accumulates across the row loop = the column
// reduction.  4 such independent slabs cover the full 128 cols (the engine
// config); 1 slab is the latency unit, so 1 tile is the decisive probe.  bf16-S
// (FST_Q35_GDN_BF16S, A/B-lossless for argmax) is the favorable engine precision.

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV   = 128;
constexpr int COLS = 32;           // slab width (8 KB = 128*32*2)
constexpr int KST  = 8;            // K tokens
constexpr int NVH  = 48;           // v-heads
constexpr int V    = 8;            // bf16 vec lanes
constexpr int NCV  = COLS / V;      // 4 vecs per row-slice
constexpr int NVR  = HV / V;        // 16 vecs per full vector (kn/qn)

// deterministic probe inputs (synthesized, not streamed — see header comment)
static inline float fv_kn  (int i, int v, int t) { return (float)(((i*7  + v*13 + t*5 ) % 17)) / 16.0f - 0.5f; }
static inline float fv_qn  (int i, int v, int t) { return (float)(((i*11 + v*3  + t*9 ) % 13)) / 12.0f - 0.5f; }
static inline float fv_v   (int j, int v, int t) { return (float)(((j*5  + v*17 + t*7 ) % 11)) / 10.0f - 0.5f; }
static inline float fv_gdec(int v, int t)        { return 0.5f + 0.01f * (float)((v + t) % 5); }
static inline float fv_beta(int v, int t)        { return 0.7f; }
static inline float fv_s0  (int i, int j, int v)  { return (float)(((i + j*3 + v) % 19)) / 18.0f - 0.5f; }

// One token's real GDN recurrence on the 8 KB row-major stack S.  kn/qn/v are fp32
// stack scratch filled from the deterministic probe inputs (so the bf16 S is the
// ONLY large on-tile array — the thing being measured).
static inline float recur_token(bfloat16 *__restrict S, int v, int t) {
    const float gdec = fv_gdec(v, t);
    const float beta = fv_beta(v, t);
    const aie::vector<bfloat16, V> ones = aie::broadcast<bfloat16, V>((bfloat16)1.0f);

    float kn[HV]    __attribute__((aligned(32)));
    float qn[HV]    __attribute__((aligned(32)));
    float a[COLS]   __attribute__((aligned(32)));
    float b[COLS]   __attribute__((aligned(32)));
    float delta[COLS] __attribute__((aligned(32)));
    float vv[COLS]  __attribute__((aligned(32)));
    for (int i = 0; i < HV; ++i) { kn[i] = fv_kn(i, v, t); qn[i] = fv_qn(i, v, t); }
    for (int j = 0; j < COLS; ++j) vv[j] = fv_v(j, v, t);

    const auto z = aie::zeros<float, V>();
    for (int n = 0; n < NCV; ++n) { aie::store_v(a + n*V, z); aie::store_v(b + n*V, z); }

    // passA: a[c] += S[i,c]*kn[i] ; b[c] += S[i,c]*qn[i]  (stream 128 rows)
    for (int i = 0; i < HV; ++i) {
        const float kn_i = kn[i];
        const float qn_i = qn[i];
        const bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv =
                aie::mul(aie::load_v<V>(srow + n*V), ones).template to_vector<float>();
            aie::vector<float, V> av = aie::load_v<V>(a + n*V);
            aie::vector<float, V> bv = aie::load_v<V>(b + n*V);
            aie::store_v(a + n*V, aie::add(av, aie::mul(sv, kn_i).template to_vector<float>()));
            aie::store_v(b + n*V, aie::add(bv, aie::mul(sv, qn_i).template to_vector<float>()));
        }
    }
    // c_scalar = kn · qn
    float c = 0.0f;
    for (int n = 0; n < NVR; ++n)
        c += aie::reduce_add(
            aie::mul(aie::load_v<V>(kn + n*V), aie::load_v<V>(qn + n*V)).template to_vector<float>());
    // delta + y
    float y_sum = 0.0f;
    for (int n = 0; n < NCV; ++n) {
        aie::vector<float, V> av = aie::load_v<V>(a + n*V);
        aie::vector<float, V> bv = aie::load_v<V>(b + n*V);
        aie::vector<float, V> vvi = aie::load_v<V>(vv + n*V);
        aie::vector<float, V> kvm = aie::mul(av, gdec).template to_vector<float>();
        aie::vector<float, V> dv  = aie::mul(aie::sub(vvi, kvm), beta).template to_vector<float>();
        aie::store_v(delta + n*V, dv);
        aie::accum<accfloat, V> yv = aie::add(aie::mul(bv, gdec), aie::mul(dv, c));
        y_sum += aie::reduce_add(yv.template to_vector<float>());
    }
    // passB: S[i,c] = gdec*S[i,c] + kn[i]*delta[c]  (RMW stack S, store bf16)
    for (int i = 0; i < HV; ++i) {
        const float kn_i = kn[i];
        bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv =
                aie::mul(aie::load_v<V>(srow + n*V), ones).template to_vector<float>();
            aie::vector<float, V> dv = aie::load_v<V>(delta + n*V);
            aie::accum<accfloat, V> acc = aie::add(aie::mul(sv, gdec), aie::mul(dv, kn_i));
            aie::store_v(srow + n*V, acc.template to_vector<bfloat16>());
        }
    }
    return y_sum;
}

// FULL recurrence: 1 tile, single call, internal loop over 48 v-heads × K=8 tokens.
extern "C" void gdn_8kslab_full(const float *__restrict in, float *__restrict out) {
    (void)in;
    bfloat16 S[HV * COLS] __attribute__((aligned(32)));   // 8 KB stack — ONE frame for all 48
    for (int v = 0; v < NVH; ++v) {
        for (int i = 0; i < HV; ++i)                     // init S0 (deterministic)
            for (int j = 0; j < COLS; ++j)
                S[(size_t)i * COLS + j] = (bfloat16)fv_s0(i, j, v);
        float y_acc = 0.0f;
        for (int t = 0; t < KST; ++t)                     // 8 sequential recurrence steps
            y_acc += recur_token(S, v, t);
        out[v] = y_acc;                                  // checksum (prevents DCE)
    }
}

// NOOP: same 8 KB stack alloc + tiny DMA, NO recurrence (one fill + one drain per
// v-head).  Isolates the stack-alloc + DMA + loop floor so the probe can subtract
// it for a clean per-vec stack-RMW latency.
extern "C" void gdn_8kslab_noop(const float *__restrict in, float *__restrict out) {
    (void)in;
    bfloat16 S[HV * COLS] __attribute__((aligned(32)));
    const aie::vector<bfloat16, V> z = aie::broadcast<bfloat16, V>((bfloat16)0.0f);
    for (int v = 0; v < NVH; ++v) {
        for (int i = 0; i < HV; ++i)                     // fill once (stack write)
            for (int n = 0; n < NCV; ++n) aie::store_v(S + (size_t)i*COLS + n*V, z);
        float s = 0.0f;
        for (int i = 0; i < HV; ++i)                     // drain once (stack read)
            for (int n = 0; n < NCV; ++n)
                s += aie::reduce_add(
                    aie::mul(aie::load_v<V>(S + (size_t)i*COLS + n*V), z).template to_vector<float>());
        out[v] = s;
    }
}