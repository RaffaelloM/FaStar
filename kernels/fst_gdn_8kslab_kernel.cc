// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_8kslab_kernel.cc — Qwen3.5-Next Gated Delta Net (GDN) CHUNKWISE M=K
// scan, 4-TILE COLUMN-SPLIT + SINGLE-CALL 8 KB-STACK-S design.
//
// This is the engine kernel authorized after the full-recurrence 8 KB-stack probe
// (kernels/fst_gdn_8kslab_full_kernel.cc) measured 202.77 ns/vec — FAST, the ≤8 KB
// stack does NOT degrade to the ≥16 KB ~1.3 µs/vec regime under the real K=8
// recurrence.  See docs/QWOPUS_GDN_8KB_SLAB_ANALYSIS.md +
// docs/QWOPUS_GDN_8KSLAB_FULL_PROBE_POSTMORTEM.md.
//
// DESIGN — 4 INDEPENDENT tiles (no chain), each holding one 32-col bf16-S slab as
// an 8 KB STACK-LOCAL array (the probe's proven-fast access pattern, reused
// verbatim).  Column-split ⇒ recurrence INDEPENDENT per tile ⇒ NO cross-tile
// reduction, NO chain, NO forwarding, NO persistent aie.iron.Buffer (the slow
// disease).  bf16-S (FST_Q35_GDN_BF16S) is A/B-lossless for argmax on the shipped
// path and halves S volume so a 32-col slab = 8 KB (the fast threshold).
//
// SHIM DISCIPLINE per tile = exactly 2 MM2S + 2 S2MM (per the authorization plan):
//   MM2S-1  s0    : S0_slab  [HV*COLS] bf16 = 8 KB         (initial state, in)
//   MM2S-2  par   : K tokens × [kn|qn|v_slab|gdec|beta|c] bf16  (params, in)
//   S2MM-1  snew  : S_new_slab [HV*COLS] bf16 = 8 KB      (final state, out)
//   S2MM-2  yd    : K tokens × [y_slab|delta_slab] bf16    (outputs, out)
// Each fifo is 48 elements (one per v-head); the worker acquires s0+par, calls
// gdn_8kslab_vhead ONCE (stack-S lives for the call: 128-row load → K recur →
// 128-row store), releases snew+yd.  c = kn·qn is computed HOST-SIDE and packed in
// par (no scalar-tile, no on-tile dot).
//
// MATH — reused verbatim from the proven 2-tile / 3-pass / 8-tile kernel
// (kernels/fst_gdn_chunkwise_2tile_kernel.cc::recur_core).  bf16 S, fp32 compute,
// bf16 round at every passB store.  kn/qn/v/gdec/beta/c are bf16 on DMA, converted
// to fp32 on read.  All hot-loop ops are 8-lane vec load/store/mul/add.
//   passA : a[c] = Σ_i S[i,c]·kn[i] ; b[c] = Σ_i S[i,c]·qn[i]
//   delta : delta[c] = (v[c] − gdec·a[c])·beta ; y[c] = gdec·b[c] + delta[c]·c
//   passB : S[i,c] = gdec·S[i,c] + kn[i]·delta[c]   (RMW the 8 KB stack S in place)

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV   = 128;
constexpr int COLS = 32;           // columns per tile (8 KB slab = 128×32×2)
constexpr int NCV  = COLS / 8;     // 4 vectors of 8 per row-slice
constexpr int K    = 8;            // chunkwise tokens per dispatch
constexpr int V    = 8;

// par packet (bf16), per token: [kn(128) | qn(128) | v_slab(32) | gdec | beta | c | 5-pad]
// PAR_SZ is padded to a multiple of V=8 (296, not the natural 291) so that
// par + t*PAR_SZ is always 16-byte (8-bf16) aligned — aie::load_v rounds to the
// nearest 8-element-aligned address, so an unaligned token stride read garbage
// for tk>0 (the root cause of the original slab blow-up).  5 trailing pad bf16.
constexpr int PAR_SZ    = 2 * HV + COLS + 8;   // 128+128+32+8 = 296  (mult of 8)
constexpr int PAR_V     = 2 * HV;             // 256
constexpr int PAR_GDEC  = 2 * HV + COLS;       // 288
constexpr int PAR_BETA  = 2 * HV + COLS + 1;   // 289
constexpr int PAR_C     = 2 * HV + COLS + 2;  // 290
// yd packet (bf16), per token: [y_slab(32) | delta_slab(32)]
constexpr int YD_DELTA  = COLS;               // 32
constexpr int YD_SZ     = 2 * COLS;           // 64

static inline void vcopy_bf16(const bfloat16 *__restrict in,
                              bfloat16 *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n)
        aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}

// One token's recurrence on the tile's 32 cols.  par points at the token's bf16
// [kn|qn|v_slab|gdec|beta|c|pad]; yd points at [y_slab(32)|delta_slab(32)].
//
// INPUT LAYOUT = the synthetic probe's PROVEN-FAST pattern (fst_gdn_8kslab_full_
// kernel.cc::recur_token, 79 ms): a ONE-TIME SEQUENTIAL copy of the DMA `par`
// buffer into SEPARATE aligned fp32 stack arrays (kn[128], qn[128], vv[32]), then
// the hot passA/delta/passB loops read inputs ONLY from the stack — NO strided
// access to the `par` ObjectFifo DMA buffer inside the inner loop.  The earlier
// packed parf[296] (1182 ms) and the register-streamed noparf (1166 ms) both
// touched the par DMA buffer in the hot loop and were ~13× slower than the
// probe; this variant isolates whether the separate-stack / no-hot-loop-par
// layout (the probe's actual fast pattern) recovers the speed.  Total stack =
// S(8KB) + kn+qn+vv+a+b+delta (1.5KB) = 9.7KB, identical to the probe's frame.
static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict yd) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    constexpr int NG = HV / V;                       // 16 groups of 8 for kn/qn

    // SEPARATE aligned fp32 stack arrays (probe layout) — one-time sequential fill.
    float kn[HV]     __attribute__((aligned(32)));   // 512 B
    float qn[HV]     __attribute__((aligned(32)));   // 512 B
    float vv[COLS]   __attribute__((aligned(32)));   // 128 B
    float a[COLS]    __attribute__((aligned(32)));
    float b[COLS]    __attribute__((aligned(32)));
    float delta[COLS] __attribute__((aligned(32)));
    for (int n = 0; n < NG; ++n)                    // kn: par[0..127] bf16 -> fp32 stack
        aie::store_v(kn + n * V, aie::mul(aie::load_v<V>(par + n * V), ones_bf).template to_vector<float>());
    for (int n = 0; n < NG; ++n)                    // qn: par[128..255] bf16 -> fp32 stack
        aie::store_v(qn + n * V, aie::mul(aie::load_v<V>(par + HV + n * V), ones_bf).template to_vector<float>());
    for (int n = 0; n < NCV; ++n)                   // v: par[256..287] bf16 -> fp32 stack
        aie::store_v(vv + n * V, aie::mul(aie::load_v<V>(par + PAR_V + n * V), ones_bf).template to_vector<float>());
    aie::vector<float, V> tailf =                   // gdec/beta/c from par+288 (aligned)
        aie::mul(aie::load_v<V>(par + PAR_GDEC), ones_bf).template to_vector<float>();
    const float gdec = tailf[0];
    const float beta = tailf[1];
    const float c    = tailf[2];

    const auto z = aie::zeros<float, V>();
    for (int n = 0; n < NCV; ++n) { aie::store_v(a + n * V, z); aie::store_v(b + n * V, z); }

    for (int i = 0; i < HV; ++i) {            // passA: a[c]+=S[i,c]*kn[i]; b[c]+=S[i,c]*qn[i] (stack kn/qn)
        const float kn_i = kn[i];
        const float qn_i = qn[i];
        const bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> av = aie::load_v<V>(a + n * V);
            aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
            aie::store_v(a + n * V, aie::add(av, aie::mul(sv_f, kn_i).template to_vector<float>()));
            aie::store_v(b + n * V, aie::add(bv, aie::mul(sv_f, qn_i).template to_vector<float>()));
        }
    }
    for (int n = 0; n < NCV; ++n) {           // delta: kvm=gdec*a; delta=(v-kvm)*beta; y=gdec*b+delta*c (stack vv)
        aie::vector<float, V> av = aie::load_v<V>(a + n * V);
        aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
        aie::vector<float, V> vvi = aie::load_v<V>(vv + n * V);
        aie::vector<float, V> kvm = aie::mul(av, gdec).template to_vector<float>();
        aie::accum<accfloat, V> dv_acc = aie::mul(aie::sub(vvi, kvm), beta);
        aie::vector<float, V> dv = dv_acc.template to_vector<float>();
        aie::store_v(delta + n * V, dv);                              // fp32 scratch for passB
        aie::accum<accfloat, V> yv = aie::add(aie::mul(bv, gdec), aie::mul(dv, c));
        aie::store_v(yd + n * V, yv.template to_vector<bfloat16>());
        aie::store_v(yd + YD_DELTA + n * V, dv_acc.template to_vector<bfloat16>());
    }
    for (int i = 0; i < HV; ++i) {            // passB: S[i,c]=gdec*S[i,c]+kn[i]*delta[c] (stack kn, delta)
        const float kn_i = kn[i];
        bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> dv = aie::load_v<V>(delta + n * V);
            aie::accum<accfloat, V> acc = aie::add(aie::mul(sv_f, gdec), aie::mul(dv, kn_i));
            aie::store_v(srow + n * V, acc.template to_vector<bfloat16>());
        }
    }
}

// ONE v-head: load S0 (bf16 -> 8 KB bf16 stack-S) -> K recur (stack-S RMW) ->
// store S_new (bf16 stack-S -> bf16) + per-token y|delta.  Stack-S is the 8 KB
// stack-local array — the probe's proven-fast path (202.77 ns/vec).
extern "C" void gdn_8kslab_vhead(const bfloat16 *__restrict s0,  const bfloat16 *__restrict par,
                                 bfloat16 *__restrict snew,      bfloat16 *__restrict yd) {
    bfloat16 S[HV * COLS] __attribute__((aligned(32)));   // 8 KB stack-local state

    for (int i = 0; i < HV; ++i)               // load S0: 128 rows × 32 cols bf16 -> stack-S
        vcopy_bf16(s0 + (size_t)i * COLS, S + (size_t)i * COLS, NCV);

    for (int t = 0; t < K; ++t)                // K chunkwise recurrence steps (stack-S RMW)
        recur_core(S, par + (size_t)t * PAR_SZ, yd + (size_t)t * YD_SZ);

    for (int i = 0; i < HV; ++i)               // store S_new: stack-S -> snew
        vcopy_bf16(S + (size_t)i * COLS, snew + (size_t)i * COLS, NCV);
}