// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_chunkwise_kernel.cc — Qwen3.5-Next Gated Delta Net (GDN) CHUNKWISE
// M=K scan, ONE-TILE persistent-Buffer design (Stage 1.2).
//
// GOAL: collapse the shipped 3-pass-per-token GDN (3 dispatches/SSM-layer × 48
// layers = 144 dispatches/token) into ONE dispatch/SSM-layer that processes
// K=8 tokens for all 48 v-heads, holding the recurrence state S[128,128]
// ON-TILE so it never round-trips to DDR between tokens.  This is the
// prerequisite for lossless K× speculative decoding (Stage 2).
//
// WHY ONE TILE + persistent aie.iron.Buffer (the Stage-1.1 breakthrough):
//  - The shipped 3-pass streams S[128,128] (64 KB fp32) from DDR one row at a
//    time, 3×/token, because 64 KB does not fit a 64 KB tile as one packet.
//  - Holding S on-tile as a STACK array hit the AIE2P load/store immediate
//    limit [-32768,-64] at 32 KB (Peano emits no register+register fallback),
//    and the 4-tile 8 KB split worked but needed a cross-tile reduction.
//  - A persistent aie.iron.Buffer is REGISTER-ADDRESSED (the acquired pointer
//    is the base, NOT the frame pointer) → escapes the stack immediate limit →
//    a 32 KB bf16 S[128,128] fits ONE tile.  It is PERSISTENT across C-fn
//    re-calls (unlike a stack array whose frame is freed on return) and FAST
//    (C-kernel pointer + load_v/store_v = plain tile memory, 0.10 ms/call —
//    NOT the 600× slow Python-level Buffer RMW that killed M=1 fusion).
//
// STRUCTURE (1 tile, 1 worker, 4 row-sized depth-2 ObjectFifos + 2 Buffers):
//   S   = persistent Buffer(bfloat16[128*128])     — the recurrence state
//   ctr = persistent Buffer(float[2])              — row index for load/store
//   f_s0  (MM2S) : 128 fp32 rows × 48 v-heads  — initial state S0 per v-head
//   f_par (MM2S) : K×48 param packets [qn|kn|v|gdec|beta|pad] (392 fp32)
//   f_y   (S2MM) : K×48 y packets (128 fp32)    — per-token output
//   f_sf  (S2MM) : 128 fp32 rows × 48 v-heads  — final state S_final
//
//   Worker core (48 v-heads):
//     ctr=0; for i in 128: acquire(f_s0); gdn_load_s0_row(S,ctr,row); release   // S0 -> S (bf16)
//     for t in 8:  acquire(f_par); acquire(f_y); gdn_chunkwise_recur(S,par,y); release  // RMW S
//     ctr=0; for i in 128: acquire(f_sf); gdn_store_sf_row(S,ctr,row); release  // S (bf16) -> S_final
//
//   The recurrence C fn has a SMALL frame (S is a passed Buffer pointer, not a
//   stack array; only a/b/delta [1.5 KB] are stack) → the 8 KB-frame re-call
//   breakage does NOT apply → 8 small-frame re-calls/v-head are clean.  S
//   persists in the Buffer across the 8 per-token calls.  No cross-tile fifos.
//
// MATH (reused verbatim from the proven 3-pass fst_gdn_scan_kernel.cc; only S
// storage precision changes: fp32 DDR -> bf16 on-tile, the A/B-validated
// lossless mitigation FST_Q35_GDN_BF16S, 39/39 argmax-identical).  fp32 compute
// on bf16-rounded S; the bf16 round happens only at the passB store:
//   passA: a[j]=Σ_i S[i,j]*kn[i] ; b[j]=Σ_i S[i,j]*qn[i]   (row-loop 128, fp32 accum)
//   delta: c=kn·qn ; kvm=gdec*a ; delta=(v-kvm)*beta ; y=gdec*b+delta*c
//   passB: S[i,j]=gdec*S[i,j]+kn[i]*delta[j]                (row-loop 128, store bf16)
//
// bf16<->fp32 conversion idiom (proven in fst_gdn_chunkwise_4tile_micro_kernel.cc):
//   bf16->fp32 (lane-preserving): aie::mul(sv_bf, ones_bf).to_vector<float>()
//   fp32->bf16 (round)          : accfloat.to_vector<bfloat16>()
//
// qn=l2norm(q)/sqrt(128), kn=l2norm(k), gdec=exp(g_logit), beta=sigmoid(b) are
// computed UPSTREAM on ew_unified.  This kernel receives qn,kn,v,gdec,beta ready
// (all fp32).  VEC=8 fp32 lanes; 128/8 = 16 vectors per row.

#include <aie_api/aie.hpp>
#include <stdint.h>

// TEMP DIAGNOSTIC: define to make the recurrence a no-op (isolate load/store).
// #define GDN_RECUR_NOOP

constexpr int HV   = 128;
constexpr int V    = 8;
constexpr int NVEC = HV / V;          // 16
constexpr int K    = 8;               // chunk size (tokens per dispatch)
constexpr int NVH  = 48;              // GDN v-heads

// Per-token param packet [qn(128)|kn(128)|v(128)|gdec(1)|beta(1)|pad(6)] = 392 fp32.
// 392 % 8 == 0 so every token's packet base is 8-float (32-byte) aligned for
// load_v (the load-bearing alignment rule from fst_gdn_scan_kernel.cc).  gdec
// at 384, beta at 385 (scalar reads, no alignment need).
constexpr int PKT_PAR  = 392;
constexpr int PAR_GDEC = 384;
constexpr int PAR_BETA = 385;

// ── zero the persistent row counter (call before the load phase and before the
//    store phase of each v-head).  Small frame.
extern "C" void gdn_ctr_zero(float *__restrict ctr) {
    ctr[0] = 0.0f;
}

// ── load ONE S0 row (fp32 DMA) into the persistent bf16 S at row `ctr[0]`,
//    then increment the counter.  Small frame (no stack arrays).
extern "C" void gdn_load_s0_row(bfloat16 *__restrict S,
                                float *__restrict ctr,
                                const float *__restrict in) {
#ifdef GDN_LOADSTORE_NOOP
    ctr[0] += 1.0f; return;
#endif
    const int idx = (int)ctr[0];
    bfloat16 *row = S + (size_t)idx * HV;
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    for (int n = 0; n < NVEC; ++n) {
        // fp32 row -> accfloat (mul by 1) -> bf16 (round) -> S row
        aie::accum<accfloat, V> acc = aie::mul(aie::load_v<V>(in + n * V), 1.0f);
        aie::store_v(row + n * V, acc.template to_vector<bfloat16>());
    }
    ctr[0] = (float)(idx + 1);
}

// ── store ONE S row (bf16) from the persistent S at row `ctr[0]` to a fp32 DMA
//    output packet, then increment the counter.  Small frame.
extern "C" void gdn_store_sf_row(bfloat16 *__restrict S,
                                 float *__restrict ctr,
                                 float *__restrict out) {
#ifdef GDN_LOADSTORE_NOOP
    ctr[0] += 1.0f; return;
#endif
    const int idx = (int)ctr[0];
    bfloat16 *row = S + (size_t)idx * HV;
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    for (int n = 0; n < NVEC; ++n) {
        // bf16 S row -> fp32 (mul by 1) -> out row
        aie::vector<float, V> fv =
            aie::mul(aie::load_v<V>(row + n * V), ones_bf).template to_vector<float>();
        aie::store_v(out + n * V, fv);
    }
    ctr[0] = (float)(idx + 1);
}

// ── ONE token's recurrence on the persistent bf16 S.  Reads qn/kn/v/gdec/beta
//    from `par`, writes y[128] to `yout`, RMWs S in place (bf16, fp32 compute).
//    Small frame: a[128]+b[128]+delta[128] = 1.5 KB stack only.
extern "C" void gdn_chunkwise_recur(bfloat16 *__restrict S,
                                    const float *__restrict par,
                                    float *__restrict yout) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
#ifdef GDN_RECUR_NOOP
    for (int n = 0; n < NVEC; ++n) aie::store_v(yout + n * V, aie::zeros<float, V>());
    return;
#endif
#ifdef GDN_RECUR_LOADONLY
    // Diagnostic: only stream-load every S row (bf16->fp32), no MAC, no RMW.
    // Measures raw Buffer-load throughput (is the Buffer access the stall?).
    volatile float sink = 0.0f;
    for (int i = 0; i < HV; ++i) {
        const bfloat16 *s_row = S + (size_t)i * HV;
        for (int n = 0; n < NVEC; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(s_row + n * V), ones_bf).template to_vector<float>();
            sink += aie::reduce_add(sv_f);
        }
    }
    yout[0] = sink; for (int n = 1; n < HV; ++n) yout[n] = 0.0f;
    return;
#endif
    const float *qn   = par;
    const float *kn   = par + HV;
    const float *v    = par + 2 * HV;
    const float gdec  = par[PAR_GDEC];
    const float beta  = par[PAR_BETA];

    float a[HV] __attribute__((aligned(32)));
    float b[HV] __attribute__((aligned(32)));
    float delta[HV] __attribute__((aligned(32)));

    const auto z = aie::zeros<float, V>();

    // zero a, b
    for (int n = 0; n < NVEC; ++n) {
        aie::store_v(a + n * V, z);
        aie::store_v(b + n * V, z);
    }

    // passA: a[j] += S[i,j]*kn[i] ; b[j] += S[i,j]*qn[i]   (S read as bf16->fp32)
    for (int i = 0; i < HV; ++i) {
        const float kn_i = kn[i];
        const float qn_i = qn[i];
        const bfloat16 *s_row = S + (size_t)i * HV;
        for (int n = 0; n < NVEC; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(s_row + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> av = aie::load_v<V>(a + n * V);
            aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
            aie::store_v(a + n * V, aie::add(av, aie::mul(sv_f, kn_i).template to_vector<float>()));
            aie::store_v(b + n * V, aie::add(bv, aie::mul(sv_f, qn_i).template to_vector<float>()));
        }
    }

    // delta: c = kn·qn ; kvm = gdec*a ; delta = (v-kvm)*beta ; y = gdec*b + delta*c
    float c = 0.0f;
    for (int n = 0; n < NVEC; ++n) {
        c += aie::reduce_add(
            aie::mul(aie::load_v<V>(kn + n * V), aie::load_v<V>(qn + n * V)).template to_vector<float>());
    }
    for (int n = 0; n < NVEC; ++n) {
        aie::vector<float, V> av = aie::load_v<V>(a + n * V);
        aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
        aie::vector<float, V> vv = aie::load_v<V>(v + n * V);
        aie::vector<float, V> kvm = aie::mul(av, gdec).template to_vector<float>();          // gdec*a
        aie::vector<float, V> dv  = aie::mul(aie::sub(vv, kvm), beta).template to_vector<float>(); // (v-kvm)*beta
        aie::store_v(delta + n * V, dv);
        aie::vector<float, V> yv =
            aie::add(aie::mul(bv, gdec).template to_vector<float>(),
                     aie::mul(dv, c).template to_vector<float>());                            // gdec*b + delta*c
        aie::store_v(yout + n * V, yv);
    }

    // passB: S[i,j] = gdec*S[i,j] + kn[i]*delta[j]   (RMW S, fp32 compute, round to bf16 at store)
    for (int i = 0; i < HV; ++i) {
        const float kn_i = kn[i];
        bfloat16 *s_row = S + (size_t)i * HV;
        for (int n = 0; n < NVEC; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(s_row + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> dv = aie::load_v<V>(delta + n * V);
            aie::accum<accfloat, V> acc =
                aie::add(aie::mul(sv_f, gdec), aie::mul(dv, kn_i));                            // accfloat
            aie::store_v(s_row + n * V, acc.template to_vector<bfloat16>());                  // round -> bf16 S
        }
    }
}