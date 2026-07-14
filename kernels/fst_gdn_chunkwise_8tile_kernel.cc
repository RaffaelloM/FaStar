// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_chunkwise_8tile_kernel.cc — Qwen3.5-Next Gated Delta Net (GDN) CHUNKWISE
// M=K scan, 8-TILE COLUMN-SPLIT design (Stage 1.2-v2).
//
// Collapses the shipped 3-pass-per-token GDN (3 dispatches/SSM-layer × 48 layers
// = 144 dispatches/token) into ONE dispatch/SSM-layer (K=8 tokens, 48 v-heads),
// holding S on-tile across the K steps.  Prerequisite for lossless K× spec-dec.
//
// WHY 8 TILES + COLUMN-SPLIT (fix for the 1-tile persistent-Buffer dead-end
// where the 32KB 4-bank register-addressed Buffer did not pipeline —
// 1.36µs/load_v ≈ 340× a tile-local load):
//  - S[128,128] bf16 = 32KB split by COLUMN across 8 tiles → each tile holds
//    16 cols × 128 rows = 4KB (single bank — hoped to pipeline).
//  - COLUMN-SPLIT makes the recurrence INDEPENDENT per tile (a[c]/b[c] for the
//    tile's 16 c's need only the tile's 16 cols + full kn,qn; c=kn·qn
//    redundant; delta/y/passB elementwise on 16 cols) → NO per-token cross-tile
//    reduction, NO per-token handshake deadlock.
//  - 8 tiles = 8× compute parallelism → ~5.6ms/tile, hidden under ~30ms syncobj.
//
// TOPOLOGY — 2 CHAINS of 4 tiles (each chain = 4-tile cascade, 1 shim MM2S + 1
// shim S2MM per chain → 2+2 total):
//   Chain A (cols  0..63): shim MM2S1 → T0 → T1 → T2 → T3 → shim S2MM1
//   Chain B (cols 64..127): shim MM2S2 → T4 → T5 → T6 → T7 → shim S2MM2
//   Within-chain col_offset CO = 0,16,32,48.  All fifos flow FORWARD.
//
// STREAM BUDGET — an AIE2P tile has ~2 in + 2 out core↔core streams.  So the 4
// logical streams (S0, params, y, Sf) are COMBINED into TWO per tile pair:
//   f_in  (shim→T0→..→T3): carries S0 rows (128/v-head) THEN param packets
//          (8/v-head), all as PKT=328-float packets (S0 row in the first 64,
//          rest pad; params use the full 328).  One in-stream per tile.
//   f_out (T0→..→T3→shim): carries y rows (8/v-head) THEN Sf rows (128/v-head),
//          all PKT=328 (y/Sf in the first 64, rest pad).  One out-stream per tile.
//   Middle tile = 2 in + 2 out (fits the budget); the 4-separate-stream design
//   (4 in + 4 out) failed placement ("compute-peer DMA budget unsatisfiable").
//
// PKT=328 layout: [kn(128)|qn(128)|v(64)|gdec(1)|beta(1)|pad(6)] for a param
// packet; an S0/y/Sf row occupies the first 64 (CHAIN cols), rest pad.  The
// tile reads kn@0, qn@128, v@256+CO, gdec@320, beta@321 for a param packet;
// reads in_pkt[CO:CO+16] for an S0 row; writes out_pkt[CO:CO+16] for y/Sf.
//
// PER TILE: persistent S Buffer (bfloat16[128*16]=4KB) + ctr Buffer.  The IRON
// worker loops 48 v-heads; per v-head: ctr=0 → 128 load (acq f_in s0, fwd) →
// 8 recur (acq f_in par + f_out y, fwd) → ctr=0 → 128 store (acq f_out sf, fwd).
//
// MATH (reused verbatim from the proven 3-pass fst_gdn_scan_kernel.cc; S =
// bf16 on-tile, FST_Q35_GDN_BF16S lossless, 39/39 argmax).  ALL VECTORIAL:
// hot-loop ops are 8-lane vec load/store/mul/add; scalars broadcast into vec
// mul/add; the only scalar reduction is c=reduce_add(kn·qn) (one dot/token).
//
// bf16<->fp32: aie::mul(sv_bf, ones_bf).to_vector<float>() / acc.to_vector<bfloat16>()

#include <aie_api/aie.hpp>
#include <stdint.h>

// Diagnostic no-ops (set via -D in gen_gdn_chunkwise_8tile.py compile_flags):
//  GDN_8T_RECUR_NOOP   — recur_core does nothing (isolates load/store + fwd).
//  GDN_8T_LOADONLY     — recur_core streams every S row (bf16->fp32) but no MAC/RMW.
//  GDN_8T_FWD_NOOP     — skip the 328-float vcopy forwarding in lfwd/recur/store.
#ifdef GDN_8T_RECUR_NOOP
#define RECUR_BODY(S, par, y_out, CO) do { (void)(S); (void)(par); (void)(y_out); (void)(CO); } while (0)
#else
#define RECUR_BODY(S, par, y_out, CO) recur_core(S, par, y_out, CO)
#endif

constexpr int HV    = 128;
constexpr int COLS  = 16;          // columns per tile
constexpr int NCV   = COLS / 8;    // 2 vectors of 8 per row-slice
constexpr int K     = 8;
constexpr int NVH   = 48;
constexpr int CHAIN = 64;          // columns per chain (4 tiles × 16)

constexpr int PKT      = 328;      // unified packet (param = full; S0/y/Sf = first 64)
constexpr int PKT_V    = 256;      // v base offset in a param packet
constexpr int PKT_GDEC = 320;
constexpr int PKT_BETA = 321;
constexpr int ACTIVE   = CHAIN;    // 64 active cols (first 64 of a packet)
constexpr int V = 8;

// ── zero the persistent row counter.
extern "C" void gdn_ctr_zero(float *__restrict ctr) { ctr[0] = 0.0f; }

static inline void vcopy(const float *__restrict in, float *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n) aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}
static inline void vzero(float *__restrict out, int nvec) {
    const auto z = aie::zeros<float, V>();
    for (int n = 0; n < nvec; ++n) aie::store_v(out + n * V, z);
}

// ── extract this tile's 16 cols from in_pkt[0:64] -> bf16 S row idx.
static inline void load_s0_core(bfloat16 *__restrict S, int idx,
                                const float *__restrict in_pkt, int CO) {
    bfloat16 *srow = S + (size_t)idx * COLS;
    for (int n = 0; n < NCV; ++n) {
        aie::accum<accfloat, V> acc = aie::mul(aie::load_v<V>(in_pkt + CO + n * V), 1.0f);
        aie::store_v(srow + n * V, acc.template to_vector<bfloat16>());
    }
}

extern "C" void gdn_load_s0_fwd(bfloat16 *__restrict S, float *__restrict ctr,
                                const float *__restrict in_pkt,
                                float *__restrict out_pkt, int CO) {
    const int idx = (int)ctr[0];
    load_s0_core(S, idx, in_pkt, CO);
#ifndef GDN_8T_FWD_NOOP
    vcopy(in_pkt, out_pkt, PKT / V);           // forward the full 328-float packet
#else
    (void)out_pkt;
#endif
    ctr[0] = (float)(idx + 1);
}
extern "C" void gdn_load_s0_end(bfloat16 *__restrict S, float *__restrict ctr,
                                const float *__restrict in_pkt, int CO) {
    const int idx = (int)ctr[0];
    load_s0_core(S, idx, in_pkt, CO);
    ctr[0] = (float)(idx + 1);
}

// ── one token's recurrence on the tile's 16 cols.  Writes y[CO:CO+16] into
//    y_out (caller prepped y_out: copied upstream or zeroed).
static inline void recur_core(bfloat16 *__restrict S,
                              const float *__restrict par,
                              float *__restrict y_out, int CO) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    const float *kn   = par;
    const float *qn   = par + HV;
    const float *v    = par + PKT_V + CO;      // this tile's 16 cols of v
    const float gdec  = par[PKT_GDEC];
    const float beta  = par[PKT_BETA];

#ifdef GDN_8T_LOADONLY
    // Isolate the S-load cost: stream every S row bf16->fp32 into a STACK
    // accumulator (no scalar extract, no fifo store — those would serialize
    // and mask the real S-load latency).  Loads S exactly as passA does.
    float acc[COLS] __attribute__((aligned(32)));
    const auto z = aie::zeros<float, V>();
    for (int n = 0; n < NCV; ++n) aie::store_v(acc + n * V, z);
    for (int i = 0; i < HV; ++i) {
        const bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> av = aie::load_v<V>(acc + n * V);
            aie::store_v(acc + n * V, aie::add(av, sv_f));   // stack accum, no fifo/scalar
        }
    }
    // drain acc to y_out once (negligible) so the loop isn't dead-coded.
    for (int n = 0; n < NCV; ++n) aie::store_v(y_out + CO + n * V, aie::load_v<V>(acc + n * V));
    (void)kn; (void)qn; (void)v; (void)gdec; (void)beta;
    return;
#endif

    float a[COLS] __attribute__((aligned(32)));
    float b[COLS] __attribute__((aligned(32)));
    float delta[COLS] __attribute__((aligned(32)));
    const auto z = aie::zeros<float, V>();
    for (int n = 0; n < NCV; ++n) { aie::store_v(a + n * V, z); aie::store_v(b + n * V, z); }

    for (int i = 0; i < HV; ++i) {            // passA: a[c]+=S[i,c]*kn[i]; b[c]+=S[i,c]*qn[i]
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
    float c = 0.0f;                           // delta: c=kn·qn; kvm=gdec*a; delta=(v-kvm)*beta; y=gdec*b+delta*c
    for (int n = 0; n < HV / V; ++n)
        c += aie::reduce_add(
            aie::mul(aie::load_v<V>(kn + n * V), aie::load_v<V>(qn + n * V)).template to_vector<float>());
    for (int n = 0; n < NCV; ++n) {
        aie::vector<float, V> av = aie::load_v<V>(a + n * V);
        aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
        aie::vector<float, V> vv = aie::load_v<V>(v + n * V);
        aie::vector<float, V> kvm = aie::mul(av, gdec).template to_vector<float>();
        aie::vector<float, V> dv  = aie::mul(aie::sub(vv, kvm), beta).template to_vector<float>();
        aie::store_v(delta + n * V, dv);
        aie::vector<float, V> yv =
            aie::add(aie::mul(bv, gdec).template to_vector<float>(),
                     aie::mul(dv, c).template to_vector<float>());
        aie::store_v(y_out + CO + n * V, yv);
    }
    for (int i = 0; i < HV; ++i) {            // passB: S[i,c]=gdec*S[i,c]+kn[i]*delta[c] (store bf16)
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

extern "C" void gdn_recur_head(bfloat16 *__restrict S, const float *__restrict par,
                               float *__restrict par_out, float *__restrict y_out, int CO) {
#ifndef GDN_8T_FWD_NOOP
    vcopy(par, par_out, PKT / V);
#else
    (void)par_out;
#endif
    vzero(y_out, PKT / V);
    RECUR_BODY(S, par, y_out, CO);
}
extern "C" void gdn_recur_mid(bfloat16 *__restrict S, const float *__restrict par,
                              float *__restrict par_out, const float *__restrict y_in,
                              float *__restrict y_out, int CO) {
#ifndef GDN_8T_FWD_NOOP
    vcopy(par, par_out, PKT / V);
#else
    (void)par_out;
#endif
    vcopy(y_in, y_out, PKT / V);
    RECUR_BODY(S, par, y_out, CO);
}
extern "C" void gdn_recur_tail(bfloat16 *__restrict S, const float *__restrict par,
                               const float *__restrict y_in, float *__restrict y_out, int CO) {
    vcopy(y_in, y_out, PKT / V);
    RECUR_BODY(S, par, y_out, CO);
}

// ── write the tile's 16 cols of S row idx -> out_pkt[CO:CO+16] (fp32).
static inline void store_sf_core(bfloat16 *__restrict S, int idx,
                                 float *__restrict out_pkt, int CO) {
    const bfloat16 *srow = S + (size_t)idx * COLS;
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    for (int n = 0; n < NCV; ++n) {
        aie::vector<float, V> fv =
            aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
        aie::store_v(out_pkt + CO + n * V, fv);
    }
}
extern "C" void gdn_store_sf_head(bfloat16 *__restrict S, float *__restrict ctr,
                                  float *__restrict out_pkt, int CO) {
    const int idx = (int)ctr[0];
    vzero(out_pkt, PKT / V);
    store_sf_core(S, idx, out_pkt, CO);
    ctr[0] = (float)(idx + 1);
}
extern "C" void gdn_store_sf_mid(bfloat16 *__restrict S, float *__restrict ctr,
                                 const float *__restrict in_pkt,
                                 float *__restrict out_pkt, int CO) {
    const int idx = (int)ctr[0];
#ifndef GDN_8T_FWD_NOOP
    vcopy(in_pkt, out_pkt, PKT / V);
#else
    (void)in_pkt;
#endif
    store_sf_core(S, idx, out_pkt, CO);
    ctr[0] = (float)(idx + 1);
}