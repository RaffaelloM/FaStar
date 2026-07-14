// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_fused_kernel.cc — FUSED Qwen3.5-Next Gated Delta Net scan, ONE dispatch.
//
// A3 v2 (2026-07-13): 1 MM2S + 1 S2MM = the bit-correct 3-pass passA channel
// budget.  Fixes both race mechanisms (vh0 2-fill startup race; 2-S2MM
// drain-flush cascade) — see docs/QWOPUS_FUSED_GDN_RACE_ROOTCAUSE.md.
//
// Per v-head (S0 streamed from DDR, read TWICE — passA then passB — the 64 KB
// state does not fit in a 64 KB tile):
//   load params  : v/kn/qn full + gdec/beta from the 3 prepended f_s0 param-pkts
//                 into scr[512:898]
//   passA (128) : a += S0[i]*kn[i] ; b += S0[i]*qn[i]   # a=S0ᵀ@kn, b=S0ᵀ@qn
//   delta  (1)  : c=kn·qn ; kvm=gdec*a ; delta=(v-kvm)*beta ; y=gdec*b + delta*c
//   passB (128) : S2[i] = gdec*S0[i] + kn[i]*delta       # updated state
//
// ONE input fifo  f_s0: per v-head 259 136-float packets =
//   [pkt_v(136)][pkt_kn(136)][pkt_qn(136)][128 passA row-pkts][128 passB row-pkts]
//   pkt_v  = [v(128)|gdec@128|beta@129|pad(6)]
//   pkt_kn = [kn(128)|pad(8)] ; pkt_qn = [qn(128)|pad(8)]
//   row    = [S0_row(128)|kn_i@128|qn_i@129|gdec@130|pad(5)]   (passA+passB)
//
// ONE output fifo f_out (1024-pkt, depth=2): scr held in buffer A; 128 S2
// row-pkts stream through buffer B (released first); scr released last ->
// drain order per v-head = [S2(128 rows), y].  Engine unpacks S2 then y.
//
// Held scratch scr[1024] (898 used):
//   [0:128]   a        [128:256]  b        [256:384] delta    [384:512] y
//   [512:640] kn_full  [640:768] qn_full  [768:896] v_full   [896] gdec  [897] beta
// All field offsets 8-aligned (128/256/384/512/640/768/896). kn_i/qn_i/gdec/beta
// are scalar reads (no alignment need).  aie::load_v<V=8> reads 8-float-aligned.
// passA C kernel line-by-line identical to the bit-correct 3-pass gdn_passA_row.

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV = 128;
constexpr int V = 8;
constexpr int NVEC = HV / V;          // 16
constexpr int SCR_USED = 4 * HV;      // 512 = [a|b|delta|y]  (kz zeroes this)
constexpr int ROW = 136;              // [S0(128)|kn_i|qn_i|gdec|pad(5)]

// zero the held scratch [a|b|delta|y] (512 floats) before passA.  (kn/qn/v/gdec
// /beta at [512:898] are overwritten by the load kernels, so they need no zero.)
extern "C" void gdn_zero_scr(float *__restrict scr) {
    auto z = aie::zeros<float, V>();
    for (int n = 0; n < (SCR_USED / V); ++n) aie::store_v(scr + n * V, z);   // [0:512]
}

// load v_full(128) into scr[768:896], gdec into scr[896], beta into scr[897].
// pkt = [v(128)|gdec@128|beta@129|pad].
extern "C" void gdn_load_v(const float *__restrict pkt, float *__restrict scr) {
    float *v_dst = scr + 768;
    for (int n = 0; n < NVEC; ++n)
        aie::store_v(v_dst + n * V, aie::load_v<V>(pkt + n * V));
    scr[896] = pkt[128];   // gdec
    scr[897] = pkt[129];   // beta
}

// load kn_full(128) into scr[512:640].  pkt = [kn(128)|pad].
extern "C" void gdn_load_kn(const float *__restrict pkt, float *__restrict scr) {
    float *kn_dst = scr + 512;
    for (int n = 0; n < NVEC; ++n)
        aie::store_v(kn_dst + n * V, aie::load_v<V>(pkt + n * V));
}

// load qn_full(128) into scr[640:768].  pkt = [qn(128)|pad].
extern "C" void gdn_load_qn(const float *__restrict pkt, float *__restrict scr) {
    float *qn_dst = scr + 640;
    for (int n = 0; n < NVEC; ++n)
        aie::store_v(qn_dst + n * V, aie::load_v<V>(pkt + n * V));
}

// passA: ONE row per block. blk = [S0(128)|kn_i@128|qn_i@129|...]=136.
// RMW scr[0:256] = [a|b]: a[j] += S0[j]*kn_i ; b[j] += S0[j]*qn_i.
// (Line-by-line identical to the bit-correct 3-pass gdn_passA_row.)
extern "C" void gdn_passA_block(
    const float *__restrict blk,
    float *__restrict scr
) {
    float *a_acc = scr;
    float *b_acc = scr + HV;
    const float *row   = blk;
    const float *s_row = row;
    const float kn_i = row[HV + 0];      // 128
    const float qn_i = row[HV + 1];      // 129
    for (int n = 0; n < NVEC; ++n) {
        auto s = aie::load_v<V>(s_row + n * V);
        auto a = aie::load_v<V>(a_acc + n * V);
        aie::store_v(a_acc + n * V, aie::add(a, aie::mul(s, kn_i).template to_vector<float>()));
        auto b = aie::load_v<V>(b_acc + n * V);
        aie::store_v(b_acc + n * V, aie::add(b, aie::mul(s, qn_i).template to_vector<float>()));
    }
}

// delta: once per v-head. Reads a=scr[0:128], b=scr[128:256], and params from
// scr: v=scr+768, kn=scr+512, qn=scr+640, gdec=scr[896], beta=scr[897].
// Writes delta=scr[256:384], y=scr[384:512].
//   c = kn·qn ; kvm = gdec*a ; delta = (v-kvm)*beta ; y = gdec*b + delta*c
extern "C" void gdn_delta_from_ab(float *__restrict scr) {
    const float *a    = scr;
    const float *b    = scr + 128;
    const float *v    = scr + 768;
    const float *kn   = scr + 512;
    const float *qn   = scr + 640;
    const float gdec  = scr[896];
    const float beta  = scr[897];
    float *delta = scr + 256;
    float *y     = scr + 384;

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
        aie::store_v(delta + n * V, dv);
        auto yv = aie::add(aie::mul(bv, gdec).template to_vector<float>(),
                          aie::mul(dv, c).template to_vector<float>());
        aie::store_v(y + n * V, yv);
    }
}

// passB: ONE row per block. blk = [S0(128)|kn_i@128|...|gdec@130]=136.
// delta = scr[256:384] (held). out s2 = 1 row x 128 (in a 1024-pkt).
//   s2[j] = gdec*S0[j] + kn_i*delta[j]
extern "C" void gdn_passB_block_sep(
    const float *__restrict blk,
    const float *__restrict scr,
    float *__restrict s2_out
) {
    const float *delta = scr + 256;
    const float *row   = blk;
    const float *s_row = row;                  // [0,128) 8-aligned
    const float kn_i = row[HV + 0];            // 128
    const float gdec = row[HV + 2];            // 130
    float *out = s2_out;
    for (int n = 0; n < NVEC; ++n) {
        auto s = aie::load_v<V>(s_row + n * V);
        auto d = aie::load_v<V>(delta + n * V);
        auto s2 = aie::add(aie::mul(s, gdec).template to_vector<float>(),
                           aie::mul(d, kn_i).template to_vector<float>());
        aie::store_v(out + n * V, s2);
    }
}

// Copy the held scratch [a|b|delta|y] = scr[0:512] into the streaming y-packet
// ypkt[0:512], preserving the held-scr layout (a@0 b@128 delta@256 y@384) so
// the output BO matches the prior held-scr drain geometry byte-for-byte.
// (scr is now a tile-local aie.iron.Buffer, NOT a drained ObjectFifo packet.)
extern "C" void gdn_copy_scr_y(
    const float *__restrict scr,
    float *__restrict ypkt
) {
    for (int n = 0; n < (2 * SCR_USED / V); ++n)     // 2*512/8 = 128 stores
        aie::store_v(ypkt + n * V, aie::load_v<V>(scr + n * V));
}