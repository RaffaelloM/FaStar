// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_chunkwise_2tile_kernel.cc — Qwen3.5-Next Gated Delta Net (GDN) CHUNKWISE
// M=K scan, 2-TILE COLUMN-SPLIT + SINGLE-CALL STACK-S design (Stage 1.2-v3).
//
// THE FIX for the persistent-aie.iron.Buffer dead-end (proven x2: 1-tile 32KB
// and 8-tile 4KB Buffer both ~340x slow — the Buffer's register-addressed
// MLIR-allocated access does NOT pipeline at any size).  The only proven-fast S
// access is a STACK-LOCAL array inside a single-call C fn (the 4-tile micro:
// 5.87ms, 60ns/op).  This kernel uses that.
//
// DESIGN — 2 INDEPENDENT tiles (no chain), each 1 MM2S + 1 S2MM = 2+2 shim budget:
//   Tile 0 (cols 0..63), Tile 1 (cols 64..127).  Column-split ⇒ recurrence
//   INDEPENDENT per tile ⇒ NO cross-tile reduction, NO chain, NO forwarding.
//   S[128,64] bf16 = 16KB STACK-LOCAL (fits the stack-immediate limit; no Buffer).
//
// STREAMING — PER-V-HEAD unified packets (the 4-tile-micro acquire(1) pattern
// scaled): the worker loops 48 v-heads; per v-head acquires ONE in-packet + ONE
// out-packet (bf16, to fit the ~64KB tile data memory: in 21KB + out 17KB +
// 20KB stack = 58KB), calls gdn_2tile_vhead ONCE, releases.  Stack-S lives for
// the one call (128 load + 8 recur + 128 store).  All accesses are ObjectFifo
// tile-local data or stack — both FAST.  No persistent Buffer anywhere.
//
// PACKET LAYOUT (bf16, single dtype for the unified ObjectFifo):
//   in_pkt  [10768 bf16] = [ S0 (128×64) | par (8×322) ]
//     S0 row i @ in_pkt[i*64 .. i*64+64]            (bf16, copied direct to stack-S)
//     par token t @ in_pkt[8192 + t*322 .. +322] = [kn(128)|qn(128)|v(64)|gdec@320|beta@321] (bf16)
//   out_pkt [8704 bf16]  = [ y (8×64) | sf (128×64) ]
//     y token t @ out_pkt[t*64 .. t*64+64]          (bf16, fp32 recur result rounded)
//     sf row i @ out_pkt[512 + i*64 .. +64]         (bf16, copied direct from stack-S)
//
// MATH (reused verbatim from the proven 3-pass / 8-tile kernel; bf16 S, fp32
// compute, bf16 round at every passB store).  kn/qn/v/gdec/beta are bf16 on the
// DMA, converted to fp32 on read.  ALL VECTORIAL: hot-loop ops are 8-lane vec
// load/store/mul/add; scalars broadcast into vec mul/add; the only scalar
// reduction is c=reduce_add(kn·qn) (one dot/token).

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV    = 128;
constexpr int COLS  = 64;          // columns per tile
constexpr int NCV   = COLS / 8;    // 8 vectors of 8 per row-slice
constexpr int K     = 8;
constexpr int V     = 8;

constexpr int IN_S0   = HV * COLS;            // 8192
constexpr int IN_PAR  = 322;                  // [kn(128)|qn(128)|v(64)|gdec|beta]
constexpr int IN_PKT  = IN_S0 + K * IN_PAR;   // 8192 + 2576 = 10768
constexpr int OUT_Y   = K * COLS;             // 512
constexpr int OUT_SF  = HV * COLS;            // 8192
constexpr int OUT_PKT = OUT_Y + OUT_SF;       // 8704
constexpr int PAR_V   = 256, PAR_GDEC = 320, PAR_BETA = 321;

static inline void vcopy_bf16(const bfloat16 *__restrict in,
                              bfloat16 *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n)
        aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}

// one token's recurrence on the tile's 64 cols.  par points at the token's
// bf16 [kn|qn|v|gdec|beta]; y_out points at the token's 64-bf16 y slot.
// Convert the bf16 par packet to a fp32 stack buffer once (avoids the
// load_v 8-float-alignment rule on scalar bf16 element access), then run the
// proven fp32 recur logic verbatim.
static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict y_out) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    constexpr int PAR_NVEC = (IN_PAR + V - 1) / V;     // ceil(322/8) = 41
    float parf[PAR_NVEC * V] __attribute__((aligned(32)));  // fp32 par buffer (328 slots, 6 pad)
    for (int n = 0; n < PAR_NVEC; ++n)
        aie::store_v(parf + n * V,
            aie::mul(aie::load_v<V>(par + n * V), ones_bf).template to_vector<float>());
    const float *kn  = parf;
    const float *qn  = parf + HV;
    const float *v   = parf + PAR_V;
    const float gdec = parf[PAR_GDEC];
    const float beta = parf[PAR_BETA];

    float a[COLS]   __attribute__((aligned(32)));
    float b[COLS]   __attribute__((aligned(32)));
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
        aie::accum<accfloat, V> yv =
            aie::add(aie::mul(bv, gdec), aie::mul(dv, c));   // fp32 accum -> bf16 on store
        aie::store_v(y_out + n * V, yv.template to_vector<bfloat16>());
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

// ONE v-head: load S0 (bf16 -> bf16 stack-S) -> 8 recur (stack-S RMW) -> store sf
// (bf16 stack-S -> bf16).  Stack-S is a 16KB stack-local array — the FAST path.
extern "C" void gdn_2tile_vhead(const bfloat16 *__restrict in_pkt, bfloat16 *__restrict out_pkt) {
    bfloat16 S[HV * COLS] __attribute__((aligned(32)));   // 16KB stack-local state

    for (int i = 0; i < HV; ++i)               // load S0: 128 rows × 64 cols bf16 -> stack-S
        vcopy_bf16(in_pkt + i * COLS, S + (size_t)i * COLS, NCV);

    for (int t = 0; t < K; ++t)                // 8 chunkwise recurrence steps (stack-S RMW)
        recur_core(S, in_pkt + IN_S0 + t * IN_PAR, out_pkt + t * COLS);

    for (int i = 0; i < HV; ++i)               // store S_final: stack-S -> out sf
        vcopy_bf16(S + (size_t)i * COLS, out_pkt + OUT_Y + i * COLS, NCV);
}