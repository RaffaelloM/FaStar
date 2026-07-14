// fst_gdn_8kslab_mmul_kernel.cc — Step-2: passA via native aie::mmul<8,8,8>
// (emulated bfp16 to avoid the -4 register underflow, per fst_qwopus_ffn).
//
// The nomul canary (fst_gdn_8kslab_nomul) proved the MACs are ~660 ms = 56% of
// the slab's 1180 ms — the scalar-broadcast `aie::mul(sv_f, kn_i)` (kn_i a per-row
// scalar) pipelines badly under DMA contention.  MMUL<8,8,8> loads kn/qn as
// VECTOR rows (no per-row scalar broadcast) and S as 8×8 blocks, so it sidesteps
// that stall.  passA batches kn+qn as M=2 (A row 0 = kn, row 1 = qn, rows 2-7
// unused-garbage ⇒ C rows 2-7 unused) so ONE mmul K-reduction gives BOTH a and b
// for an 8-col group.  passB stays scalar-broadcast in this first cut.
//
// PRECISION: kn/qn/S are bf16; mmul is bf16×bf16→accfloat.  Upcasting bf16→fp32
// is exact, so bf16×bf16→fp32-accum == the prior fp32×fp32→fp32-accum passA
// (bit-identical a/b).  No argmax risk from passA.  passB unchanged.
//
// Built via: FST_GDN_SRC=kernels/fst_gdn_8kslab_mmul_kernel.cc \
//            FST_GDN_SUFFIX=_mmul python3 kernels/gen_gdn_8kslab.py
// native bf16 mmul (no BFP16 emulate — bit-identical products, avoids recurrence explosion)
#include <aie_api/aie.hpp>
#include <stdint.h>
using namespace aie::operators;

constexpr int HV   = 128;
constexpr int COLS = 32;
constexpr int NCV  = COLS / 8;     // 4
constexpr int K    = 8;
constexpr int V    = 8;
constexpr int NG   = HV / V;       // 16
constexpr int PAR_SZ    = 2 * HV + COLS + 8;   // 296
constexpr int PAR_V     = 2 * HV;             // 256
constexpr int PAR_GDEC  = 2 * HV + COLS;       // 288
constexpr int YD_DELTA  = COLS;
constexpr int YD_SZ     = 2 * COLS;

using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;  // A[8,8]@B[8,8]->C[8,8] fp32

static inline void vcopy_bf16(const bfloat16 *__restrict in,
                              bfloat16 *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n)
        aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}

static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict yd) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    // kn/qn/vv kept BF16 (MMUL wants bf16; no upcast).
    bfloat16 kn_b[HV]  __attribute__((aligned(32)));
    bfloat16 qn_b[HV]  __attribute__((aligned(32)));
    bfloat16 vv_b[COLS] __attribute__((aligned(32)));
    for (int n = 0; n < NG;  ++n) aie::store_v(kn_b + n * V, aie::load_v<V>(par + n * V));
    for (int n = 0; n < NG;  ++n) aie::store_v(qn_b + n * V, aie::load_v<V>(par + HV + n * V));
    for (int n = 0; n < NCV; ++n) aie::store_v(vv_b + n * V, aie::load_v<V>(par + PAR_V + n * V));
    aie::vector<float, V> tailf =
        aie::mul(aie::load_v<V>(par + PAR_GDEC), ones_bf).template to_vector<float>();
    const float gdec = tailf[0], beta = tailf[1], c = tailf[2];

    float a[COLS]     __attribute__((aligned(32)));
    float b[COLS]     __attribute__((aligned(32)));
    float delta[COLS] __attribute__((aligned(32)));

    // ── passA via MMUL<8,8,8>: a[c]=Σ_i S[i,c]·kn[i], b[c]=Σ_i S[i,c]·qn[i] ──
    // A row 0 = kn chunk, row 1 = qn chunk, rows 2-7 = garbage (C rows 2-7 unused).
    // B = S[8k..8k+7, 8n..8n+7] packed n-fast then transposed (the FFN pattern).
    alignas(128) bfloat16 A_buf[64];
    alignas(128) bfloat16 B_buf[64];
    for (int n = 0; n < NCV; ++n) {            // 4 col-groups of 8
        MMUL C;                                // NOTE: ctor does NOT zero the acc
        for (int k = 0; k < NG; ++k) {         // 16 row-groups of 8 (K-reduction)
            aie::store_v(A_buf + 0 * 8, aie::load_v<V>(kn_b + k * V));   // row 0 = kn
            aie::store_v(A_buf + 1 * 8, aie::load_v<V>(qn_b + k * V));   // row 1 = qn
            aie::vector<bfloat16, 64> A_op = aie::load_v<64>(A_buf);
            for (int r = 0; r < 8; ++r)                                  // 8 S rows -> B_buf n-fast
                aie::store_v(B_buf + r * 8, aie::load_v<V>(S + (size_t)(8 * k + r) * COLS + 8 * n));
            aie::vector<bfloat16, 64> B_op = aie::transpose(aie::load_v<64>(B_buf), 8, 8);
            if (k == 0) C.mul(A_op, B_op);     // OVERWRITE (kills stale acc across tokens)
            else        C.mac(A_op, B_op);     // accumulate
        }
        aie::vector<float, 64> cv = C.template to_vector<float>();       // cv[m*8+j] = C[m,j]
        aie::store_v(a + n * V, cv.template extract<V>(0));              // row 0 = a
        aie::store_v(b + n * V, cv.template extract<V>(1));              // row 1 = b
    }

    // ── delta block (unchanged scalar; vv upcast bf16->fp32) ──
    for (int n = 0; n < NCV; ++n) {
        aie::vector<float, V> av = aie::load_v<V>(a + n * V);
        aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
        aie::vector<float, V> vv_f =
            aie::mul(aie::load_v<V>(vv_b + n * V), ones_bf).template to_vector<float>();
        aie::vector<float, V> kvm = aie::mul(av, gdec).template to_vector<float>();
        aie::accum<accfloat, V> dv_acc = aie::mul(aie::sub(vv_f, kvm), beta);
        aie::vector<float, V> dv = dv_acc.template to_vector<float>();
        aie::store_v(delta + n * V, dv);
        aie::accum<accfloat, V> yv = aie::add(aie::mul(bv, gdec), aie::mul(dv, c));
        aie::store_v(yd + n * V, yv.template to_vector<bfloat16>());
        aie::store_v(yd + YD_DELTA + n * V, dv_acc.template to_vector<bfloat16>());
    }

    // ── passB (unchanged scalar-broadcast): S[i,c]=gdec*S[i,c]+kn[i]*delta[c] ──
    // kn upcast bf16->fp32 per row (scalar).  This is the 3rd mul the nomul removed;
    // a follow-up cut will MMUL the kn⊗delta outer product via <8,1,8>.
    for (int i = 0; i < HV; ++i) {
        const float kn_i = aie::mul(aie::load_v<V>(kn_b + (i / V) * V), ones_bf).template to_vector<float>()[i % V];
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

extern "C" void gdn_8kslab_vhead(const bfloat16 *__restrict s0,  const bfloat16 *__restrict par,
                                 bfloat16 *__restrict snew,      bfloat16 *__restrict yd) {
    bfloat16 S[HV * COLS] __attribute__((aligned(32)));
    for (int i = 0; i < HV; ++i) vcopy_bf16(s0 + (size_t)i * COLS, S + (size_t)i * COLS, NCV);
    for (int t = 0; t < K; ++t) recur_core(S, par + (size_t)t * PAR_SZ, yd + (size_t)t * YD_SZ);
    for (int i = 0; i < HV; ++i) vcopy_bf16(S + (size_t)i * COLS, snew + (size_t)i * COLS, NCV);
}