// fst_gdn_8kslab_nomul_kernel.cc — Step-2 diagnostic: same memory structure as
// fst_gdn_8kslab_kernel.cc (separate fp32 stack arrays kn/qn/vv, 8 KB bf16
// stack-S RMW, same DMA streams, same packet layout) but the MACs are replaced
// by ADDS — `a += sv_f` instead of `a += sv_f*kn_i`, `S = gdec*S + delta`
// instead of `S = gdec*S + kn_i*delta`.  ALL loads (kn[i], qn[i], delta[c], vv)
// are preserved so memory traffic is byte-identical; only the multiply is
// dropped.  If the slab latency stays ~1180 ms ⇒ MEMORY-bound (MACs negligible)
// ⇒ aie::mmul (which only changes the MACs) CANNOT help the slab.  If it drops
// ⇒ MAC-bound ⇒ MMUL worth building.  Output is numerically meaningless
// (correctness not expected); this measures LATENCY only.
//
// Built via: FST_GDN_SRC=kernels/fst_gdn_8kslab_nomul_kernel.cc \
//            FST_GDN_SUFFIX=_nomul python3 kernels/gen_gdn_8kslab.py
#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV   = 128;
constexpr int COLS = 32;
constexpr int NCV  = COLS / 8;
constexpr int K    = 8;
constexpr int V    = 8;
constexpr int PAR_SZ    = 2 * HV + COLS + 8;   // 296
constexpr int PAR_V     = 2 * HV;             // 256
constexpr int PAR_GDEC  = 2 * HV + COLS;       // 288
constexpr int YD_DELTA  = COLS;
constexpr int YD_SZ     = 2 * COLS;

static inline void vcopy_bf16(const bfloat16 *__restrict in,
                              bfloat16 *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n)
        aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}
// float vector -> bf16 vector (round via the accfloat accum path).
static inline aie::vector<bfloat16, V> f2bf(aie::vector<float, V> v) {
    return aie::from_vector<accfloat>(v).template to_vector<bfloat16>();
}

static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict yd) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    constexpr int NG = HV / V;
    float kn[HV]     __attribute__((aligned(32)));
    float qn[HV]     __attribute__((aligned(32)));
    float vv[COLS]   __attribute__((aligned(32)));
    float a[COLS]    __attribute__((aligned(32)));
    float b[COLS]    __attribute__((aligned(32)));
    float delta[COLS] __attribute__((aligned(32)));
    for (int n = 0; n < NG; ++n)
        aie::store_v(kn + n * V, aie::mul(aie::load_v<V>(par + n * V), ones_bf).template to_vector<float>());
    for (int n = 0; n < NG; ++n)
        aie::store_v(qn + n * V, aie::mul(aie::load_v<V>(par + HV + n * V), ones_bf).template to_vector<float>());
    for (int n = 0; n < NCV; ++n)
        aie::store_v(vv + n * V, aie::mul(aie::load_v<V>(par + PAR_V + n * V), ones_bf).template to_vector<float>());
    aie::vector<float, V> tailf =
        aie::mul(aie::load_v<V>(par + PAR_GDEC), ones_bf).template to_vector<float>();
    const float gdec = tailf[0];
    (void)tailf[1]; (void)tailf[2];   // beta/c unused (no mul)

    const auto z = aie::zeros<float, V>();
    for (int n = 0; n < NCV; ++n) { aie::store_v(a + n * V, z); aie::store_v(b + n * V, z); }

    // passA — NOMUL: a += sv_f (not sv_f*kn_i); kn_i/qn_i LOADED but not multiplied.
    for (int i = 0; i < HV; ++i) {
        const float kn_i = kn[i];   // load preserved (memory traffic)
        const float qn_i = qn[i];   // load preserved
        (void)kn_i; (void)qn_i;
        const bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> av = aie::load_v<V>(a + n * V);
            aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
            aie::store_v(a + n * V, aie::add(av, sv_f));          // was add(av, mul(sv_f, kn_i))
            aie::store_v(b + n * V, aie::add(bv, sv_f));          // was add(bv, mul(sv_f, qn_i))
        }
    }
    // delta block — keep structure, drop the muls (kvm=0, delta=vv, y=b).
    for (int n = 0; n < NCV; ++n) {
        aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
        aie::vector<float, V> vvi = aie::load_v<V>(vv + n * V);
        aie::vector<float, V> dv = vvi;                            // delta = vv (no (v-gdec*a)*beta)
        aie::store_v(delta + n * V, dv);
        aie::vector<float, V> yv = aie::add(bv, dv);               // y = b + delta (no gdec/c muls)
        aie::store_v(yd + n * V, f2bf(yv));
        aie::store_v(yd + YD_DELTA + n * V, f2bf(dv));
    }
    // passB — NOMUL: S = gdec*S + delta (not gdec*S + kn_i*delta); kn_i/delta LOADED.
    for (int i = 0; i < HV; ++i) {
        const float kn_i = kn[i];   // load preserved
        (void)kn_i;
        bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> dv = aie::load_v<V>(delta + n * V);   // load preserved
            aie::vector<float, V> snew = aie::add(aie::mul(sv_f, gdec).template to_vector<float>(), dv);
            aie::store_v(srow + n * V, f2bf(snew));
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