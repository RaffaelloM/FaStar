// fst_gdn_8kslab_vecmul_kernel.cc — Step-2 decisive canary.
//
// nomul (520 ms) replaced `mul(sv_f, kn_i)` with `add(av, sv_f)` — changing BOTH
// the per-row scalar broadcast AND the mul instruction.  This canary isolates the
// two: it keeps a VECTOR×VECTOR MUL (`mul(sv_f, ones_f)`, no scalar broadcast) in
// place of the per-row scalar-broadcast muls (passA kn_i/qn_i, passB kn_i).  Math
// is wrong (multiplies by 1.0) — LATENCY ONLY, like nomul.
//
//   vecmul ≈ 520 ms (≈ nomul) ⇒ the SCALAR BROADCAST is the stall; the mul instr
//            is cheap.  Then a no-broadcast exact path (e.g. fp32 mmul) could win.
//   vecmul ≈ 1180 ms (≈ full) ⇒ the MUL INSTRUCTION itself stalls under DMA
//            contention; only a different instr (mmul) helps — and bf16 mmul is
//            bfp16-lossy (explodes the recurrence) ⇒ slab is structurally dead.
//
// Built via: FST_GDN_SRC=kernels/fst_gdn_8kslab_vecmul_kernel.cc \
//            FST_GDN_SUFFIX=_vecmul python3 kernels/gen_gdn_8kslab.py
#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV   = 128;
constexpr int COLS = 32;
constexpr int NCV  = COLS / 8;
constexpr int K    = 8;
constexpr int V    = 8;
constexpr int PAR_SZ    = 2 * HV + COLS + 8;
constexpr int PAR_V     = 2 * HV;
constexpr int PAR_GDEC  = 2 * HV + COLS;
constexpr int YD_DELTA  = COLS;
constexpr int YD_SZ     = 2 * COLS;

static inline void vcopy_bf16(const bfloat16 *__restrict in,
                              bfloat16 *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n)
        aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}
static inline aie::vector<bfloat16, V> f2bf(aie::vector<float, V> v) {
    return aie::from_vector<accfloat>(v).template to_vector<bfloat16>();
}

static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict yd) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    const aie::vector<float, V> ones_f = aie::broadcast<float, V>(1.0f);
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
    (void)tailf[1]; (void)tailf[2];

    const auto z = aie::zeros<float, V>();
    for (int n = 0; n < NCV; ++n) { aie::store_v(a + n * V, z); aie::store_v(b + n * V, z); }

    // passA — VECMUL: a += mul(sv_f, ones_f)  (vector×vector mul, NO scalar broadcast).
    // kn_i/qn_i LOADED (memory traffic preserved) but NOT used as broadcast scalars.
    for (int i = 0; i < HV; ++i) {
        (void)kn[i]; (void)qn[i];                 // load preserved
        const bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> av = aie::load_v<V>(a + n * V);
            aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
            aie::store_v(a + n * V, aie::add(av, aie::mul(sv_f, ones_f).template to_vector<float>()));
            aie::store_v(b + n * V, aie::add(bv, aie::mul(sv_f, ones_f).template to_vector<float>()));
        }
    }
    // delta block — drop muls (kvm=0; delta=vv; y=b+delta), like nomul.
    for (int n = 0; n < NCV; ++n) {
        aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
        aie::vector<float, V> vvi = aie::load_v<V>(vv + n * V);
        aie::vector<float, V> dv = vvi;
        aie::store_v(delta + n * V, dv);
        aie::vector<float, V> yv = aie::add(bv, dv);
        aie::store_v(yd + n * V, f2bf(yv));
        aie::store_v(yd + YD_DELTA + n * V, f2bf(dv));
    }
    // passB — VECMUL: snew = gdec*sv + mul(dv, ones_f)  (kn_i broadcast removed;
    // dv×ones is vector×vector).  kn_i LOADED, gdec scalar mul kept (per-token, few).
    for (int i = 0; i < HV; ++i) {
        (void)kn[i];                               // load preserved
        bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n) {
            aie::vector<float, V> sv_f =
                aie::mul(aie::load_v<V>(srow + n * V), ones_bf).template to_vector<float>();
            aie::vector<float, V> dv = aie::load_v<V>(delta + n * V);
            aie::vector<float, V> snew =
                aie::add(aie::mul(sv_f, gdec).template to_vector<float>(),
                         aie::mul(dv, ones_f).template to_vector<float>());
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