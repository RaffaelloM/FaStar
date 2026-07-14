// fst_gdn_8kslab_mmuln4_kernel.cc — Step-2: passA via NATIVE bf16 aie::mmul<8,8,4>.
//
// <8,8,8,bf16,bf16> is UNCONDITIONALLY bfp16-converted (mmul_bf16_bf16.hpp:112
// calls to_v64bfp16ebs8 + mac_8x8_8x8T_conf) — the EMULATE flag only gates <4,8,8>.
// So the _mmul/_mmulnat variants (both 733 ms) are ALWAYS bfp16 ⇒ ~1e-3 per-product
// quantization ⇒ the GDN S-feedback loop amplifies it over K=8 ⇒ explosion
// (argmax 83.6%, max|S|=50.75 vs 18.22).  There is NO exact bf16 mmul at <8,8,8>.
//
// <8,8,4,bf16,bf16> uses mac_8x8_8x4_bf16 — NATIVE bf16, NO bfp16 conversion:
// bf16×bf16→fp32-accum, exact products (upcast bf16→fp32 is exact).  This is the
// only EXACT mmul path on AIE2P.  N=4 ⇒ 4 col-groups of 8 split into two 4-col
// macs (128 macs/token vs 64 for <8,8,8>) but NO conversion overhead — net speed
// uncertain, must measure.  If fast enough AND exact, the slab could be viable.
//
// Built via: FST_GDN_SRC=kernels/fst_gdn_8kslab_mmuln4_kernel.cc \
//            FST_GDN_SUFFIX=_mmuln4 python3 kernels/gen_gdn_8kslab.py
#include <aie_api/aie.hpp>
#include <stdint.h>
using namespace aie::operators;

constexpr int HV   = 128;
constexpr int COLS = 32;
constexpr int NCV  = COLS / 8;     // 4 groups of 8 (each split into two 4-col macs)
constexpr int K    = 8;
constexpr int V    = 8;
constexpr int NG   = HV / V;       // 16
constexpr int PAR_SZ    = 2 * HV + COLS + 8;
constexpr int PAR_V     = 2 * HV;
constexpr int PAR_GDEC  = 2 * HV + COLS;
constexpr int YD_DELTA  = COLS;
constexpr int YD_SZ     = 2 * COLS;

// NATIVE bf16 mmul: A[8,8]@B[8,4]->C[8,4] fp32-accum, exact (no bfp16).
using MMUL = aie::mmul<8, 8, 4, bfloat16, bfloat16, accfloat>;

static inline void vcopy_bf16(const bfloat16 *__restrict in,
                              bfloat16 *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n)
        aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}

static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict yd) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
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

    // ── passA via NATIVE <8,8,4>: a[c]=Σ_i S[i,c]·kn[i], b[c]=Σ_i S[i,c]·qn[i] ──
    // 4 col-groups of 8; each split into two 4-col macs (half 0/1).  B = 8 rows × 4
    // cols of S, packed k-outer n-inner via SCALAR 4-copy (aie_api has no 4-elem bf16
    // vector).  C (32 fp32) scattered to a/b via a temp (no extract<4> for float).
    alignas(128) bfloat16 A_buf[64];
    alignas(128) bfloat16 B_buf[32];
    alignas(128) float cv_buf[32];
    for (int n = 0; n < NCV; ++n) {
        for (int half = 0; half < 2; ++half) {       // lo-4 / hi-4 of the 8-col group
            MMUL C;
            for (int k = 0; k < NG; ++k) {            // 16 row-groups of 8 (K-reduction)
                aie::store_v(A_buf + 0 * 8, aie::load_v<V>(kn_b + k * V));   // row 0 = kn
                aie::store_v(A_buf + 1 * 8, aie::load_v<V>(qn_b + k * V));   // row 1 = qn
                aie::vector<bfloat16, 64> A_op = aie::load_v<64>(A_buf);
                const int co = 8 * n + half * 4;                              // col offset
                for (int r = 0; r < 8; ++r) {                                 // 8 S rows -> B_buf (4 cols, k-outer)
                    const bfloat16 *sr = S + (size_t)(8 * k + r) * COLS + co;
                    B_buf[r * 4 + 0] = sr[0]; B_buf[r * 4 + 1] = sr[1];
                    B_buf[r * 4 + 2] = sr[2]; B_buf[r * 4 + 3] = sr[3];
                }
                aie::vector<bfloat16, 32> B_op = aie::load_v<32>(B_buf);
                if (k == 0) C.mul(A_op, B_op);
                else        C.mac(A_op, B_op);
            }
            aie::store_v(cv_buf, C.template to_vector<float>());            // cv[m*4+j]
            for (int j = 0; j < 4; ++j) {                                    // row 0 = a, row 1 = b
                a[8 * n + half * 4 + j] = cv_buf[j];
                b[8 * n + half * 4 + j] = cv_buf[4 + j];
            }
        }
    }

    // ── delta block (byte-identical to the working 99.5% slab; vv upcast bf16->fp32) ──
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

    // ── passB (byte-identical to the working 99.5% slab) ──
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