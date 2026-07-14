// fst_gdn_8kslab_mmulfp_kernel.cc — Step-2: passA via FP32 aie::mmul.
//
// The bf16 mmul variants (_mmul / _mmulnat) recovered latency (733 ms vs 1180 ms
// scalar) but EXPLODED (argmax 83.6%, max|S|=50.75 vs ref 18.22): on AIE2P every
// aie::mmul<bfloat16> is BFP16-emulated (block-floating-point, shared exponent
// per 8-block) — the ~1e-3 per-product quantization is fine for a NON-recurrent
// FFN but the GDN S-feedback loop amplifies that bias over K=8 tokens.  The
// recurrence demands FP32-EXACT products.
//
// fp32 mmul (aie::mmul<8,8,8,float,float>) does fp32×fp32→fp32-accum — bit-identical
// to the scalar fp32 path (products exact, only sum-order/accum-precision differ
// trivially).  This tests whether MMUL's vector-load structure (no per-row scalar
// broadcast) recovers the ~446 ms the nomul canary attributed to the scalar-broadcast
// MAC stall, WHILE keeping the recurrence exact.  passA batches kn+qn as M=2 (A
// row0=kn, row1=qn).  delta/passB are byte-identical to the working 99.5% slab
// (fp32 stack kn/qn/vv), so any divergence isolates to passA.
//
// Built via: FST_GDN_SRC=kernels/fst_gdn_8kslab_mmulfp_kernel.cc \
//            FST_GDN_SUFFIX=_mmulfp python3 kernels/gen_gdn_8kslab.py
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

// fp32 mmul: A[8,8]@B[8,8]->C[8,8] fp32-accum (exact products). accfloat accum.
using MMUL = aie::mmul<8, 8, 8, float, float, accfloat>;

static inline void vcopy_bf16(const bfloat16 *__restrict in,
                              bfloat16 *__restrict out, int nvec) {
    for (int n = 0; n < nvec; ++n)
        aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}

static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict yd) {
    const aie::vector<bfloat16, V> ones_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    // kn/qn/vv as FP32 stack (exact upcast of the bf16 DMA values) — same as the
    // working 99.5% slab.  A operand of the fp32 mmul reads these directly.
    float kn[HV]     __attribute__((aligned(32)));
    float qn[HV]     __attribute__((aligned(32)));
    float vv[COLS]   __attribute__((aligned(32)));
    float a[COLS]    __attribute__((aligned(32)));
    float b[COLS]    __attribute__((aligned(32)));
    float delta[COLS] __attribute__((aligned(32)));
    for (int n = 0; n < NG;  ++n)
        aie::store_v(kn + n * V, aie::mul(aie::load_v<V>(par + n * V), ones_bf).template to_vector<float>());
    for (int n = 0; n < NG;  ++n)
        aie::store_v(qn + n * V, aie::mul(aie::load_v<V>(par + HV + n * V), ones_bf).template to_vector<float>());
    for (int n = 0; n < NCV; ++n)
        aie::store_v(vv + n * V, aie::mul(aie::load_v<V>(par + PAR_V + n * V), ones_bf).template to_vector<float>());
    aie::vector<float, V> tailf =
        aie::mul(aie::load_v<V>(par + PAR_GDEC), ones_bf).template to_vector<float>();
    const float gdec = tailf[0], beta = tailf[1], c = tailf[2];

    // ── passA via FP32 MMUL<8,8,8>: a[c]=Σ_i S[i,c]·kn[i], b[c]=Σ_i S[i,c]·qn[i] ──
    // A row 0 = kn chunk, row 1 = qn chunk (fp32); rows 2-7 garbage (C rows 2-7
    // unused).  B = upcast S[8k..8k+7, 8n..8n+7] -> B_buf_f n-fast, then transposed.
    alignas(128) float A_buf[64];
    alignas(128) float B_buf[64];
    for (int n = 0; n < NCV; ++n) {            // 4 col-groups of 8
        MMUL C;
        for (int k = 0; k < NG; ++k) {         // 16 row-groups of 8 (K-reduction)
            aie::store_v(A_buf + 0 * 8, aie::load_v<V>(kn + k * V));   // row 0 = kn (fp32)
            aie::store_v(A_buf + 1 * 8, aie::load_v<V>(qn + k * V));   // row 1 = qn (fp32)
            aie::vector<float, 64> A_op = aie::load_v<64>(A_buf);
            for (int r = 0; r < 8; ++r)        // upcast 8 S rows (bf16->fp32) -> B_buf n-fast
                aie::store_v(B_buf + r * 8,
                    aie::mul(aie::load_v<V>(S + (size_t)(8 * k + r) * COLS + 8 * n), ones_bf).template to_vector<float>());
            aie::vector<float, 64> B_op = aie::transpose(aie::load_v<64>(B_buf), 8, 8);
            if (k == 0) C.mul(A_op, B_op);
            else        C.mac(A_op, B_op);
        }
        aie::vector<float, 64> cv = C.template to_vector<float>();       // cv[m*8+j] = C[m,j]
        aie::store_v(a + n * V, cv.template extract<V>(0));              // row 0 = a
        aie::store_v(b + n * V, cv.template extract<V>(1));              // row 1 = b
    }

    // ── delta block (byte-identical to the working 99.5% slab) ──
    for (int n = 0; n < NCV; ++n) {
        aie::vector<float, V> av = aie::load_v<V>(a + n * V);
        aie::vector<float, V> bv = aie::load_v<V>(b + n * V);
        aie::vector<float, V> vvi = aie::load_v<V>(vv + n * V);
        aie::vector<float, V> kvm = aie::mul(av, gdec).template to_vector<float>();
        aie::accum<accfloat, V> dv_acc = aie::mul(aie::sub(vvi, kvm), beta);
        aie::vector<float, V> dv = dv_acc.template to_vector<float>();
        aie::store_v(delta + n * V, dv);
        aie::accum<accfloat, V> yv = aie::add(aie::mul(bv, gdec), aie::mul(dv, c));
        aie::store_v(yd + n * V, yv.template to_vector<bfloat16>());
        aie::store_v(yd + YD_DELTA + n * V, dv_acc.template to_vector<bfloat16>());
    }

    // ── passB (byte-identical to the working 99.5% slab) ──
    for (int i = 0; i < HV; ++i) {
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

extern "C" void gdn_8kslab_vhead(const bfloat16 *__restrict s0,  const bfloat16 *__restrict par,
                                 bfloat16 *__restrict snew,      bfloat16 *__restrict yd) {
    bfloat16 S[HV * COLS] __attribute__((aligned(32)));
    for (int i = 0; i < HV; ++i) vcopy_bf16(s0 + (size_t)i * COLS, S + (size_t)i * COLS, NCV);
    for (int t = 0; t < K; ++t) recur_core(S, par + (size_t)t * PAR_SZ, yd + (size_t)t * YD_SZ);
    for (int i = 0; i < HV; ++i) vcopy_bf16(S + (size_t)i * COLS, snew + (size_t)i * COLS, NCV);
}