// fst_qwopus_ffn_n4_live2_kernel.cc — 2-LIVE-MMUL ALIGNAS PROBE (Step 1).
//
// THE LINCHPIN.  Every amortized-dequant path (M=4/M=8 dequant-once-across-nt4)
// needs TWO aie::mmul<8,8,4> accumulators (C0 for nt4=0, C1 for nt4=1) BOTH alive
// across all 32 groups — dequant runs once per (nt,g) and is reused by both
// N-tiles-of-4.  The n4v kernel avoids this by processing the 2 nt4 SEQUENTIALLY
// (1 live MMUL, dequant re-run per nt4 = 2x dequant, no amortization).  The
// 2-live version was diagnosed to crash the AIE2P stack ("a vector slot lands
// at a non-64-B-aligned offset") but the alignas fix was NEVER TRIED.
//
// aie::mmul stores its accumulator as `accum_type data` at offset 0 of the
// mmul object (aie_api/detail/mmul.hpp:408), so `alignas(128)` on the mmul
// variable aligns the spilled accumulator — this kernel tests whether that
// is enough to keep 2 live accumulators from crashing.
//
// Structure = M=1 dequant-once (h broadcast in all 8 A rows; C0/C1 each reduce
// their 4 outputs over the full K).  If this compiles + runs + argmax-MATCHes
// the host reference (up + down), the 2-live linchpin is broken and the M=8
// dequant-once kernel (Step 2) is unblocked.  It is ALSO the dequant-once M=1
// kernel: if correct, its latency should be ~the n4v 2x-dequant latency MINUS
// one dequant pass (~85ms - 23.6ms ≈ 61ms on 8 tiles) — the amortization win
// made visible at M=1.
//
// Packet geometry = M=1 (gen_qwopus_ffn.py defaults: H_BYTES=2048, 10784B < 16KB).
// Builds via: FST_FFN_SRC=...n4_live2_kernel.cc FST_FFN_FN=ffn_matvec_restream
//             FST_FFN_XCLBIN=fst_qwopus_ffn_n4_live2 FST_FFN_OBJ=...n4_live2.o
#include <aie_api/aie.hpp>
#include <stdint.h>
using namespace aie::operators;
constexpr int M_n = 16, G = 32, HDR = 32;
constexpr int H_BYTES  = G * 32 * 2;            // 2048
constexpr int SC_BYTES = M_n * G;               // 512
constexpr int NB_BYTES = M_n * G * 16;          // 8192
constexpr int N_TILE_C = 2176;
using MMUL = aie::mmul<8, 8, 4, bfloat16, bfloat16, accfloat>;

static inline void dequant_group(const uint8_t *scales, const uint8_t *nibbles,
                                 int nt, int g, bfloat16 *dq) {
    alignas(128) uint16_t tmp[32];
    for (int n = 0; n < 8; ++n) {
        const int rg = (nt * 8 + n) * G + g;
        const uint8_t sc = scales[rg];
        bfloat16 *dqrow = dq + n * 8;
        if (sc == 0 || sc == 255) {
            for (int ks = 0; ks < 4; ++ks)
                for (int kr = 0; kr < 8; ++kr) dqrow[ks * 64 + kr] = (bfloat16)0;
            continue;
        }
        const uint8_t *nb = nibbles + (size_t)((nt * 8 + n) * G + g) * 16;
        const aie::vector<uint16_t, 16> bv = aie::load_v<16>(nb).template unpack<uint16_t>();
        const aie::vector<uint16_t, 16> sign_l = (uint16_t)0x08 & bv;
        const aie::vector<uint16_t, 16> e_l    = (uint16_t)0x06 & bv;
        const aie::vector<uint16_t, 16> m_l    = (uint16_t)0x01 & bv;
        const aie::vector<uint16_t, 16> sign_h = (uint16_t)0x80 & bv;
        const aie::vector<uint16_t, 16> e_h    = (uint16_t)0x60 & bv;
        const aie::vector<uint16_t, 16> m_h    = (uint16_t)0x10 & bv;
        const uint8_t sc_eff = (sc > 252) ? (uint8_t)252 : sc;
        const uint16_t scm1_sh7 = (uint16_t)((sc_eff - 1) << 7);
        const aie::vector<uint16_t, 16> exp_l = (e_l << (unsigned)6) + scm1_sh7;
        const aie::mask<16> e_nz_l = e_l != (uint16_t)0;
        const aie::vector<uint16_t, 16> man_l = aie::select((uint16_t)0, m_l << (unsigned)6, e_nz_l);
        aie::vector<uint16_t, 16> bits_l = (sign_l << (unsigned)12) | exp_l | man_l;
        const aie::mask<16> nz_l = (e_l | m_l) != (uint16_t)0;
        bits_l = aie::select((uint16_t)0, bits_l, nz_l);
        const aie::vector<uint16_t, 16> exp_h = (e_h << (unsigned)2) + scm1_sh7;
        const aie::mask<16> e_nz_h = e_h != (uint16_t)0;
        const aie::vector<uint16_t, 16> man_h = aie::select((uint16_t)0, m_h << (unsigned)2, e_nz_h);
        aie::vector<uint16_t, 16> bits_h = (sign_h << (unsigned)8) | exp_h | man_h;
        const aie::mask<16> nz_h = (e_h | m_h) != (uint16_t)0;
        bits_h = aie::select((uint16_t)0, bits_h, nz_h);
        const aie::vector<uint16_t, 32> bits = aie::concat(aie::interleave_zip(bits_l, bits_h, 1));
        aie::store_v(tmp, bits);
        for (int ks = 0; ks < 4; ++ks)
            for (int kr = 0; kr < 8; ++kr)
                *reinterpret_cast<uint16_t *>(dqrow + ks * 64 + kr) = tmp[ks * 8 + kr];
    }
}

// M=1 dequant-once-across-nt4: TWO live MMUL accumulators (C0=nt4 0..3, C1=nt4
// 4..7), both accumulated across all 32 groups; dequant_group runs ONCE per
// (nt,g) and is read by both nt4.  h is broadcast across all 8 A-rows (M=1).
// alignas(128) on C0/C1 is the fix under test.
extern "C" void ffn_matvec_restream(const uint8_t *__restrict pkt, float *__restrict o){
    const int row_base = *reinterpret_cast<const int *>(pkt);
    const int kc       = *reinterpret_cast<const int *>(pkt + 4);
    const bfloat16 *__restrict h = reinterpret_cast<const bfloat16 *>(pkt + HDR);
    const uint8_t  *__restrict scales  = pkt + HDR + H_BYTES;
    const uint8_t  *__restrict nibbles = scales + SC_BYTES;
    ::aie::set_rounding(aie::rounding_mode::floor); event0();
    alignas(128) bfloat16 A_buf[64], B0_buf[32], B1_buf[32], dq[256];
    for (int nt = 0; nt < 2; ++nt) {
        alignas(128) MMUL C0;          // nt4=0 (outputs 0..3) — LIVE across all g
        alignas(128) MMUL C1;          // nt4=1 (outputs 4..7) — LIVE across all g
        for (int g = 0; g < G; ++g) {
            dequant_group(scales, nibbles, nt, g, dq);   // ONCE per (nt,g)
            for (int ks = 0; ks < 4; ++ks) {
                #pragma unroll
                for (int m = 0; m < 8; ++m)
                    #pragma unroll
                    for (int k = 0; k < 8; ++k)
                        A_buf[m * 8 + k] = h[g * 32 + ks * 8 + k];   // broadcast h (M=1)
                aie::vector<bfloat16, 64> A_op = aie::load_v<64>(A_buf);
                #pragma unroll
                for (int k = 0; k < 8; ++k)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        B0_buf[k * 4 + j] = dq[ks * 64 + (0 + j) * 8 + k];   // nt4=0, nbase 0
                        B1_buf[k * 4 + j] = dq[ks * 64 + (4 + j) * 8 + k];   // nt4=1, nbase 4
                    }
                aie::vector<bfloat16, 32> B0_op = aie::load_v<32>(B0_buf);
                aie::vector<bfloat16, 32> B1_op = aie::load_v<32>(B1_buf);
                if (g == 0 && ks == 0) { C0.mul(A_op, B0_op); C1.mul(A_op, B1_op); }
                else                   { C0.mac(A_op, B0_op); C1.mac(A_op, B1_op); }
            }
        }
        aie::vector<float, 32> cv0 = C0.template to_vector<float>();  // cv0[m*4+j], m=0..7 (h broadcast -> all equal)
        aie::vector<float, 32> cv1 = C1.template to_vector<float>();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v0 = cv0[j], v1 = cv1[j];               // row 0 = the matvec output (h broadcast)
            int i0 = row_base + nt * 8 + 0 + j;
            int i1 = row_base + nt * 8 + 4 + j;
            if (kc == 0) { o[i0] = v0; o[i1] = v1; } else { o[i0] += v0; o[i1] += v1; }
        }
    }
    event1();
}
extern "C" void ffn_matvec_noop_stream(const uint8_t *__restrict pkt, float *__restrict o){
    const int row_base=*reinterpret_cast<const int*>(pkt);
    const bfloat16*h=reinterpret_cast<const bfloat16*>(pkt+HDR);
    volatile float sink=0.0f;
    for(int g=0;g<G;++g){aie::vector<bfloat16,8> hv=aie::load_v<8>(h+g*32); sink=(float)hv[0]+(float)hv[4];}
    for(int r=0;r<M_n;r++) o[row_base+r]=1.0f; (void)sink;
}