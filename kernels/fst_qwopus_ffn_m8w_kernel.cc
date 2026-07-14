// fst_qwopus_ffn_m8w_kernel.cc — M=8 FFN, 16-tile N-split, WIDE packet (M_n=32).
//
// Same as fst_qwopus_ffn_m8_kernel.cc but M_n=32 outputs/packet (4 dequant
// tiles nt 0..3, not 2) -> NCHUNKS=34 (was 68), NPKT=340/tile (was 680): HALF
// the packets.  The 16-tile DMA floor is per-packet-dispatch-overhead-bound
// (the M=16 variant was DMA-bound at 48 ms, 2.81 GB/s, 10x under the 8-tile's
// 30 GB/s); halving the packets (and total bytes 137->92 MB) targets ~half the
// floor.  Packet = 32 + 8192 (8 h's, G_pkt=16) + 32*16*17=8704 = 16928 B (the
// M=4-proven safe size, just over 16 KB but DMA-safe).  Compute is unchanged
// (same total MMULs + dequant; 4 nt x 2 nt4 x G x 4 = 512 MMUL/packet x 340
// packets vs 256 x 680 — identical).  Held output 8x1088x4 = 34.8 KB (NT=16).
//
// If the DMA floor halves (~24 ms) and compute stays ~42.5 ms, the kernel goes
// compute-bound at ~42.5 ms (2x over host M=8 85.6 ms) instead of DMA-bound at
// 48 ms (1.76x).
#include <aie_api/aie.hpp>
#include <stdint.h>
using namespace aie::operators;
constexpr int M_n = 32, MH = 8, G = 16, HDR = 32;          // M_n=32 (wide), G_pkt=16
constexpr int H_BYTES   = MH * G * 32 * 2;                 // 8192 (8 bf16 h-vectors)
constexpr int SC_BYTES = M_n * G;                           // 512
constexpr int NB_BYTES = M_n * G * 16;                      // 8192
constexpr int N_TILE_C = 1088;                              // held output width per h-vector (NT=16)
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

extern "C" void ffn_matvec_restream(const uint8_t *__restrict pkt, float *__restrict o){
    const int row_base = *reinterpret_cast<const int *>(pkt);
    const int kc       = *reinterpret_cast<const int *>(pkt + 4);
    const bfloat16 *__restrict h0 = reinterpret_cast<const bfloat16 *>(pkt + HDR);
    const uint8_t  *__restrict scales  = pkt + HDR + H_BYTES;
    const uint8_t  *__restrict nibbles = scales + SC_BYTES;
    ::aie::set_rounding(aie::rounding_mode::floor); event0();
    alignas(128) bfloat16 A_buf[64], B_buf[32], dq[256];
    for (int nt = 0; nt < 4; ++nt) {                 // 4 dequant tiles (M_n=32 outputs)
        for (int nt4 = 0; nt4 < 2; ++nt4) {
            const int nbase = nt4 * 4;
            MMUL C;
            for (int g = 0; g < G; ++g) {
                dequant_group(scales, nibbles, nt, g, dq);   // 2x dequant (per nt4)
                for (int ks = 0; ks < 4; ++ks) {
                    #pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        const bfloat16 *__restrict hm = h0 + (size_t)m * (G * 32);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) A_buf[m * 8 + k] = hm[g * 32 + ks * 8 + k];
                    }
                    aie::vector<bfloat16, 64> A_op = aie::load_v<64>(A_buf);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) B_buf[k * 4 + j] = dq[ks * 64 + (nbase + j) * 8 + k];
                    aie::vector<bfloat16, 32> B_op = aie::load_v<32>(B_buf);
                    if (g == 0 && ks == 0) C.mul(A_op, B_op); else C.mac(A_op, B_op);
                }
            }
            aie::vector<float, 32> cv = C.template to_vector<float>();  // cv[m*4+j], m=0..7
            #pragma unroll
            for (int m = 0; m < MH; ++m)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v = cv[m * 4 + j];
                    int idx = row_base + nt * 8 + nbase + j;
                    size_t off = (size_t)m * N_TILE_C + idx;
                    if (kc == 0) o[off] = v; else o[off] += v;
                }
        }
    }
    event1();
}
extern "C" void ffn_matvec_noop_stream(const uint8_t *__restrict pkt, float *__restrict o){
    const int row_base=*reinterpret_cast<const int*>(pkt);
    const bfloat16*h=reinterpret_cast<const bfloat16*>(pkt+HDR);
    volatile float sink=0.0f;
    for(int g=0;g<G;++g) for(int m=0;m<MH;++m){aie::vector<bfloat16,8> hv=aie::load_v<8>(h+(size_t)m*(G*32)+g*32); sink=(float)hv[0]+(float)hv[4];}
    for(int r=0;r<M_n;r++) o[row_base+r]=1.0f; (void)sink;
}