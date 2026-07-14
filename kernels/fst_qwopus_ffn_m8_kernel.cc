// fst_qwopus_ffn_m8_kernel.cc — M=8 SPECULATIVE-DECODING FFN, 32-tile N-split.
//
// GOAL: beat host M=8 FFN (≈85.6 ms, 8 h-vectors, OpenMP mxfp4_matvec_f32).  Two
// structural moves vs the dead M=4 line:
//   (1) NT=16 (the NPU2 has 32 compute tiles but only 8 shims, each with 2 MM2S +
//       2 S2MM DMA channels => 16 input + 16 output channels = 16-tile ceiling
//       for a 1-fifo-in + 1-fifo-out-per-tile design; NT=32 does NOT place — "no
//       ShimNOCTile has sufficient DMA capacity").  N_TILE = 17408/16 = 1088;
//       M=8 held output = 8×1088×4 = 34.8 KB (fits a 64 KB tile; M=8 on NT=8 was
//       69 KB = the wall).  2× the N-parallelism of the 8-tile kernels.
//   (2) G_pkt=16 (not 32) so the M=8 packet = 32 + 8×(16×32×2) + 16×16×17 =
//       12576 B < 16 KB — stays on the fast DMA side of the packet-buffer wall
//       (≥16 KB input buffers the core reads drop to 0.29 GB/s).  G_pkt=16 is the
//       only G that divides BOTH up GROUPS=160 and down GROUPS=544 (GCD=32) and
//       keeps the packet under 16 KB.
//
// ARCHITECTURE: 2×-DEQUANT, 1-LIVE MMUL (the n4v structure), 8 h-vectors in the
// 8 A-rows of <8,8,4>.  This is DELIBERATELY NOT dequant-once/2-live: the
// 2-live-MMUL alignas probe (fst_qwopus_ffn_n4_live2) PROVED dequant-once is
// CORRECT but 7 ms SLOWER than 2×-dequant (91.93 vs 84.99 ms, same 8-tile
// geometry) — the 2-live-MMUL overhead exceeds the saved dequant pass.  So the
// dequant-amortization premise is FALSE on AIE2P; M=8 takes its win from NT=16
// parallelism (2×), not dequant amortization.  The 8 h-vectors ride in the 8
// A-rows FREE (same MMUL count as M=1 — A[8,8] holds 8 h-rows × 8 k).
//
// Projected (8-tile M=1 n4v = 85 ms; NT=16 = 2× parallelism): M=8 ≈ 85/2 ≈ 42.5
// ms full for 8 tokens = 5.3 ms/h vs host 10.7 ms/h → ~2× win.  (NT=32 would be
// 4× / ~21 ms but does not place — the 16-channel shim DMA ceiling.)
//
// MMUL mapping (native <8,8,4>, A[8,8]@B[8,4]->C[8,4] fp32-acc, exact):
//   A[m,k] = h_m[g*32 + ks*8 + k]   (m=0..7 = the 8 h-vectors; k=0..7)
//   B[k,j] = dq(ks*64 + (nbase+j)*8 + k)   (nbase = nt4*4; j=0..3)
//   C[m*4+j] = sum_k h_m[k]*dq(W[output_{nbase+j}, k]) = output for h_m, output_{nbase+j}
// 2 dequant tiles (nt 0,1; 8 outputs each) × 2 N-tiles-of-4 (nt4 0,1) = 16
// n-outputs/packet; × 8 h-rows = 128 h-output values/packet.  G_pkt=16 groups
// per packet; K-reduction across G×4×K=512 k per packet (full K=5120 across
// KCHUNKS=10).  Held output o[m*N_TILE_C + row_base + nt*8 + nbase + j],
// kc==0 => store, else += (RMW across k_chunks).
#include <aie_api/aie.hpp>
#include <stdint.h>
using namespace aie::operators;
constexpr int M_n = 16, MH = 8, G = 16, HDR = 32;          // G = G_pkt = 16
constexpr int H_BYTES   = MH * G * 32 * 2;                 // 8192 (8 bf16 h-vectors)
constexpr int SC_BYTES = M_n * G;                           // 256
constexpr int NB_BYTES = M_n * G * 16;                      // 4096
constexpr int N_TILE_C = 1088;                              // held output width per h-vector (NT=16)
using MMUL = aie::mmul<8, 8, 4, bfloat16, bfloat16, accfloat>;   // mac_8x8_8x4_bf16 (exact, fast)

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
    // n4v structure (1 live MMUL, 2x dequant): nt4 OUTSIDE g, C per nt4 accumulates
    // all G=16 groups.  8 h-vectors in A rows 0..7 (the <8,8,4> A is 8x8 = 8 h-rows
    // x 8 k — all 8 h's used, no waste, same MMUL count as M=1).
    for (int nt = 0; nt < 2; ++nt) {
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