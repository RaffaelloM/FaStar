// fst_qwopus_ffn_n4_dump_kernel.cc — dequant DUMP diagnostic.
// Same dequant_group as fst_qwopus_ffn_n4_kernel.cc (the suspect).  On the FIRST
// packet of each tile (row_base==0 && kc==0) it dequants nt=0, g=0 into dq[0..255]
// and writes dq to o[0..255]; all other packets do nothing (preserve the dump).
// Read back tile 0's o[0..255] (= dq[ks*64+n*8+kr] for global outputs 0..7, group 0)
// and compare to the host scalar LUT dequant => pinpoints the exact vector-op bug.
#include <aie_api/aie.hpp>
#include <stdint.h>
using namespace aie::operators;

constexpr int M_n = 16, G = 32, HDR = 32;
constexpr int H_BYTES  = G * 32 * 2;
constexpr int SC_BYTES  = M_n * G;
constexpr int NB_BYTES  = M_n * G * 16;

static inline void dequant_group(const uint8_t *scales, const uint8_t *nibbles,
                                 int nt, int g, bfloat16 *dq) {
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
        const aie::vector<uint16_t, 16> bv =
            aie::load_v<16>(nb).template unpack<uint16_t>();
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
        const aie::vector<uint16_t, 16> man_l =
            aie::select((uint16_t)0, m_l << (unsigned)6, e_nz_l);
        aie::vector<uint16_t, 16> bits_l =
            (sign_l << (unsigned)12) | exp_l | man_l;
        const aie::mask<16> nz_l = (e_l | m_l) != (uint16_t)0;
        bits_l = aie::select((uint16_t)0, bits_l, nz_l);
        const aie::vector<uint16_t, 16> exp_h = (e_h << (unsigned)2) + scm1_sh7;
        const aie::mask<16> e_nz_h = e_h != (uint16_t)0;
        const aie::vector<uint16_t, 16> man_h =
            aie::select((uint16_t)0, m_h << (unsigned)2, e_nz_h);
        aie::vector<uint16_t, 16> bits_h =
            (sign_h << (unsigned)8) | exp_h | man_h;
        const aie::mask<16> nz_h = (e_h | m_h) != (uint16_t)0;
        bits_h = aie::select((uint16_t)0, bits_h, nz_h);
        const aie::vector<uint16_t, 32> bits =
            aie::concat(aie::interleave_zip(bits_l, bits_h, 1));
        uint16_t *dqrow_u = reinterpret_cast<uint16_t *>(dqrow);
        aie::store_unaligned_v(dqrow_u + 0 * 64, bits.template extract<8>(0));
        aie::store_unaligned_v(dqrow_u + 1 * 64, bits.template extract<8>(8));
        aie::store_unaligned_v(dqrow_u + 2 * 64, bits.template extract<8>(16));
        aie::store_unaligned_v(dqrow_u + 3 * 64, bits.template extract<8>(24));
    }
}

extern "C" void ffn_matvec_restream(const uint8_t *__restrict pkt, float *__restrict o) {
    const int row_base = *reinterpret_cast<const int *>(pkt);
    const int kc       = *reinterpret_cast<const int *>(pkt + 4);
    const uint8_t  *__restrict scales  = pkt + HDR + H_BYTES;
    const uint8_t  *__restrict nibbles = scales + SC_BYTES;
    if (row_base == 0 && kc == 0) {
        alignas(128) bfloat16 dq[256];
        dequant_group(scales, nibbles, 0, 0, dq);   // nt=0, g=0
        for (int i = 0; i < 256; ++i) o[i] = (float)dq[i];
    }
    // else: do nothing (preserve the first packet's dump in o[0..255])
}

extern "C" void ffn_matvec_noop_stream(const uint8_t *__restrict pkt, float *__restrict o) {
    (void)pkt; (void)o;
}