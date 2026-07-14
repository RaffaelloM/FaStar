// fst_qwopus_ffn_n4v_dump_kernel.cc — dq-dump for the FIXED vectorized dequant.
#include <aie_api/aie.hpp>
#include <stdint.h>
using namespace aie::operators;
constexpr int M_n = 16, G = 32, HDR = 32;
constexpr int H_BYTES  = G * 32 * 2;
constexpr int SC_BYTES  = M_n * G;

static inline void dequant_group(const uint8_t *scales, const uint8_t *nibbles,
                                 int nt, int g, bfloat16 *dq) {
    alignas(128) uint16_t tmp[32];  // flat dequant staging: tmp[ks*8+kr]=dequant(out, g*32+ks*8+kr)
    for (int n = 0; n < 8; ++n) {
        const int rg = (nt * 8 + n) * G + g;
        const uint8_t sc = scales[rg];                       // e8m0 scale byte
        bfloat16 *dqrow = dq + n * 8;                        // dqrow[ks*64 + kr]
        if (sc == 0 || sc == 255) {                           // dead block -> 32 zeros
            for (int ks = 0; ks < 4; ++ks)
                for (int kr = 0; kr < 8; ++kr) dqrow[ks * 64 + kr] = (bfloat16)0;
            continue;
        }
        const uint8_t *nb = nibbles + (size_t)((nt * 8 + n) * G + g) * 16;  // 16-aligned
        // widen to uint16 FIRST so the upshifted field values fit (uint8<<12 overflows).
        const aie::vector<uint16_t, 16> bv =
            aie::load_v<16>(nb).template unpack<uint16_t>(); // each byte 0..255 in low 8 bits
        // low-nibble fields (bits 0-3): bit_and has only (scalar,vec) -> scalar first.
        const aie::vector<uint16_t, 16> sign_l = (uint16_t)0x08 & bv;   // bit3  (0/8)
        const aie::vector<uint16_t, 16> e_l    = (uint16_t)0x06 & bv;  // E1E0<<1 (0/2/4/6)
        const aie::vector<uint16_t, 16> m_l    = (uint16_t)0x01 & bv;   // M       (0/1)
        // high-nibble fields (bits 4-7):
        const aie::vector<uint16_t, 16> sign_h = (uint16_t)0x80 & bv;  // bit7   (0/128)
        const aie::vector<uint16_t, 16> e_h    = (uint16_t)0x60 & bv;  // E1E0<<5 (0/32/64/96)
        const aie::vector<uint16_t, 16> m_h    = (uint16_t)0x10 & bv;  // M<<4    (0/16)

        const uint8_t sc_eff = (sc > 252) ? (uint8_t)252 : sc;  // clamp (scalar)
        const uint16_t scm1_sh7 = (uint16_t)((sc_eff - 1) << 7);  // (sc-1)<<7, broadcast

        // low nibbles -> 16 bf16 bits.  exp=(sc-1)<<7 + E1E0<<7 ; E1E0<<7 = e_l<<6.
        const aie::vector<uint16_t, 16> exp_l = (e_l << (unsigned)6) + scm1_sh7; // vec+scalar
        const aie::mask<16> e_nz_l = e_l != (uint16_t)0;       // E1E0>0 (subnormal clears man)
        const aie::vector<uint16_t, 16> man_l =
            aie::select((uint16_t)0, m_l << (unsigned)6, e_nz_l); // e_nz ? (m_l<<6) : 0
        aie::vector<uint16_t, 16> bits_l =
            (sign_l << (unsigned)12) | exp_l | man_l;
        const aie::mask<16> nz_l = (e_l | m_l) != (uint16_t)0;  // zero-nibble mask
        bits_l = aie::select((uint16_t)0, bits_l, nz_l);        // nz_l ? bits_l : 0

        // high nibbles -> 16 bf16 bits.  E1E0<<7 = e_h<<2 ; M<<6 = m_h<<2.
        const aie::vector<uint16_t, 16> exp_h = (e_h << (unsigned)2) + scm1_sh7;
        const aie::mask<16> e_nz_h = e_h != (uint16_t)0;
        const aie::vector<uint16_t, 16> man_h =
            aie::select((uint16_t)0, m_h << (unsigned)2, e_nz_h);
        aie::vector<uint16_t, 16> bits_h =
            (sign_h << (unsigned)8) | exp_h | man_h;
        const aie::mask<16> nz_h = (e_h | m_h) != (uint16_t)0;
        bits_h = aie::select((uint16_t)0, bits_h, nz_h);

        // interleave: bits[2i]=bits_l[i] (k=2i, low nibble), bits[2i+1]=bits_h[i] (k=2i+1).
        const aie::vector<uint16_t, 32> bits =
            aie::concat(aie::interleave_zip(bits_l, bits_h, 1));   // [k0,k1,..,k31]

        // store the uint16 bits contiguously to a flat stack temp (single store_v of
        // the 32-elem `bits` vector — NO extract<8>, which was the staging bug: the 4
        // extract<8>(8/16/24) strided stores all wrote k=0..7, losing k=8..31).  Then a
        // scalar strided copy lays tmp[ks*8+kr] into dq[ks*64 + n*8 + kr] (the layout the
        // <8,8,4> B-gather reads).  Uses only proven store_v + scalar paths.
        aie::store_v(tmp, bits);
        for (int ks = 0; ks < 4; ++ks)
            for (int kr = 0; kr < 8; ++kr)
                *reinterpret_cast<uint16_t *>(dqrow + ks * 64 + kr) = tmp[ks * 8 + kr];
    }
}

// Re-stream-h NATIVE-<8,8,4> matvec.  Dequant is the Phase 4a vectorized bitwise
// path (unchanged, bit-exact); the MMUL is now native exact bf16 (no bfp16).
//
// Layout (vectorized dequant): per (dequant-tile nt, group g) the 8 outputs ×
// 32 k are dequanted ONCE into dq[ks*64 + n*8 + k_rel] = dequant(output_n,
// ks*8+k_rel) (n=0..7 of the nt tile).  Each native <8,8,4> N-tile-of-4 gathers
// 4 of those 8 outputs into B_buf[k*4 + j] = dq[ks*64 + (nbase+j)*8 + k]
// (scalar 4-copy — aie_api has no 4-elem bf16 vector) for j=0..3, k=0..7, the
// exact B[k,j] = dequant(output_{nbase+j}, K_idx+k) the GDN mmuln4 kernel uses.
//   A[m,k] = h[K_idx+k] for all m (broadcast M=8 — 7 M-rows wasted, the r=8 tile
//     minimum for an M=1 matvec).
//   C.mac(A,B) => C[0,j] = sum_k h[K_idx+k]*dq(W[output_{nbase+j},K_idx+k]) = partial[j].
// 2 dequant tiles (nt 0,1) × 2 N-tiles-of-4 (nt4 0,1) = 16 outputs/packet.
// The 2 N-tiles-of-4 are processed SEQUENTIALLY (1 live MMUL — the 2-live-MMUL
// stack crash: a vector slot lands at a non-64-B-aligned offset); dequant is
// re-run per nt4 (2× dequant, acceptable for the M=1 correctness baseline; the
// M=4 kernel amortizes the dequant once across 4 h-vectors).
// Held output: kc==0 => o[..]=partial, else o[..]+=partial (RMW across k_chunks).

extern "C" void ffn_matvec_restream(const uint8_t *__restrict pkt, float *__restrict o) {
    const int row_base = *reinterpret_cast<const int *>(pkt);
    const int kc       = *reinterpret_cast<const int *>(pkt + 4);
    const uint8_t  *__restrict scales  = pkt + HDR + H_BYTES;
    const uint8_t  *__restrict nibbles = scales + SC_BYTES;
    if (row_base == 0 && kc == 0) {
        alignas(128) bfloat16 dq[256];
        dequant_group(scales, nibbles, 0, 0, dq);
        for (int i = 0; i < 256; ++i) o[i] = (float)dq[i];
    }
}
extern "C" void ffn_matvec_noop_stream(const uint8_t *__restrict pkt, float *__restrict o){(void)pkt;(void)o;}
