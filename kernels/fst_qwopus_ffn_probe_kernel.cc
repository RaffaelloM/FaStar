// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_qwopus_ffn_probe_kernel.cc — Stage-1.1 micro-probe for the NPU FFN lever.
//
// Purpose: measure the ONE make-or-break unknown for moving Qwopus's M=1 SwiGLU
// FFN (40% of the token, currently host at ~4.7 GB/s) onto the NPU: the NPU's
// sustained AGGREGATE weight-read bandwidth with large BDs + a 20 KB read-only
// held input vector h (the same on-tile-array class that killed chunkwise GDN,
// but read-only here, not RMW).  If one tile sustains >= ~1.5 GB/s, 8 tiles give
// >= ~12 GB/s aggregate -> FFN could beat host.  If ~200 MB/s/tile -> dead-end.
//
// This is a REAL gate-projection matvec (out[n] = sum_k W[n,k]·h[k], W MXFP4),
// so the probe also validates the matvec kernel's correctness vs the host fp32
// reference.  gate/up/down are all matvecs of this shape (different N,K).
//
// Layout (host packs, 17-byte .fst block padded to 18 for 8-aligned stride):
//   wrow : 160 blocks × 18 B = 2880 B/row.  block = [e8m0 scale | 16 nibble bytes]
//   h    : K=5120 fp32 (held, read across all rows).
//   out  : one fp32 per row.
//
// Math matches mxfp4_matvec_f32 (host): for each group g (32 elems), scale =
// ldexp(1, sc-127), acc += FP4[nib] * scale * h[...], fp32 accumulate.  The
// inner 8-element MAC is vectorised (8-lane mul + reduce_add); 4-bit nibble
// unpack is scalar (unavoidable, same as the proven fused-dequant kernel).

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV_K      = 5120;          // K (hidden)
constexpr int GROUPS    = HV_K / 32;     // 160 MXFP4 blocks per row
constexpr int PAD_BYTES = 18;            // 17-byte .fst block + 1 pad (8-aligned stride)
constexpr int ROW_BYTES = GROUPS * PAD_BYTES;   // 2880
constexpr int V         = 8;             // fp32 lanes

// FP4 (E2M1) -> fp32 lookup (matches host FP4_TABLE).
static const float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// e8m0 (8-bit bias-127 exponent) -> fp32 scale = ldexp(1, e-127).  Bit-identical
// to the host's std::ldexp(1.0f, (int)sc - 127) for sc in [1,254]; sc==0 -> 0.
static inline float e8m0_to_f32(uint8_t e) {
    if (e == 0 || e == 255) return 0.0f;
    union { uint32_t u; float f; } cvt;
    cvt.u = ((uint32_t)e) << 23;     // fp32 exponent field, bias 127, mantissa 0
    return cvt.f;
}

// Process RPB rows from one weight packet.  wblk_f carries the raw MXFP4 bytes
// reinterpreted as float (the IRON uint8-as-2nd-input contract: a uint8 fifo
// alongside a float fifo does NOT DMA, so the weight stream is typed float and
// reinterpreted here).  h is the held input vector (pointer to a depth-1
// ObjectFifo element held across all row packets).  out points at this batch's
// slot in the held output buffer.  rpb is a runtime arg so the generator can
// reuse one kernel for the final partial batch.
extern "C" void ffn_matvec_rows(
    const float    *__restrict wblk_f, // rpb × ROW_BYTES bytes, as float
    const float    *__restrict h,      // HV_K
    float          *__restrict out,    // rpb
    int              rpb)
{
    const uint8_t *wblk = reinterpret_cast<const uint8_t *>(wblk_f);
    for (int r = 0; r < rpb; ++r) {
        const uint8_t *row = wblk + (size_t)r * ROW_BYTES;
        float acc = 0.0f;
        for (int g = 0; g < GROUPS; ++g) {
            const uint8_t *blk = row + (size_t)g * PAD_BYTES;
            const float scale = e8m0_to_f32(blk[0]);
            const uint8_t *nb = blk + 1;
            const float   *hb = h + g * 32;
            for (int i4 = 0; i4 < 4; ++i4) {        // 4 × 8 elems = 32 / group
                const uint8_t b0 = nb[i4 * 4 + 0];
                const uint8_t b1 = nb[i4 * 4 + 1];
                const uint8_t b2 = nb[i4 * 4 + 2];
                const uint8_t b3 = nb[i4 * 4 + 3];
                float wv[8] = {
                    FP4_LUT[b0 & 0xF]       * scale, FP4_LUT[(b0 >> 4) & 0xF] * scale,
                    FP4_LUT[b1 & 0xF]       * scale, FP4_LUT[(b1 >> 4) & 0xF] * scale,
                    FP4_LUT[b2 & 0xF]       * scale, FP4_LUT[(b2 >> 4) & 0xF] * scale,
                    FP4_LUT[b3 & 0xF]       * scale, FP4_LUT[(b3 >> 4) & 0xF] * scale,
                };
                aie::vector<float, V> wvec = aie::load_v<V>(wv);
                aie::vector<float, V> hvec = aie::load_v<V>(hb + i4 * 8);
                acc += aie::reduce_add(aie::mul(wvec, hvec).template to_vector<float>());
            }
        }
        out[r] = acc;
    }
}
// No-op diagnostic: stream weight packets, drain a constant per row.  NO held h,
// NO compute on the weight bytes (just a vector load to force the DMA read).
// Isolates the weight-stream DMA rate from the 20 KB held-h array access.
extern "C" void ffn_noop_rows(
    const float    *__restrict wblk_f,   // rpb × ROW_BYTES bytes, as float
    const float    *__restrict h,        // unused (kept for arg-layout compat)
    float          *__restrict out,      // rpb
    int              rpb)
{
    const uint8_t *wblk = reinterpret_cast<const uint8_t *>(wblk_f);
    (void)h;
    // Touch each row with one vector load so the DMA read is not elided.
    volatile float sink = 0.0f;
    for (int r = 0; r < rpb; ++r) {
        const float *p = reinterpret_cast<const float *>(wblk + (size_t)r * ROW_BYTES);
        aie::vector<float, V> v = aie::load_v<V>(p);
        sink = aie::reduce_add(v);
        out[r] = 1.0f;
    }
    (void)sink;
}
