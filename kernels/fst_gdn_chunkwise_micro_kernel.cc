// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_gdn_chunkwise_micro_kernel.cc — Stage 1.1 MICRO-TEST for the chunkwise
// M=K GDN kernel.  Resolves the #1 risk cheaply: is a stack-local
// bfloat16 Sbuf[128*128] (32 KB) RMW'd with aie::load_v / aie::store_v FAST
// (~30 ms dispatch floor, NOT the 60 s the M=1 aie.iron.Buffer RMW hit) and
// NUMERICALLY CLEAN (load_v/store_v on a stack bf16 array returns real values,
// not the garbage that static .bss produces — see static-bss-incoherent-with-
// loadv)?
//
// The real chunkwise kernel holds the GDN state S[128,128] ON-TILE as bf16
// (fp32 = 64 KB fills the whole AIE2P tile) and RMWs it across K=8 recurrence
// steps in ONE dispatch.  This micro-test isolates exactly that mechanism:
//
//   - stack bfloat16 Sbuf[128*128];            // 32 KB, core-private (NOT a
//                                              //   DMA channel — that is the
//                                              //   point: avoids the 2-S2MM
//                                              //   race that killed M=1 fusion)
//   - fill   : row r <- bf16(r+1)              // fp32->bf16->store_v (stack)
//   - RMW x8 : Sbuf = gdec*Sbuf, gdec=0.5      // load_v bf16, mul, store_v bf16
//   - checksum: sum all Sbuf as fp32           // load_v bf16, widen, reduce
//
// CHOICE OF gdec=0.5 + INTEGER fill (bit-exact in bf16): makes the RMW
// rounding-mode-INDEPENDENT.  The AIE accfloat->bf16 conversion may truncate or
// round-to-nearest — we do not want the bit-clean check to depend on which.
// Integers 1..128 are exact in bf16, and 0.5^k is exact (only the exponent
// changes), so the whole RMW is bit-exact in bf16 regardless of AIE rounding.
// The host reference is then exactly:
//     sum = 128 * sum_{r=1..128} (r * 0.5^8) = 128 * (128*129/2)/256 = 4128.0
// If load_v/store_v on a stack bf16 array returned garbage (the .bss failure
// mode), the checksum would not be 4128.0.
//
// The checksum depends on EVERY element of Sbuf, so the K=8 RMW over all 16K
// elements is observable and cannot be dead-code-eliminated — the dispatch
// latency therefore genuinely reflects stack-bf16 RMW cost (the speed test).
//
// DMA stays fp32 (bf16 ONLY on-tile, converted in the kernel) — matches the
// real kernel's plan and reuses the proven fp32 ObjectFifo path.
//
// One C call handles ONE v-head (one 32 KB stack frame, reused across the 48
// v-heads the IRON Worker loops).  Input packet = [gdec(1)|pad(7)] = 8 fp32;
// output packet = [checksum(1)|pad(7)] = 8 fp32.

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV   = 128;
constexpr int VB   = 16;          // bf16 native lanes (AIE2P)
constexpr int NVB  = HV / VB;      // 8 bf16-vecs per 128-element row
constexpr int KST  = 8;           // recurrence steps (K)

extern "C" void gdn_micro_vhead(const float *__restrict in,   // [8] fp32
                                 float *__restrict out)        // [8] fp32
{
    const float gdec = in[0];   // host passes 0.5 (bit-exact in bf16)

    // ── stack-local bf16 state — the thing under test (32 KB).  Stack locals
    //    are permitted by the static-bss rule; only static .bss is forbidden.
    bfloat16 Sbuf[HV * HV];
    // Access Sbuf through an OPAQUE register pointer so load_v/store_v use
    // register+register addressing, NOT frame-pointer immediate offsets (the
    // AIE2P load/store immediate is limited to [-32768, -64]; a 32 KB stack
    // array + ~128 B frame-save overhead would otherwise overflow it at -32896).
    // The inline-asm materialization defeats Peano's SROA/alias folding back to
    // a frame offset — S is an opaque register, &S[0] is computed once via an
    // ADD with a materialized 32-bit immediate (legal), and every load_v/store_v
    // is S + small_offset (register+register, no 32 KB limit).
    bfloat16 *__restrict S;
    __asm__ __volatile__("" : "=r"(S) : "0"(Sbuf));

    // ── fill: row r <- bf16(r+1).  fp32 scalar -> bf16 scalar -> broadcast ->
    //    store_v into the stack array.  Tests the fp32->bf16->stack-store path
    //    the real kernel uses for S0 ingest.  Integers 1..128 are exact in bf16
    //    so the conversion has no rounding ambiguity.
    for (int r = 0; r < HV; ++r) {
        bfloat16 *row = S + r * HV;
        bfloat16 val_bf = (bfloat16)(float)(r + 1);
        aie::vector<bfloat16, VB> bv = aie::broadcast<bfloat16, VB>(val_bf);
        for (int n = 0; n < NVB; ++n)
            aie::store_v(row + n * VB, bv);
    }

    // ── RMW K=8 steps: Sbuf = gdec*Sbuf.  load_v bf16, mul by broadcast gdec
    //    (bf16 x bf16 -> accfloat), to_vector<bfloat16> (truncate/round), store_v.
    //    With gdec=0.5 this is bit-exact in bf16 (exponent only).  This is the
    //    hot loop the speed test measures: 8 x 128 x 8 = 8192 load_v/store_v
    //    pairs over the 32 KB stack array.
    {
        const bfloat16 gdec_bf = (bfloat16)gdec;
        aie::vector<bfloat16, VB> gv = aie::broadcast<bfloat16, VB>(gdec_bf);
        for (int k = 0; k < KST; ++k) {
            for (int r = 0; r < HV; ++r) {
                bfloat16 *row = S + r * HV;
                for (int n = 0; n < NVB; ++n) {
                    aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
                    aie::accum<accfloat, VB> acc = aie::mul(sv, gv);
                    aie::vector<bfloat16, VB> rv = acc.to_vector<bfloat16>();
                    aie::store_v(row + n * VB, rv);
                }
            }
        }
    }

    // ── checksum: sum all Sbuf elements as fp32.  load_v bf16, widen to fp32
    //    (bf16 x 1.0 -> accfloat -> float), reduce_add per vec, accumulate.
    //    Depends on every element -> forces the whole 32 KB RMW to be live.
    float sum = 0.0f;
    {
        aie::vector<bfloat16, VB> ones = aie::broadcast<bfloat16, VB>((bfloat16)1.0f);
        for (int r = 0; r < HV; ++r) {
            bfloat16 *row = S + r * HV;
            for (int n = 0; n < NVB; ++n) {
                aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
                aie::accum<accfloat, VB> acc = aie::mul(sv, ones);
                aie::vector<float, VB> fv = acc.to_vector<float>();
                sum += aie::reduce_add(fv);
            }
        }
    }
    out[0] = sum;   // expected exactly 4128.0 (see header)
}