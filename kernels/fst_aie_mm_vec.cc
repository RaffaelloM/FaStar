// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_aie_mm_vec.cc — VECTORIZED SET-then-ACC bf16 matmul for FaStar FFN GEMMs.
//
// IRON's matmul_vectorized_4x4 does C += A·B as read-modify-write (loads acc
// from the C ObjectFifo slot, MACs, stores) and relies on a `zero` pass to
// clear C. IRON forces ObjectFifo depth=2, so C slots are reused across
// output tiles; the `+=` reads the previous tile's residual and accumulates
// on it → the (1 + t//2)× error. The fix is SET-then-ACC: TWO sibling kernel
// symbols compiled from one .o — the worker calls SET for K-tile 0 and ACC
// for K-tiles 1+, NO zero pass.
//
//   matmul_set_vectorized_bf16_bf16 : C =  A*B  (SET — i==0 uses aie::mmul::mul,
//                                                 OVERWRITES the accumulator,
//                                                 no read of pC, no zero pass)
//   matmul_acc_vectorized_bf16_bf16 : C += A*B  (ACC — load-init from pC, mac)
// SET uses `mul` (overwrite) at i==0 so the uninitialised MMUL default-construct
// is never read — this also fixes the prior `qds_device::wait()` crash.  Two
// separate symbols (no runtime scalar flag) because IRON ExternalFunction in
// this codebase has no precedent for scalar int args.
//
// Tile m=16 (not 32) so BOTH symbols fit AIE program memory: 4x4 expansion
// unrolls (m/16)*(n/16)*(k/8)*16 mmul per symbol = 1*2*4*16 = 128 each, 256
// total.  M=32 with m=16 → M_div_m=2, handled by the single_core tile_row loop.
// 4x4 bf16 needs m%16==0 (16✓), n%16==0 (32✓).

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifdef B_COL_MAJ
constexpr bool is_b_row_maj = false;
#else
constexpr bool is_b_row_maj = true;
#endif
#ifdef C_COL_MAJ
constexpr bool is_c_row_maj = false;
#else
constexpr bool is_c_row_maj = true;
#endif

// 4x4 expansion of aie::mmul<r,s,t>.  SET=true: i==0 uses mul (overwrite, no
// read of pC/default-construct); SET=false: load-init from pC, mac at i==0.
template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t, bool SET,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void
matmul_vec_4x4(const T_in *__restrict pA, const T_in *__restrict pB, T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (unsigned z = 0; z < rowA; z += 4) {
        T_out *__restrict pC1, *__restrict pC2, *__restrict pC3, *__restrict pC4;
        if constexpr (c_row_maj) {
            pC1 = pC + (z * colB) * MMUL::size_C;
            pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
            pC3 = pC + ((z + 2) * colB) * MMUL::size_C;
            pC4 = pC + ((z + 3) * colB) * MMUL::size_C;
        }
        for (unsigned j = 0; j < colB; j += 4) {
            if constexpr (!c_row_maj) {
                pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
                pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
                pC3 = pC + (j + 2) * rowA * MMUL::size_C + z * MMUL::size_C;
                pC4 = pC + (j + 3) * rowA * MMUL::size_C + z * MMUL::size_C;
            }
            const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
            const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
            const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
            const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;
            const T_in *__restrict pB1, *__restrict pB2, *__restrict pB3, *__restrict pB4;
            if constexpr (b_row_maj) {
                pB1 = pB + (j) * MMUL::size_B;
                pB2 = pB + (j + 1) * MMUL::size_B;
                pB3 = pB + (j + 2) * MMUL::size_B;
                pB4 = pB + (j + 3) * MMUL::size_B;
            } else {
                pB1 = pB + (j * colA) * MMUL::size_B;
                pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
                pB3 = pB + ((j + 2) * colA) * MMUL::size_B;
                pB4 = pB + ((j + 3) * colA) * MMUL::size_B;
            }

            aie::vector<T_in, MMUL::size_A> A0, A1, A2, A3;
            aie::vector<T_in, MMUL::size_B> B0, B1, B2, B3;

            MMUL C00, C01, C02, C03, C10, C11, C12, C13;
            MMUL C20, C21, C22, C23, C30, C31, C32, C33;
            if constexpr (!SET) {
                if constexpr (c_row_maj) {
                    C00 = MMUL(aie::load_v<MMUL::size_C>(pC1));
                    C01 = MMUL(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C));
                    C02 = MMUL(aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C));
                    C03 = MMUL(aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C));
                    C10 = MMUL(aie::load_v<MMUL::size_C>(pC2));
                    C11 = MMUL(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C));
                    C12 = MMUL(aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C));
                    C13 = MMUL(aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C));
                    C20 = MMUL(aie::load_v<MMUL::size_C>(pC3));
                    C21 = MMUL(aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C));
                    C22 = MMUL(aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C));
                    C23 = MMUL(aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C));
                    C30 = MMUL(aie::load_v<MMUL::size_C>(pC4));
                    C31 = MMUL(aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C));
                    C32 = MMUL(aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C));
                    C33 = MMUL(aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C));
                } else {
                    C00 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r));
                    C01 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r));
                    C02 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC3), t, r));
                    C03 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC4), t, r));
                    C10 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r));
                    C11 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r));
                    C12 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C), t, r));
                    C13 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C), t, r));
                    C20 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C), t, r));
                    C21 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C), t, r));
                    C22 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C), t, r));
                    C23 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C), t, r));
                    C30 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C), t, r));
                    C31 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C), t, r));
                    C32 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C), t, r));
                    C33 = MMUL(aie::transpose(aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C), t, r));
                }
            }
            // SET: Cxx default-constructed; i==0 mul overwrites (no read).

            // ── i == 0 : SET → mul (overwrite), ACC → mac (load-init done) ──
            A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
            A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
            A2 = aie::load_v<MMUL::size_A>(pA3); pA3 += MMUL::size_A;
            A3 = aie::load_v<MMUL::size_A>(pA4); pA4 += MMUL::size_A;
            if constexpr (b_row_maj) {
                B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * colB;
                B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * colB;
                B2 = aie::load_v<MMUL::size_B>(pB3); pB3 += MMUL::size_B * colB;
                B3 = aie::load_v<MMUL::size_B>(pB4); pB4 += MMUL::size_B * colB;
            } else {
                B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s); pB1 += MMUL::size_B;
                B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s); pB2 += MMUL::size_B;
                B2 = aie::transpose(aie::load_v<MMUL::size_B>(pB3), t, s); pB3 += MMUL::size_B;
                B3 = aie::transpose(aie::load_v<MMUL::size_B>(pB4), t, s); pB4 += MMUL::size_B;
            }
            if constexpr (SET) {
                C00.mul(A0, B0); C01.mul(A0, B1); C02.mul(A0, B2); C03.mul(A0, B3);
                C10.mul(A1, B0); C11.mul(A1, B1); C12.mul(A1, B2); C13.mul(A1, B3);
                C20.mul(A2, B0); C21.mul(A2, B1); C22.mul(A2, B2); C23.mul(A2, B3);
                C30.mul(A3, B0); C31.mul(A3, B1); C32.mul(A3, B2); C33.mul(A3, B3);
            } else {
                C00.mac(A0, B0); C01.mac(A0, B1); C02.mac(A0, B2); C03.mac(A0, B3);
                C10.mac(A1, B0); C11.mac(A1, B1); C12.mac(A1, B2); C13.mac(A1, B3);
                C20.mac(A2, B0); C21.mac(A2, B1); C22.mac(A2, B2); C23.mac(A2, B3);
                C30.mac(A3, B0); C31.mac(A3, B1); C32.mac(A3, B2); C33.mac(A3, B3);
            }

            // ── i >= 1 : always mac ──
            for (unsigned i = 1; i < colA; ++i) {
                A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
                A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
                A2 = aie::load_v<MMUL::size_A>(pA3); pA3 += MMUL::size_A;
                A3 = aie::load_v<MMUL::size_A>(pA4); pA4 += MMUL::size_A;
                if constexpr (b_row_maj) {
                    B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * colB;
                    B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * colB;
                    B2 = aie::load_v<MMUL::size_B>(pB3); pB3 += MMUL::size_B * colB;
                    B3 = aie::load_v<MMUL::size_B>(pB4); pB4 += MMUL::size_B * colB;
                } else {
                    B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s); pB1 += MMUL::size_B;
                    B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s); pB2 += MMUL::size_B;
                    B2 = aie::transpose(aie::load_v<MMUL::size_B>(pB3), t, s); pB3 += MMUL::size_B;
                    B3 = aie::transpose(aie::load_v<MMUL::size_B>(pB4), t, s); pB4 += MMUL::size_B;
                }
                C00.mac(A0, B0); C01.mac(A0, B1); C02.mac(A0, B2); C03.mac(A0, B3);
                C10.mac(A1, B0); C11.mac(A1, B1); C12.mac(A1, B2); C13.mac(A1, B3);
                C20.mac(A2, B0); C21.mac(A2, B1); C22.mac(A2, B2); C23.mac(A2, B3);
                C30.mac(A3, B0); C31.mac(A3, B1); C32.mac(A3, B2); C33.mac(A3, B3);
            }

            if constexpr (c_row_maj) {
                aie::store_v(pC1, C00.template to_vector<T_out>()); pC1 += MMUL::size_C;
                aie::store_v(pC1, C01.template to_vector<T_out>()); pC1 += MMUL::size_C;
                aie::store_v(pC1, C02.template to_vector<T_out>()); pC1 += MMUL::size_C;
                aie::store_v(pC1, C03.template to_vector<T_out>());
                aie::store_v(pC2, C10.template to_vector<T_out>()); pC2 += MMUL::size_C;
                aie::store_v(pC2, C11.template to_vector<T_out>()); pC2 += MMUL::size_C;
                aie::store_v(pC2, C12.template to_vector<T_out>()); pC2 += MMUL::size_C;
                aie::store_v(pC2, C13.template to_vector<T_out>());
                aie::store_v(pC3, C20.template to_vector<T_out>()); pC3 += MMUL::size_C;
                aie::store_v(pC3, C21.template to_vector<T_out>()); pC3 += MMUL::size_C;
                aie::store_v(pC3, C22.template to_vector<T_out>()); pC3 += MMUL::size_C;
                aie::store_v(pC3, C23.template to_vector<T_out>());
                aie::store_v(pC4, C30.template to_vector<T_out>()); pC4 += MMUL::size_C;
                aie::store_v(pC4, C31.template to_vector<T_out>()); pC4 += MMUL::size_C;
                aie::store_v(pC4, C32.template to_vector<T_out>()); pC4 += MMUL::size_C;
                aie::store_v(pC4, C33.template to_vector<T_out>());
            } else {
                aie::store_v(pC1, aie::transpose(C00.template to_vector<T_out>(), r, t));
                aie::store_v(pC2, aie::transpose(C01.template to_vector<T_out>(), r, t));
                aie::store_v(pC3, aie::transpose(C02.template to_vector<T_out>(), r, t));
                aie::store_v(pC4, aie::transpose(C03.template to_vector<T_out>(), r, t));
                aie::store_v(pC1 + MMUL::size_C, aie::transpose(C10.template to_vector<T_out>(), r, t));
                aie::store_v(pC2 + MMUL::size_C, aie::transpose(C11.template to_vector<T_out>(), r, t));
                aie::store_v(pC3 + MMUL::size_C, aie::transpose(C12.template to_vector<T_out>(), r, t));
                aie::store_v(pC4 + MMUL::size_C, aie::transpose(C13.template to_vector<T_out>(), r, t));
                aie::store_v(pC1 + 2 * MMUL::size_C, aie::transpose(C20.template to_vector<T_out>(), r, t));
                aie::store_v(pC2 + 2 * MMUL::size_C, aie::transpose(C21.template to_vector<T_out>(), r, t));
                aie::store_v(pC3 + 2 * MMUL::size_C, aie::transpose(C22.template to_vector<T_out>(), r, t));
                aie::store_v(pC4 + 2 * MMUL::size_C, aie::transpose(C23.template to_vector<T_out>(), r, t));
                aie::store_v(pC1 + 3 * MMUL::size_C, aie::transpose(C30.template to_vector<T_out>(), r, t));
                aie::store_v(pC2 + 3 * MMUL::size_C, aie::transpose(C31.template to_vector<T_out>(), r, t));
                aie::store_v(pC3 + 3 * MMUL::size_C, aie::transpose(C32.template to_vector<T_out>(), r, t));
                aie::store_v(pC4 + 3 * MMUL::size_C, aie::transpose(C33.template to_vector<T_out>(), r, t));
            }
        }
    }
}

extern "C" {

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

constexpr unsigned VR = 4, VS = 8, VT = 4;

void matmul_set_vectorized_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out)
{
    matmul_vec_4x4<bfloat16, bfloat16, (DIM_M / VR), (DIM_K / VS), (DIM_N / VT),
                   VR, VS, VT, true, is_b_row_maj, is_c_row_maj>(a_in, b_in, c_out);
}

void matmul_acc_vectorized_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out)
{
    matmul_vec_4x4<bfloat16, bfloat16, (DIM_M / VR), (DIM_K / VS), (DIM_N / VT),
                   VR, VS, VT, false, is_b_row_maj, is_c_row_maj>(a_in, b_in, c_out);
}

} // extern "C"