// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_aie_mm.cc — SET-then-ACC scalar bf16 matmul for FaStar NPU GEMMs.
//
// IRON's matmul_scalar does C += A*B (read-modify-write) and relies on a
// separate zero_scalar pass to clear the C slot before the K-reduction.  On
// AIE2, with the C ObjectFifo forced to depth 2 (slot reuse across N-tiles),
// the read-modify-write at n_local == 3 (mod 4) picks up the previous tile's
// residual in the reused slot, producing a (1 + t//2)x error from N-tile 2 on.
//
// This file provides two sibling symbols compiled from one .o:
//   matmul_set_scalar_bf16_bf16 : C =  A*B   (SET, no read — overwrites residual)
//   matmul_acc_scalar_bf16_bf16 : C += A*B   (ACC, same as IRON's matmul_scalar)
// The worker calls SET for K-tile 0 and ACC for K-tiles 1+, eliminating the
// dependence on zero clearing the reused slot.  No zero pass is needed.

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// SET: c[row*colB + col] = running_sum  (no read of c)
template <typename T_in, typename T_out, int rowA, int colA, int colB,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_set_scalar(T_in *a, T_in *b, T_out *c)
{
    for (int row = 0; row < rowA; row++) {
        for (int col = 0; col < colB; col++) {
            T_out running_sum = 0;
            for (int i = 0; i < colA; i++) {
                T_in a_val = a[row * colA + i];
                T_in b_val;
                if constexpr (b_row_maj) {
                    b_val = b[i * colB + col];
                } else {
                    b_val = b[i + col * colA];
                }
                running_sum += a_val * b_val;
            }
            T_out *c_ptr;
            if constexpr (c_row_maj) {
                c_ptr = &c[row * colB + col];
            } else {
                c_ptr = &c[row + col * rowA];
            }
            *c_ptr = running_sum;  // SET
        }
    }
}

// ACC: c[row*colB + col] += running_sum  (identical to IRON matmul_scalar)
template <typename T_in, typename T_out, int rowA, int colA, int colB,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_acc_scalar(T_in *a, T_in *b, T_out *c)
{
    for (int row = 0; row < rowA; row++) {
        for (int col = 0; col < colB; col++) {
            T_out running_sum = 0;
            for (int i = 0; i < colA; i++) {
                T_in a_val = a[row * colA + i];
                T_in b_val;
                if constexpr (b_row_maj) {
                    b_val = b[i * colB + col];
                } else {
                    b_val = b[i + col * colA];
                }
                running_sum += a_val * b_val;
            }
            T_out *c_ptr;
            if constexpr (c_row_maj) {
                c_ptr = &c[row * colB + col];
            } else {
                c_ptr = &c[row + col * rowA];
            }
            *c_ptr += running_sum;  // ACC
        }
    }
}

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

extern "C" {

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

// Only bf16->bf16 is needed by FaStar's MLA/FFN GEMMs.
void matmul_set_scalar_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out)
{
    matmul_set_scalar<bfloat16, bfloat16, DIM_M, DIM_K, DIM_N,
                      is_b_row_maj, is_c_row_maj>(a_in, b_in, c_out);
}

void matmul_acc_scalar_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out)
{
    matmul_acc_scalar<bfloat16, bfloat16, DIM_M, DIM_K, DIM_N,
                      is_b_row_maj, is_c_row_maj>(a_in, b_in, c_out);
}

} // extern "C"