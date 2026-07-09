// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// fst_bf16_transpose_kernel.cc — NPU bf16 [N,K] -> [K,N] transpose (4x8 tile).
//
// The MXFP4 dequant emits expert weights as B[N, K] row-major (each 17-byte
// dense block unpacks to 32 contiguous K-elements for one N-row — a layout
// that cannot be scattered to [K,N] without stride-N stores that destroy the
// dequant's vectorization).  The vectorized FFN GEMM uses b_row_maj, which
// requires B in [K, N] (the IRON default, well-tested path).  This kernel
// transposes the dequant output BO-to-BO on the NPU — pure data movement,
// no mmul, no CPU.
//
// Tile: 4 N-rows x 8 K-cols (32 bf16 = one v32).  The IRON strided DMA TAPs
// gather a [4,8] block from src[N,K] (row stride K) into a contiguous 32-elem
// ObjectFifo element, this kernel transposes it in-register to [8,4], and the
// drain TAP scatters the 32-elem result to dst[K,N] (row stride N) at the
// transposed position (k0, n0).
//
// aie::transpose(v, Row=4, Col=8) -> [8,4].  One load, one transpose, one store
// per call.  Multi-core (16 cores) distributes N-slabs; each core transposes
// 16384 tiles per projection.

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void fst_transpose_4x8(const bfloat16 *__restrict in,
                                  bfloat16 *__restrict out)
{
    event0();
    aie::vector<bfloat16, 32> v = aie::load_v<32>(in);
    // [4 N-rows, 8 K-cols] -> [8 K-rows, 4 N-cols]
    aie::vector<bfloat16, 32> t = aie::transpose(v, 4, 8);
    aie::store_v(out, t);
    event1();
}