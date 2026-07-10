// SPDX-License-Identifier: Apache-2.0
// fst_mla_unified_kernels.cc -- Unified MLA GEMM kernel for DS4-XDNA
//
// Single AIE2 kernel used by every MLA projection (qc, kvc, qk, sv, oa, ob,
// wq_b, k_pe).  The engine tiles each GEMM into [m=8, k=64, n=64] sub-tiles
// and the orchestrator calls mla_gemm() once per (k,n) sub-tile, accumulating
// over the K dimension into the same C buffer (mla_zero() before the first
// k-tile).  mla_zero/mla_gemm share the same .o so the design binds both as
// sibling ExternalFunctions.
//
// CORRECTNESS NOTE — C layout (the bug fixed here):
// The previous hand-written kernel wrote C tile-major: each [r=4,t=8] mmul
// result block was stored contiguously, blocks laid out as a [rowA,colB] grid
// of tiles.  IRON's drain TAP, however, reads the C BO as ROW-MAJOR [m,n]
// (element (i,j) at i*n+j) — same TAP FFN uses.  Tile-major vs row-major
// scattered every output element to the wrong host position → the ~3.7x
// magnitude / sign-flip vs the host-float reference.
//
// FFN (compile_ffn_unified.py) sidesteps this by using IRON's built-in
// kernels.mm with dim_m=32, which selects the 4x4-vectorized bf16 microkernel
// (requires m % 16 == 0).  MLA needs m=8, which violates that divisibility
// (the static_assert is silenced by NOCPP, and the 4x4 expansion writes 4
// mmul tile-rows when only 2 exist → out-of-bounds + tile-major).  The scalar
// kernels.mm variant writes correct row-major C but has no pipelining pragmas
// and deadlocks the AIE memory system at m=8.
//
// This kernel keeps the proven 2x2-vectorized aie::mmul<4,8,8> expansion
// (m % (2*r) == 0 → 8 % 8 == 0 ✓, runs fast with chess_prepare_for_pipelining)
// but GATHERS the running partial C from row-major and SCATTERS the updated
// mmul result back to row-major, so the C BO matches what the drain TAP
// expects.  Each [4,8] mmul tile at (row_base, col_base) is 4 strided 8-elem
// stores at stride n=64 within the [8,64] C sub-tile.

#include <aie_api/aie.hpp>

// ============================================================================
// Row-major gather/scatter helpers for one [R=4, T=8] mmul tile inside an
// [m, n=64] C sub-tile.  The mmul to_vector()/ctor use tile-row-major
// [r0c0..r0c7, r1c0..r1c7, ...]; concat/load_v reconstruct that from the
// four strided rows of the row-major C buffer.
// ============================================================================
template <unsigned N = 64>
static inline aie::vector<bfloat16, 32>
gather_tile_rm(const bfloat16 *__restrict pC, unsigned row_base,
               unsigned col_base)
{
    auto r0 = aie::load_v<8>(pC + (row_base + 0) * N + col_base);
    auto r1 = aie::load_v<8>(pC + (row_base + 1) * N + col_base);
    auto r2 = aie::load_v<8>(pC + (row_base + 2) * N + col_base);
    auto r3 = aie::load_v<8>(pC + (row_base + 3) * N + col_base);
    return aie::concat(r0, r1, r2, r3);
}

template <unsigned N = 64>
static inline void
scatter_tile_rm(bfloat16 *__restrict pC, unsigned row_base, unsigned col_base,
                const aie::vector<bfloat16, 32> &v)
{
    aie::store_v(pC + (row_base + 0) * N + col_base, v.template extract<8>(0));
    aie::store_v(pC + (row_base + 1) * N + col_base, v.template extract<8>(1));
    aie::store_v(pC + (row_base + 2) * N + col_base, v.template extract<8>(2));
    aie::store_v(pC + (row_base + 3) * N + col_base, v.template extract<8>(3));
}

// ============================================================================
// Common GEMM template (2x2 vectorized 4x8x8 bf16 mmul, row-major C)
// ============================================================================
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_bf16(const bfloat16 *__restrict pA,
            const bfloat16 *__restrict pB,
            bfloat16 *__restrict pC)
{
    constexpr unsigned r = 4, s = 8, t = 8;
    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
    constexpr unsigned rowA = m / r, colA = k / s, colB = n / t;

    ::aie::set_rounding(aie::rounding_mode::floor);
    event0();

    for (unsigned z = 0; z < rowA; z += 2)
        chess_prepare_for_pipelining chess_loop_range(4, )
        {
            const unsigned rb0 = z * r;        // row_base for tile-rows z, z+1
            const unsigned rb1 = (z + 1) * r;

            for (unsigned j = 0; j < colB; j += 2)
                chess_flatten_loop
            {
                const unsigned cb0 = j * t;    // col_base for tile-cols j, j+1
                const unsigned cb1 = (j + 1) * t;

                const bfloat16 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                const bfloat16 *__restrict pB1 = pB + j * MMUL::size_B;
                const bfloat16 *__restrict pB2 = pB + (j + 1) * MMUL::size_B;

                // Load the running partial from C (row-major) so each call
                // ADDS its k=64 tile to the accumulated K-reduction.  Without
                // this load (or with a tile-major load), only the last K-tile
                // would survive and every K>64 GEMM would silently break.
                MMUL C00(gather_tile_rm<n>(pC, rb0, cb0));
                MMUL C01(gather_tile_rm<n>(pC, rb0, cb1));
                MMUL C10(gather_tile_rm<n>(pC, rb1, cb0));
                MMUL C11(gather_tile_rm<n>(pC, rb1, cb1));

                for (unsigned i = 0; i < colA; ++i)
                    chess_flatten_loop
                {
                    aie::vector<bfloat16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
                    aie::vector<bfloat16, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
                    aie::vector<bfloat16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * colB;
                    aie::vector<bfloat16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * colB;
                    C00.mac(A0, B0); C01.mac(A0, B1);
                    C10.mac(A1, B0); C11.mac(A1, B1);
                }

                // Scatter the updated [4,8] tiles back to row-major C so the
                // drain TAP (row-major [m,n]) reads the right element order.
                scatter_tile_rm<n>(pC, rb0, cb0, C00.template to_vector<bfloat16>());
                scatter_tile_rm<n>(pC, rb0, cb1, C01.template to_vector<bfloat16>());
                scatter_tile_rm<n>(pC, rb1, cb0, C10.template to_vector<bfloat16>());
                scatter_tile_rm<n>(pC, rb1, cb1, C11.template to_vector<bfloat16>());
            }
        }
    event1();
}

// ============================================================================
// Zero the C sub-tile (m*n bf16 = 8*64 = 512 elems) before the first k-tile.
// Layout-independent — zeroing is the same for row-major or tile-major C.
// ============================================================================
template <unsigned ELEMS>
static inline void zero_accum(bfloat16 *c)
{
    constexpr unsigned VEC = 32;            // 8-elem... 32-elem vector store
    const aie::vector<bfloat16, VEC> zeros = aie::zeros<bfloat16, VEC>();
    bfloat16 *c_end = c + ELEMS;
    for (; c < c_end; c += VEC)
        aie::store_v(c, zeros);
}

// Single symbol pair bound by compile_mla_unified.py for ALL MLA GEMMs.
extern "C" void mla_gemm(bfloat16 *A, bfloat16 *B, bfloat16 *C)
{
    matmul_bf16<8, 64, 64>(A, B, C);
}

// DIAGNOSTIC: constant-fill kernel.  If the engine audit value changes to
// 1.5 when this is bound instead of mla_gemm, the kernel output reaches the
// host; if it stays -0.2129, the engine is not using the kernel C store.
extern "C" void mla_gemm_diag(bfloat16 *A, bfloat16 *B, bfloat16 *C)
{
    (void)A; (void)B;
    constexpr unsigned VEC = 32;
    const aie::vector<bfloat16, VEC> fill =
        aie::broadcast<bfloat16, VEC>(bfloat16{0x3fc0}); // 1.5 in bf16
    for (unsigned i = 0; i < 8 * 64; i += VEC)
        aie::store_v(C + i, fill);
}

extern "C" void mla_zero(bfloat16 *C)
{
    zero_accum<8 * 64>(C);
}

// ============================================================================
// QK scale: multiply each of the 8*64 output bf16s by 1/sqrt(K_full_dim).
// (Ported from fst_mla_unified_kernel.cc so the unified xclbin's qk/sv
//  runtime_sequences can bind mla_qk_scale / mla_sv_scale, which the row-major
//  fixed kernel lacked.  The scale step is layout-independent.)
// ============================================================================
extern "C" void mla_qk_scale(bfloat16 *__restrict data, int32_t K_full_dim)
{
    event0();
    constexpr int VEC = 64;
    constexpr int TOTAL = 8 * 64;
    float scale_f = 1.0f / __builtin_aie2p_sqrtf((float)K_full_dim);
    aie::vector<bfloat16, VEC> scale_vec = aie::broadcast<bfloat16, VEC>((bfloat16)scale_f);
    for (int i = 0; i < TOTAL; i += VEC) {
        aie::vector<bfloat16, VEC> v = aie::load_v<VEC>(data + i);
        v = aie::mul(v, scale_vec).to_vector<bfloat16>();
        aie::store_v(data + i, v);
    }
    event1();
}

// SV scale: identity (kept for symmetry with the qk pipeline).
extern "C" void mla_sv_scale(bfloat16 *__restrict data, int32_t D)
{
    (void)data;
    (void)D;
}