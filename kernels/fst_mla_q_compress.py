#!/usr/bin/env python3
"""FaStar MLA Q Compress — Fused RMSNorm + Q8_0 GEMM for XDNA2 (NPU2).

DeepSeek V4 Flash Q compression step:
    h_norm = RMSNorm(hidden)          [M, 4096]
    latent_Q = h_norm @ attn_q_a^T    [M, 1024]

where attn_q_a is in Q8_0 format [4096, 1024].

Fusion: the intermediate h_norm buffer is never materialized in DRAM.
Raw hidden states are pre-multiplied by per-row rms_scales on the host
(trivial 8×4096 = 32768 MACs), then the kernel applies RMSNorm weights
and performs the GEMM in a single pass.

Tile geometry:
    M = 8      (speculative batch tokens)
    K = 4096   (hidden_size)
    N = 1024   (q_lora_rank)

    K_tile = 32   (1 Q8_0 block)
    N_tile = 32   (32 output columns per kernel call)
    A_normW: [M + 1, K_tile] BF16 = 9×32 = 576 bytes (packed A_scaled + NormW)

Without fusion:   2 DRAM buffers (h_norm + latent_Q)  →  2 × 8×4096 = 64 KB + 16 KB
With fusion:      1 DRAM buffer (latent_Q)             →  16 KB
RMSNorm overhead: 0 ms (absorbed into GEMM tile loop)
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

M_BATCH = 8
K_HIDDEN = 4096
N_LORA = 1024
TILE_K = 32  # 1 Q8_0 block
TILE_N = 32
K_BLOCKS = K_HIDDEN // TILE_K  # 128
N_TILES = N_LORA // TILE_N     # 32
TOTAL_TILES = N_TILES * K_BLOCKS  # 4096 kernel calls
Q8_0_BLOCK = 34  # 2 bytes BF16 scale + 32 int8 values

# ── Host-side helper: pre-scale hidden states by per-row rms_scale ──────

def rms_scale_f32(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute per-row RMSNorm scales. x shape: [M, K]."""
    sum_sq = np.sum(x.astype(np.float32) ** 2, axis=1)
    return 1.0 / np.sqrt(sum_sq / x.shape[1] + eps)


@iron.jit
def fst_mla_q_compress(
    input0: In,   # A_normW: [M_total, K_total_packed] BF16
    input1: In,   # B_q8: [N_total * K_blocks * 34] uint8
    output: Out,  # C: [M * N] BF16
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
):
    """Fused RMSNorm + Q8_0 GEMM for Q compression."""
    kt, nt = TILE_K, TILE_N
    kb, nti = K_BLOCKS, N_TILES

    assert M == 8, f"M must be 8, got {M}"
    assert K == K_HIDDEN, f"K must be {K_HIDDEN}, got {K}"
    assert N == N_LORA, f"N must be {N_LORA}, got {N}"

    # ── Tile types ────────────────────────────────────────────────────
    # Packed A_normW: concatenated A_scaled[M, K_tile] and NormW[K_tile]
    a_packed_ty = np.ndarray[((M + 1) * kt,), np.dtype[bfloat16]]
    # B_q8: N_tile columns × 34 bytes each (non-contiguous, handled by TAP)
    b_tile_ty = np.ndarray[(nt * Q8_0_BLOCK,), np.dtype[np.uint8]]
    c_tile_ty = np.ndarray[(M * nt,), np.dtype[bfloat16]]

    A_ty = np.ndarray[((M + 1) * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(N * kb * Q8_0_BLOCK,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]

    # ── External kernel ────────────────────────────────────────────────
    fused_kernel = ExternalFunction(
        "fst_fused_rmsnorm_gemm_q8_0_8x32x32",
        source_file="fst_fused_rmsnorm_gemm_kernel.cc",
        arg_types=[a_packed_ty, b_tile_ty, c_tile_ty],
        include_dirs=[aie_config.cxx_header_path()],
        object_file_name="fst_fused_rmsnorm_gemm.o",
    )

    # ── ObjectFifos ────────────────────────────────────────────────────
    fifo_AW = ObjectFifo(a_packed_ty, name="AW_L1", depth=2)
    fifo_B  = ObjectFifo(b_tile_ty,  name="B_L1",  depth=2)
    fifo_C  = ObjectFifo(c_tile_ty,  name="C_L1",  depth=2)

    def core_fn(of_aw, of_b, of_c, kernel):
        # Nested: for each N-tile, zero C, then accumulate over all K-tiles
        for _ni in range_(nti) if nti > 1 else range(1):
            elem_out = of_c.acquire(1)
            # Zero accumulator before reduction over K
            for i in range_(M * nt) if (M * nt) > 1 else range(1):
                elem_out[i] = 0
            for _ki in range_(kb) if kb > 1 else range(1):
                elem_aw = of_aw.acquire(1)
                elem_b  = of_b.acquire(1)
                kernel(elem_aw, elem_b, elem_out)
                of_aw.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [fifo_AW.cons(), fifo_B.cons(), fifo_C.prod(), fused_kernel],
        stack_size=0x400,
    )

    # ── TAP for packed A+NormW ─────────────────────────────────────────
    # Input layout: M+1 rows of K elements each (M rows of A_scaled,
    # 1 row of NormW). Packed per K_tile:
    #   [A_scaled[0:M, kt_start:kt_start+kt], NormW[kt_start:kt_start+kt]]
    #
    # For each kernel call (k_idx, n_idx):
    #   K_tile start = k_idx * kt
    #   A_scaled rows 0..M-1, columns kt_start..kt_start+kt
    #   NormW columns kt_start..kt_start+kt
    #
    # The kernel is called once per (k, n) pair. A_packed is repeated for
    # each N-tile (nti times), so the A DMA pattern repeats the same K-tile
    # across all N-tiles.
    #
    # 4D TAP: [N_tiles, K_blocks, M+1, K_tile]
    a_tap = TensorAccessPattern(
        tensor_dims=((M + 1) * K,),
        offset=0,
        sizes=[nti, kb, M + 1, kt],
        strides=[0, (M + 1) * kt, kt, 1],
    )

    # ── TAP for B_q8 ───────────────────────────────────────────────────
    # Q8_0 layout: N columns × K_BLOCKS blocks each, 34 bytes/block.
    # Column j's k_idx-th block at: (j * K_BLOCKS + k_idx) * 34.
    #
    # Tiling: nti outer (groups of nt=32 columns), kb inner (K-blocks).
    # Within each (n_tile, k_idx) tile: 32 Q8_0 blocks (one per column
    # in this N-tile group at this K-position), NOT contiguous.
    # Stride between consecutive columns at same K-block: K_BLOCKS * 34.
    #
    # 5D TAP: [nti, kb, 2, 16, Q8_0_BLOCK]
    #   strides[0] = nt * K_BLOCKS * 34   (next 32-column group)
    #   strides[1] = 34                    (next K-block, shifts all cols)
    #   strides[2] = 16 * K_BLOCKS * 34   (next 16-column sub-group)
    #   strides[3] = K_BLOCKS * 34        (next column, same K-block)
    #   strides[4] = 1                    (within 34-byte block)
    # BD size = 16 * 34 = 544  < 1023.
    b_tap = TensorAccessPattern(
        tensor_dims=(N * kb * Q8_0_BLOCK,),
        offset=0,
        sizes=[nti, kb, 2, 16, Q8_0_BLOCK],
        strides=[
            nt * K_BLOCKS * Q8_0_BLOCK,   # nti: skip 32 columns worth
            Q8_0_BLOCK,                    # kb: next K-block
            16 * K_BLOCKS * Q8_0_BLOCK,    # sub-group: skip 16 columns
            K_BLOCKS * Q8_0_BLOCK,         # column: next column's block
            1,                             # within 34 bytes
        ],
    )

    # ── TAP for C ──────────────────────────────────────────────────────
    # Output layout: row-major [M, N]
    # For each (k_idx, n_idx):
    #   C[M, n_tile_start..n_tile_start+nt] accumulates across K
    #   Since the kernel C is in/out accumulator, the same N-tile slot is
    #   reused for all K-tiles.
    #
    #   sizes=[nti, kb, M, nt]
    #   strides=[nt, 0, N, 1]
    c_tap = TensorAccessPattern(
        tensor_dims=(M * N,),
        offset=0,
        sizes=[nti, kb, M, nt],
        strides=[nt, 0, N, 1],
    )

    # ── Runtime ────────────────────────────────────────────────────────
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(fifo_AW.prod(), A, tap=a_tap, task_group=tg)
        rt.fill(fifo_B.prod(), B, tap=b_tap, task_group=tg)
        rt.drain(fifo_C.cons(), C, tap=c_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


# ── Reference implementation for validation ─────────────────────────────

def ref_q_compress(hidden: np.ndarray, attn_norm_w: np.ndarray,
                   attn_q_a_q8: np.ndarray,
                   K: int = K_HIDDEN, N: int = N_LORA,
                   eps: float = 1e-6) -> np.ndarray:
    """CPU reference: RMSNorm + Q8_0 GEMM for Q compression.

    attn_q_a_q8 is a flat uint8 array in DS4 Q8_0 layout:
    [N, K/32, 34] where each 34-byte block = [scale_u16, q8_int8[32]].
    """
    M = hidden.shape[0]
    sum_sq = np.sum(hidden.astype(np.float32) ** 2, axis=1)
    rms = 1.0 / np.sqrt(sum_sq / hidden.shape[1] + eps)
    h_norm = hidden.astype(np.float32) * rms[:, np.newaxis] * attn_norm_w.astype(np.float32)

    result = np.zeros((M, N), dtype=np.float32)
    n_blocks = K // 32
    for j in range(N):
        for b in range(n_blocks):
            blk_start = (j * n_blocks + b) * 34
            scale_u16 = (attn_q_a_q8[blk_start].astype(np.uint16) |
                         (attn_q_a_q8[blk_start + 1].astype(np.uint16) << 8))
            scale = (scale_u16.astype(np.uint32) << 16).view(np.float32)
            q8 = attn_q_a_q8[blk_start + 2 : blk_start + 34].view(np.int8).astype(np.float32)
            for i in range(M):
                # Manually unrollable but scalar for reference correctness
                for k in range(32):
                    result[i, j] += h_norm[i, b * 32 + k] * scale * q8[k]
    return result


# ── AOT compile ─────────────────────────────────────────────────────────

def aot_compile():
    xclbin, insts = fst_mla_q_compress.specialize(
        M=M_BATCH, K=K_HIDDEN, N=N_LORA,
    ).compile()
    print(f"AOT compiled FUSED RMSNorm+GEMM Q Compress "
          f"M={M_BATCH} K={K_HIDDEN} N={N_LORA} (NPU2)")
    print(f"  xclbin: {xclbin}")
    print(f"  insts:  {insts}")


# ── Quick test ──────────────────────────────────────────────────────────

def test_fusion():
    """Validate fused kernel against CPU reference using random data."""
    print("=" * 60)
    print("Testing Fused RMSNorm+GEMM Q Compress")
    print("=" * 60)

    rng = np.random.RandomState(42)
    M, K, N = M_BATCH, K_HIDDEN, N_LORA

    # Generate random inputs
    hidden_bf16 = rng.randn(M, K).astype(np.float32) * 0.5
    attn_norm_w_bf16 = rng.randn(K).astype(np.float32) * 0.1

    # Fake Q8_0 weights: pack as uint8 array in DS4 layout
    # [N, K_blocks, 34] flattened
    attn_q_a_q8 = np.zeros(N * K_BLOCKS * Q8_0_BLOCK, dtype=np.uint8)
    for j in range(N):
        for b in range(K_BLOCKS):
            off = (j * K_BLOCKS + b) * Q8_0_BLOCK
            scale_f32 = np.float32(rng.randn() * 0.1 + 0.5)
            scale_u16 = np.uint16(scale_f32.view(np.uint32) >> 16)
            attn_q_a_q8[off:off+2] = np.frombuffer(scale_u16.tobytes(), dtype=np.uint8)
            q8_vals = (rng.randint(-4, 4, size=32)).astype(np.int8)
            attn_q_a_q8[off+2:off+34] = q8_vals.view(np.uint8)

    # ── CPU reference ────────────────────────────────────────────────
    ref = ref_q_compress(hidden_bf16, attn_norm_w_bf16, attn_q_a_q8)

    # ── Prepare NPU inputs ───────────────────────────────────────────
    # Pre-scale hidden by rms_scale (float32 for precision)
    rms = rms_scale_f32(hidden_bf16)
    hidden_scaled = hidden_bf16.astype(np.float32) * rms[:, np.newaxis]

    # Pack A_scaled + NormW as BF16: [M+1, K] row-major
    packed_aw_bf16 = np.zeros(((M + 1) * K), dtype=bfloat16)
    packed_aw_bf16[:M*K] = hidden_scaled.ravel().astype(bfloat16)
    packed_aw_bf16[M*K:] = attn_norm_w_bf16.astype(bfloat16)

    # ── Run NPU ──────────────────────────────────────────────────────
    import aie.iron as iron
    from aie.iron.device import NPU2
    from aie.utils import set_current_device
    set_current_device(NPU2())

    A_bo = iron.zeros((M + 1) * K, dtype=bfloat16, device="npu")
    B_bo = iron.zeros(N * K_BLOCKS * Q8_0_BLOCK, dtype=np.uint8, device="npu")
    C_bo = iron.zeros(M * N, dtype=bfloat16, device="npu")

    A_bo.numpy()[:] = packed_aw_bf16
    B_bo.numpy()[:] = attn_q_a_q8

    t = __import__("time").perf_counter()
    fst_mla_q_compress(A_bo, B_bo, C_bo, M=M, K=K, N=N)
    dt_ms = (__import__("time").perf_counter() - t) * 1000

    out = C_bo.numpy().astype(np.float32).reshape(M, N)

    max_err = np.max(np.abs(out - ref))
    print(f"\n  NPU time:      {dt_ms:.1f} ms")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Result OK:     {max_err < 0.1}")
    return max_err < 0.1


if __name__ == "__main__":
    if test_fusion():
        print("\nFusion verified. AOT compile...")
        aot_compile()
    else:
        print("\nFusion FAILED. Fix kernel before AOT compiling.")
