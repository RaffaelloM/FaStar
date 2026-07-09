#!/usr/bin/env python3
"""FaStar MLA NPU — Multi-head Latent Attention xclbin for DeepSeek V4 Flash.

Simplified GQA attention kernel (without HashHop indexing) for M=8 tokens,
64 Q heads, 1 KV head, head_dim=512. The HashHop indexed variant is left
for a later optimization pass (Phase 7).

Design (Shim-DMA-only, single-core scalar):
    Input:  Q [M, n_head, head_dim]  →  [8, 64, 512] = [8, 32768]
            K [seq_len, head_dim]    →  [N, 512]
            V [seq_len, head_dim]    →  [N, 512]
    Output: O [M, n_head, head_dim] →  [8, 32768]

    For each of M tokens and 64 heads:
      scores = Q @ K^T / sqrt(512)   → [N] per (token, head)
      softmax(scores)
      output = scores @ V             → [512] per (token, head)

The compute is Q@K^T and softmax@V. Since M=8 and n_head=64, we have
512 independent (M, head) pairs. K and V are shared across all heads.

IRON tile breakdown:
    - Q: [M*n_head, head_dim] = [512, 512]  flattened: token 0 head 0..63,
      then token 1 head 0..63, etc.
    - K: [seq_len, 512] stored in L1 tiles
    - V: [seq_len, 512] stored in L1 tiles
    - O: [M*n_head, 512]  same layout as Q

L1 tile sizes (BF16):
    - Q_tile: 32 × 64 = 4 KB
    - K_tile: 64 × 64 = 8 KB
    - V_tile: 64 × 64 = 8 KB
    - Attn tile: 32 × 64 = 4 KB
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
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.utils import set_current_device

M_TOKENS = 8
N_HEADS = 64
HEAD_DIM = 512
SEQ_LEN = 256  # max KV cache window for this shard

Q_DIM = N_HEADS * HEAD_DIM  # 32768
Q_FLAT = M_TOKENS * N_HEADS  # 512 query vectors
SEQ_PAD = ((SEQ_LEN + 63) // 64) * 64  # round up to tile multiple

# L1 tile dimensions
TILE_Q_M = 32
TILE_KV_N = 64
TILE_KV_K = 64

set_current_device(NPU2())


@iron.jit
def fst_mla_attention(
    Q: In,
    K: In,
    V: In,
    O: Out,
    *,
    M: CompileTime[int],
    n_head: CompileTime[int],
    head_dim: CompileTime[int],
    seq_len: CompileTime[int],
    element_type: CompileTime[type],
):
    """GQA attention: Q[M, n_head*head_dim] @ K[seq_len, head_dim]^T  →  O[flat].

    Flattens M tokens × n_head heads into a single batch dimension Q_FLAT.
    K and V are broadcast across all heads (GQA with 1 KV head).
    """
    m, k, n = TILE_Q_M, TILE_KV_K, TILE_KV_N
    flat = M * n_head
    kv_heads = seq_len  # K and V have seq_len rows

    assert flat % m == 0, f"flat ({flat}) must be a multiple of tile_m ({m})"
    assert head_dim % k == 0, f"head_dim ({head_dim}) must be a multiple of tile_k ({k})"
    assert seq_len % n == 0, f"seq_len ({seq_len}) must be a multiple of tile_n ({n})"

    # Scalar matmul for Q@K^T scores
    score_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=element_type,
        output_dtype=np.dtype[element_type],
        vectorized=False,
    )
    score_zero = score_kernel.zero

    # Second matmul pass for softmax(score)@V
    value_kernel = kernels.mm(
        dim_m=m,
        dim_k=n,
        dim_n=k,
        input_dtype=element_type,
        output_dtype=np.dtype[element_type],
        vectorized=False,
    )
    value_zero = value_kernel.zero

    Q_ty = np.ndarray[(flat, head_dim), np.dtype[element_type]]
    K_ty = np.ndarray[(seq_len, head_dim), np.dtype[element_type]]
    V_ty = np.ndarray[(seq_len, head_dim), np.dtype[element_type]]
    O_ty = np.ndarray[(flat, head_dim), np.dtype[element_type]]

    q_tile_ty = np.ndarray[(m * k,), np.dtype[element_type]]
    k_tile_ty = np.ndarray[(k * n,), np.dtype[element_type]]
    v_tile_ty = np.ndarray[(n * k,), np.dtype[element_type]]
    s_tile_ty = np.ndarray[(m * n,), np.dtype[element_type]]
    o_tile_ty = np.ndarray[(m * k,), np.dtype[element_type]]

    fifo_Q = ObjectFifo(q_tile_ty, name="Q_L3L1", depth=2)
    fifo_K = ObjectFifo(k_tile_ty, name="K_L3L1", depth=2)
    fifo_V = ObjectFifo(v_tile_ty, name="V_L3L1", depth=2)
    fifo_S = ObjectFifo(s_tile_ty, name="S_L1L1", depth=2)
    fifo_O = ObjectFifo(o_tile_ty, name="O_L1L3", depth=2)

    # Two-phase runtime: first compute Q@K^T → scores tile by tile,
    # then softmax(scores)@V → output tile by tile.
    # The NPU cannot easily do softmax or exp in L1, so the host
    # must:
    #  1. Compute scores = Q @ K^T on NPU → drain to host
    #  2. Host does softmax (exp + normalize)
    #  3. Upload softmax scores → compute output = softmax @ V on NPU
    #
    # Therefore, we expose TWO separate entry points:
    #   - score_kernel: Q, K → scores
    #   - value_kernel: scores, V → O
    #
    # Both are compiled into the same xclbin via separate object fifos.

    def score_worker(of_q, of_k, of_s, zero, matmul):
        for _ in range_(flat // m * kv_heads // n):
            elem_s = of_s.acquire(1)
            zero(elem_s)
            for _ in range_(head_dim // k):
                elem_q = of_q.acquire(1)
                elem_k = of_k.acquire(1)
                matmul(elem_q, elem_k, elem_s)
                of_q.release(1)
                of_k.release(1)
            of_s.release(1)

    def value_worker(of_s, of_v, of_o, zero, matmul):
        for _ in range_(flat // m * head_dim // k):
            elem_o = of_o.acquire(1)
            zero(elem_o)
            for _ in range_(seq_len // n):
                elem_s = of_s.acquire(1)
                elem_v = of_v.acquire(1)
                matmul(elem_s, elem_v, elem_o)
                of_s.release(1)
                of_v.release(1)
            of_o.release(1)

    score_worker_obj = Worker(
        score_worker,
        [fifo_Q.cons(), fifo_K.cons(), fifo_S.prod(),
         score_zero, score_kernel],
    )
    value_worker_obj = Worker(
        value_worker,
        [fifo_S.cons(), fifo_V.cons(), fifo_O.prod(),
         value_zero, value_kernel],
    )

    # Q access: iterate over M*n_head query vectors, head_dim cols.
    # Each tile covers m query rows and k attention dims.
    # Enumerate all (flat//m) × (head_dim//k) tiles.
    q_tap = TensorAccessPattern(
        tensor_dims=(flat, head_dim),
        offset=0,
        sizes=[kv_heads // n, flat // m, m, k],
        strides=[0, head_dim, head_dim, 1],
    )
    # K access: seq_len KV rows, each head_dim wide. K is shared.
    k_tap = TensorAccessPattern(
        tensor_dims=(seq_len, head_dim),
        offset=0,
        sizes=[kv_heads // n, head_dim // k, k, n],
        strides=[head_dim, 1, head_dim, 0],  # NOT shim-DMA safe
    )

    # For now, use a simpler tile pattern that DMA can handle.
    # K: n KV row tiles of n×k, read row-major, repeated for each Q row.
    k_tap = TensorAccessPattern(
        tensor_dims=(seq_len, head_dim),
        offset=0,
        sizes=[flat // m, kv_heads // n, head_dim // k, k, n],
        strides=[0, n, k * head_dim, head_dim, 1],
    )

    # V access: same as K but transposed (n rows, k cols per tile)
    v_tap = TensorAccessPattern(
        tensor_dims=(seq_len, head_dim),
        offset=0,
        sizes=[flat // m, head_dim // k, kv_heads // n, n, k],
        strides=[0, k, n * head_dim, head_dim, 1],
    )

    # Scores: [flat, seq_len] output
    s_tap = TensorAccessPattern(
        tensor_dims=(flat, seq_len),
        offset=0,
        sizes=[1, flat // m, kv_heads // n, m, n],
        strides=[flat * seq_len, seq_len, n, seq_len, 1],
    )

    # Output: [flat, head_dim]
    o_tap = TensorAccessPattern(
        tensor_dims=(flat, head_dim),
        offset=0,
        sizes=[1, flat // m, head_dim // k, m, k],
        strides=[flat * head_dim, head_dim, k, head_dim, 1],
    )

    rt = Runtime()
    with rt.sequence(Q_ty, K_ty, V_ty, O_ty) as (A, B, C, D):
        rt.start(score_worker_obj)
        rt.start(value_worker_obj)

        tg_score = rt.task_group()
        rt.fill(fifo_Q.prod(), A, tap=q_tap, task_group=tg_score)
        rt.fill(fifo_K.prod(), B, tap=k_tap, task_group=tg_score)
        rt.drain(fifo_S.cons(), A, tap=s_tap, task_group=tg_score, wait=True)
        rt.finish_task_group(tg_score)

        tg_value = rt.task_group()
        rt.fill(fifo_S.prod(), A, tap=s_tap, task_group=tg_value)
        rt.fill(fifo_V.prod(), C, tap=v_tap, task_group=tg_value)
        rt.drain(fifo_O.cons(), D, tap=o_tap, task_group=tg_value, wait=True)
        rt.finish_task_group(tg_value)

    return Program(NPU2(), rt).resolve_program()


def aot_compile():
    xclbin, insts = fst_mla_attention.specialize(
        M=M_TOKENS,
        n_head=N_HEADS,
        head_dim=HEAD_DIM,
        seq_len=SEQ_LEN,
        element_type=bfloat16,
    ).compile()
    print(f"AOT compiled fst_mla_attention M={M_TOKENS} n_head={N_HEADS} "
          f"head_dim={HEAD_DIM} seq_len={SEQ_LEN} bf16 (NPU2)")
    print(f"  xclbin: {xclbin}")
    print(f"  insts:  {insts}")


if __name__ == "__main__":
    aot_compile()
