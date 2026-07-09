#!/usr/bin/env python3
"""Compile the FUSED dequant+GEMM xclbin (fst_fused_dequant_gemm.xclbin).

One dispatch: takes packed .fst 17-byte MXFP4 blocks (uint8) + BF16 hidden state A,
dequants B on-the-fly inside the AIE kernel, and computes a K-tile partial of the
GEMM C — all in a single kernel call per (N-tile, K-tile).

Target: M=16, K=4096, N=2048 (gate/up projection).
  Tile:  TILE_M=16, TILE_K=64, TILE_N=64  -> 1 M-tile × 32 N-tiles × 64 K-tiles
  MMUL:  aie::mmul<r=4,s=8,t=8, bf16, bf16, accauto>
         rowA = 16/4 = 4, colA = 64/8 = 8, colB = 64/8 = 8

=== WHY THE FLAT SINGLE-LOOP DESIGN (the B-DMA fix) =======================
The earlier 3D-TAP / nested-worker design hit a fatal IRON DMA bug: IRON hoists
the worker's OUTER loop (M//m * N//n = 32) into the DMA `repeat_count`, but the B
TAP already carried that same 32 as its outermost dim.  Because the expressions
differ, IRON did NOT fuse them — it STACKED them: the B BD walked 32×64×2304 B per
step AND repeated 32×, reading 150 MB from a 4.7 MB BO (past-end → zeros), so the
kernel dequanted fixed memory and B had zero effect on C.  (A was harmless because
its outermost TAP dim was stride-0 — it replayed the same data.)

The proven-working compile_dequant_q4k.py avoids this because its single worker
loop count (384) EQUALS its TAP outermost-product (384), so IRON fuses them into
ONE repeat (no stack).  The fix here mirrors that: a SINGLE worker loop over the
total call count (32×64 = 2048) with FLAT [2048, elem] TAPs so worker-count ==
TAP-outermost → one repeat, no over-read.

Consequences of flattening (K-accumulation can no longer live in the worker):
  - A is reused across N-tiles but differs per K-tile.  A flat TAP can't express
    the k-tile modulo (idx % 64), so the host REPLICATES A's 64 k-chunks 32× into a
    4 MB BO and A's TAP is a flat sequential walk.  (3.9 MB extra host memcpy per
    dispatch — cheap on CPU; dwarfed by the 48 MB dequanted-B intermediate this
    fusion eliminates.)
  - C is emitted as 2048 independent K-tile PARTIALS (one fresh mmul per call, no
    K-accum in the kernel).  The host reads back the 4 MB partial BO and sums the
    64 partials per N-tile to form the full C[16,2048].
  - Each kernel call = one (A_chunk, B_chunk) -> one C_partial.  Elements stay
    small (1024 bf16 / 2304 B / 1024 bf16) — no local-memory OOM.
=========================================================================
"""
import os, sys, time, shutil, subprocess
from pathlib import Path

os.environ.setdefault("PATH",
    os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
set_current_device(NPU2())

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
INC = [str(Path(config.cxx_header_path()).parent), str(PROJ_ROOT)]

# ── Tile / GEMM dimensions ────────────────────────────────────────────────
M, K, N = 16, 4096, 2048
TILE_M, TILE_K, TILE_N = 16, 64, 32
BLOCK_BYTES = 17
BLOCK_ELEMS = 32
PAD_BYTES   = 18                      # host pads each 17-B block to 18 B (17+1)
K_BLOCKS = TILE_K // BLOCK_ELEMS     # 2 blocks per N column within a tile
N_COLS = TILE_N                      # 64
B_TILE_BYTES = N_COLS * K_BLOCKS * PAD_BYTES    # 64 * 2 * 18 = 2304
A_TILE_ELEMS = TILE_M * TILE_K        # 1024
C_TILE_ELEMS = TILE_M * TILE_N        # 1024

# Total independent (N-tile, K-tile) calls = 32 × 64 = 2048.  This is the single
# worker loop count AND the flat TAP outermost dim — they match, so IRON fuses
# them into one BD repeat (no over-read).
N_TILES = N // TILE_N                 # 32
K_TILES = K // TILE_K                 # 64
CALLS = N_TILES * K_TILES             # 2048

SRC = str(PROJ_ROOT / "fst_fused_dequant_gemm_kernel.cc")

# Multi-core: mirror compile_dequant_q4k's proven structure (per-core ObjectFifos
# + per-core TAP offsets).  The single-fifo uint8 fill did not deliver the B BO
# to the kernel (the acquired buffer held a fixed ramp); dequant's multi-core
# uint8 fill works on this machine (the engine relies on it).  8 cores keep the
# per-core BD small (256 calls = 1 MB/core) and the shim-fifo count at 24 (< the
# 32 dequant uses).  256 = 16×16×1.
NUM_CORES = 8
CALLS_PER_CORE = CALLS // NUM_CORES            # 256


@iron.jit
def fused_gemm_op(AB: In, C: Out,
                  *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
                  el: CompileTime[type]):
    # SOLE combined uint8 input (dequant-proven path): a single uint8 ObjectFifo
    # via direct shim->core DOES DMA its BO.  uint8 as a 2nd input (relay or not)
    # never DMAs.  So A (1024 BF16 = 2048 B) + B (2304 B) travel in ONE 4352-B
    # uint8 element; the kernel casts the first 2048 B to BF16 for A.
    AB_TILE = A_TILE_ELEMS * 2 + B_TILE_BYTES     # 4352 uint8
    gemm_k = ExternalFunction(
        "fst_fused_dequant_gemm_16x64x64",
        source_file=SRC,
        arg_types=[
            np.ndarray[(AB_TILE,), np.dtype[np.uint8]],
            np.ndarray[(C_TILE_ELEMS,), np.dtype[el]],
        ],
        include_dirs=INC,
        object_file_name="fst_fused_dequant_gemm_16x64x64.o",
    )

    fifos_AB = [ObjectFifo(np.ndarray[(AB_TILE,), np.dtype[np.uint8]], name=f"fAB{i}", depth=2)
                for i in range(NUM_CORES)]
    fifos_C = [ObjectFifo(np.ndarray[(C_TILE_ELEMS,), np.dtype[el]], name=f"fC{i}", depth=2)
               for i in range(NUM_CORES)]

    def make_core_fn():
        def core_fn(of_ab, of_c, g_k):
            for _ in range_(CALLS_PER_CORE):
                eab = of_ab.acquire(1); ec = of_c.acquire(1)
                g_k(eab, ec)              # dequant B + fresh mmul into C
                of_ab.release(1); of_c.release(1)
        return core_fn

    workers = [
        Worker(make_core_fn(),
               [fifos_AB[i].cons(), fifos_C[i].prod(), gemm_k],
               stack_size=0x2400)
        for i in range(NUM_CORES)
    ]

    # 4D TAPs: outermost 16 -> repeat 15 (≤ [0:255]); NO degenerate size=1 dim
    # (dequant's working TAP [12,16,2,elem] has none).  512 = 16×16×2.
    R0, R1, R2 = 16, 16, 2    # = CALLS_PER_CORE
    def taps_with_offset(elem):
        return [
            TensorAccessPattern(
                (1, CALLS * elem),
                i * CALLS_PER_CORE * elem,
                [R0, R1, R2, elem],
                [R1 * R2 * elem, R2 * elem, elem, 1])
            for i in range(NUM_CORES)
        ]
    taps_AB = taps_with_offset(AB_TILE)
    taps_C = taps_with_offset(C_TILE_ELEMS)
    total_AB = CALLS * AB_TILE

    rt = Runtime()
    with rt.sequence(np.ndarray[(total_AB,), np.dtype[np.uint8]],
                     np.ndarray[(CALLS * C_TILE_ELEMS,), np.dtype[el]]) as (ab, c):
        for w in workers:
            rt.start(w)
        tg = rt.task_group()
        for i in range(NUM_CORES):
            rt.fill(fifos_AB[i].prod(), ab, tap=taps_AB[i], task_group=tg)
            rt.drain(fifos_C[i].cons(), c, tap=taps_C[i], task_group=tg, wait=(i == 0))
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def main():
    print("=== Compile FUSED dequant+GEMM xclbin (flat single-loop) ===\n")
    t_total = time.perf_counter()

    print(f"Target: M={M} K={K} N={N}  tile=({TILE_M},{TILE_K},{TILE_N})  "
          f"calls={CALLS}  B_tile={B_TILE_BYTES}B  "
          f"A_BO={CALLS*A_TILE_ELEMS*2}B  B_BO={CALLS*B_TILE_BYTES}B  "
          f"C_BO={CALLS*C_TILE_ELEMS*2}B")

    design = fused_gemm_op.specialize(M=M, K=K, N=N, el=bfloat16)
    mlir_text = design.as_mlir()
    mlir_path = PROJ_ROOT / "fst_fused_dequant_gemm.mlir"
    with open(mlir_path, 'w') as f:
        f.write(mlir_text)
    print(f"MLIR written to {mlir_path} ({len(mlir_text)} chars)")

    print("\n=== Compiling via IRON spec.compile() ===")
    t0 = time.perf_counter()
    try:
        xclbin_obj, insts_obj = design.compile()
    except Exception as e:
        print(f"FAILED: compile() raised: {e}")
        print(f"=== Total time: {time.perf_counter() - t_total:.1f}s ===")
        return False
    dt = time.perf_counter() - t0

    xclbin_path = PROJ_ROOT / "fst_fused_dequant_gemm.xclbin"
    insts_path = PROJ_ROOT / "fst_fused_dequant_gemm_insts.bin"
    shutil.copy2(xclbin_obj, xclbin_path)
    shutil.copy2(insts_obj, insts_path)
    xclbin_sz = xclbin_path.stat().st_size
    insts_sz = insts_path.stat().st_size
    print(f"\nSUCCESS: {xclbin_path}: {xclbin_sz}B  insts: {insts_sz}B ({dt:.1f}s)")

    print(f"=== Total time: {time.perf_counter() - t_total:.1f}s ===")
    return True


if __name__ == "__main__":
    main()