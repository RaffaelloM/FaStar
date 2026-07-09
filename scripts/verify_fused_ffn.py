#!/usr/bin/env python3
"""Standalone verify for fst_fused_dequant_gemm.xclbin (flat single-loop design).

  (1) CPU DEQUANT-ADDRESSING CHECK — replicates the kernel's 32-byte-padded
      block addressing + col-major B_buf scatter; checks vs fst_dequant_v4.
  (2) FULL GEMM cosine vs a CPU ref over random MXFP4 blocks.

Flat design: the worker is a single loop over 2048 (N-tile × K-tile) calls; each
call produces ONE fresh K-tile partial C[16,64] = A[16,64] @ dequant(B[64,64]).
The host:
  - REPLICATES A: A_BO[idx] = A's k-chunk (idx % 64), idx = n_tile*64 + k_tile.
  - Sums the 64 K-tile partials per N-tile -> full C[16, 2048].
"""
import os, sys, subprocess, ctypes
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

PROJ = Path(os.path.dirname(os.path.abspath(__file__)))
XCLBIN = PROJ / "fst_fused_dequant_gemm.xclbin"
INSTS = PROJ / "fst_fused_dequant_gemm_insts.bin"

M, K, N = 16, 4096, 2048
TILE_M, TILE_K, TILE_N = 16, 64, 32
BLOCK_BYTES = 17
BLOCK_ELEMS = 32
PAD_BYTES   = 18
K_BLOCKS = TILE_K // BLOCK_ELEMS      # 2
N_COLS = TILE_N                       # 64
B_TILE_BYTES = N_COLS * K_BLOCKS * PAD_BYTES    # 2304
A_TILE_ELEMS = TILE_M * TILE_K        # 1024
C_TILE_ELEMS = TILE_M * TILE_N        # 1024
N_TILES = N // TILE_N                 # 32
K_TILES = K // TILE_K                 # 64
CALLS = N_TILES * K_TILES             # 2048
A_BYTES = TILE_M * TILE_K * 2        # 2048
AB_TILE = A_BYTES + B_TILE_BYTES      # 4352

FP4_LUT = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                    0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def e8m0_to_bf16(scale_u8):
    s = int(scale_u8) & 0xFF
    if s == 0 or s == 255:
        return np.float32(0.0)
    u = s << 7
    bf16_bits = u & 0xFFFF
    f32_bits = bf16_bits << 16
    return np.float32(ctypes.c_float(ctypes.c_uint32(f32_bits).value).value)


def dequant_block_v4(block):
    scale = e8m0_to_bf16(block[0])
    vals = np.empty(32, dtype=np.float32)
    for i in range(16):
        byte = block[1 + i]
        vals[i * 2]     = FP4_LUT[byte & 0x0F] * scale
        vals[i * 2 + 1] = FP4_LUT[(byte >> 4) & 0x0F] * scale
    return vals


def kernel_block_for(k_local, n_col, tile_blocks):
    k_block = k_local // BLOCK_ELEMS
    block_idx = n_col * K_BLOCKS + k_block
    return tile_blocks[block_idx], k_block


def kernel_nibble_for(block, k_local):
    k_elem = k_local % BLOCK_ELEMS
    byte = block[1 + (k_elem // 2)]
    if k_elem % 2 == 0:
        return byte & 0x0F
    return (byte >> 4) & 0x0F


def addressing_check(rng):
    print("=== (1) Dequant addressing check (kernel vs fst_dequant_v4) ===")
    blocks = []
    for _ in range(128):
        blk = rng.integers(0, 256, BLOCK_BYTES, dtype=np.uint8)
        blk[0] = np.uint8(rng.integers(120, 134))
        blocks.append(blk)
    packed = np.zeros(B_TILE_BYTES, dtype=np.uint8)
    for n_col in range(N_COLS):
        for k_block in range(K_BLOCKS):
            bidx = n_col * K_BLOCKS + k_block
            packed[bidx * PAD_BYTES: bidx * PAD_BYTES + BLOCK_BYTES] = blocks[bidx]

    B_ref = np.zeros((TILE_K, TILE_N), dtype=np.float32)
    for n_col in range(N_COLS):
        for k_block in range(K_BLOCKS):
            bidx = n_col * K_BLOCKS + k_block
            vals = dequant_block_v4(blocks[bidx])
            for j in range(32):
                B_ref[k_block * 32 + j, n_col] = vals[j]

    tile_blocks = [packed[bidx * PAD_BYTES: bidx * PAD_BYTES + BLOCK_BYTES]
                   for bidx in range(128)]
    B_kernel = np.zeros((TILE_K, TILE_N), dtype=np.float32)
    for k_local in range(TILE_K):
        for n_col in range(N_COLS):
            blk, k_block = kernel_block_for(k_local, n_col, tile_blocks)
            nib = kernel_nibble_for(blk, k_local)
            scale = e8m0_to_bf16(blk[0])
            val = FP4_LUT[nib] * scale
            ii = k_local // 8; rr = k_local % 8
            nt = n_col // 8;   n  = n_col % 8
            kk = ii * 8 + rr
            nn = nt * 8 + n
            B_kernel[kk, nn] = val

    if np.array_equal(B_kernel, B_ref):
        print("  PASS: kernel addressing reproduces fst_dequant_v4 dequant exactly.")
        return True
    diff = np.abs(B_kernel - B_ref)
    print(f"  FAIL: max abs diff = {diff.max()}")
    return False


def cpu_ref(A_bf16, all_blocks):
    """C[16,2048] = A[16,4096] @ dequant(B)[4096,2048] (float32)."""
    B_full = np.zeros((K, N), dtype=np.float32)
    for n_tile in range(N_TILES):
        for k_tile in range(K_TILES):
            blocks = all_blocks[(n_tile, k_tile)]
            for n_col in range(N_COLS):
                for k_block in range(K_BLOCKS):
                    bidx = n_col * K_BLOCKS + k_block
                    vals = dequant_block_v4(blocks[bidx])
                    for j in range(32):
                        B_full[k_tile * TILE_K + k_block * 32 + j,
                               n_tile * TILE_N + n_col] = vals[j]
    return A_bf16.astype(np.float32) @ B_full


def pack_full_B(all_blocks):
    """Flat 8 MB BO: tile idx = n_tile*64 + k_tile at byte idx*4096."""
    total = CALLS * B_TILE_BYTES
    buf = np.zeros(total, dtype=np.uint8)
    for n_tile in range(N_TILES):
        for k_tile in range(K_TILES):
            base = (n_tile * K_TILES + k_tile) * B_TILE_BYTES
            blocks = all_blocks[(n_tile, k_tile)]
            for n_col in range(N_COLS):
                for k_block in range(K_BLOCKS):
                    bidx = n_col * K_BLOCKS + k_block
                    buf[base + bidx * PAD_BYTES:
                        base + bidx * PAD_BYTES + BLOCK_BYTES] = blocks[bidx]
    return buf


def pack_AB(A_bf16, B_packed):
    """Combined AB BO: per call idx, [A_chunk[idx%64] (2048 B) | B_tile[idx] (2304 B)].
    A_chunk[k] = A[:, k*64:(k+1)*64] row-major (1024 bf16 = 2048 B)."""
    A_f = A_bf16.astype(np.float32)
    k_chunks = np.empty((K_TILES, A_BYTES), dtype=np.uint8)
    for k in range(K_TILES):
        chunk = A_f[:, k * TILE_K:(k + 1) * TILE_K].reshape(-1).astype(bfloat16)
        k_chunks[k] = np.frombuffer(chunk.tobytes(), dtype=np.uint8)
    B_tiles = B_packed.reshape(CALLS, B_TILE_BYTES)
    ab = np.empty((CALLS, AB_TILE), dtype=np.uint8)
    for idx in range(CALLS):
        ab[idx, :A_BYTES] = k_chunks[idx % K_TILES]
        ab[idx, A_BYTES:] = B_tiles[idx]
    return ab.reshape(-1)


def sum_partials(C_bo):
    """C_bo: 2048 partials × 1024 bf16. Sum the 64 K-tile partials per N-tile."""
    C_bo = C_bo.reshape(CALLS, C_TILE_ELEMS).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    for n_tile in range(N_TILES):
        acc = np.zeros(C_TILE_ELEMS, dtype=np.float32)
        for k_tile in range(K_TILES):
            acc += C_bo[n_tile * K_TILES + k_tile]
        C[:, n_tile * TILE_N:(n_tile + 1) * TILE_N] = acc.reshape(M, TILE_N)
    return C


def npu_run(A_bf16, B_packed):
    print("=== (3) NPU run via pyxrt + aiebu-asm ===")
    if not XCLBIN.exists() or not INSTS.exists():
        print(f"  SKIP: missing xclbin/insts")
        return None, "missing artifacts"
    try:
        import pyxrt
    except Exception as e:
        print(f"  SKIP: pyxrt unavailable ({e})")
        return None, "no pyxrt"

    elf_path = Path("/tmp/fused_verify_elf.out")
    try:
        r = subprocess.run(
            ["aiebu-asm", "-t", "aie2txn", "-c", str(INSTS), "-o", str(elf_path)],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not elf_path.exists():
            print(f"  SKIP: aiebu-asm failed: {r.stderr[:200]}")
            return None, "aiebu-asm failed"
        print(f"  aiebu-asm: ELF {elf_path.stat().st_size}B")
    except Exception as e:
        return None, f"aiebu-asm exception: {e}"

    try:
        dev = pyxrt.device(0)
        xb = pyxrt.xclbin(str(XCLBIN))
        dev.register_xclbin(xb)
        ctx = pyxrt.hw_context(dev, xb.get_uuid())
        elf = pyxrt.elf(str(elf_path))
        mod = pyxrt.module(elf)
        kname = next((k.get_name() for k in xb.get_kernels()
                     if k.get_name().rfind("MLIR_AIE", 0) == 0), None)
        if kname is None:
            return None, "no MLIR_AIE kernel"
        print(f"  kernel: {kname}")
        krnl = pyxrt.ext.kernel(ctx, mod, kname)
        grp = krnl.group_id(2)
        ab_sz = CALLS * AB_TILE
        c_sz = CALLS * C_TILE_ELEMS * 2

        def dispatch(ab_np):
            bo_ab = pyxrt.ext.bo(dev, ab_sz)
            bo_c = pyxrt.ext.bo(dev, c_sz)
            bo_ab.write(ab_np.tobytes(), 0)
            bo_ab.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            run = krnl(2, 0, 0, bo_ab, bo_c)
            st = run.wait()
            bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            C_bo = np.frombuffer(bo_c.read(c_sz, 0), dtype=bfloat16)
            return C_bo, st

        # --- B-delivery probe: A impulse at call 0 (A[0,0]=1), zero-B vs 0xFF-B
        # in the B part of the AB element.  C partial[0,:8] MUST differ now. ---
        def probe_ab(B_fill):
            ab = np.zeros((CALLS, AB_TILE), dtype=np.uint8)
            ab[0, 0:2] = np.frombuffer(
                np.array([bfloat16(1.0)], dtype=bfloat16).tobytes(), dtype=np.uint8)
            ab[:, A_BYTES:] = B_fill
            return ab.reshape(-1)
        C_z, _ = dispatch(probe_ab(np.zeros((CALLS, B_TILE_BYTES), dtype=np.uint8)))
        C_f, st = dispatch(probe_ab(np.full((CALLS, B_TILE_BYTES), 0xFF, dtype=np.uint8)))
        b_read = not np.array_equal(C_z, C_f)
        print(f"  run state: {st}")
        print(f"  B read by kernel: {b_read}  (zero-B vs 0xFF-B partial[0,:8] "
              f"z={C_z[:8]} ff={C_f[:8]})")
        if not b_read:
            print("  -> B stream still not reaching the kernel.")
            return None, "B stream not connected"

        # --- full run ---
        ab_full = pack_AB(A_bf16, B_packed)
        C_bo, _ = dispatch(ab_full)
        C_npu = sum_partials(C_bo)
        return C_npu, "ok"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"{type(e).__name__}: {e}"


def main():
    rng = np.random.default_rng(1234)
    ok1 = addressing_check(rng)

    print("\n=== (2) Full GEMM CPU reference (random MXFP4) ===")
    A_bf16 = (rng.standard_normal((M, K)) * 0.5).astype(bfloat16)
    all_blocks = {}
    for n_tile in range(N_TILES):
        for k_tile in range(K_TILES):
            blks = []
            for _ in range(128):
                blk = rng.integers(0, 256, BLOCK_BYTES, dtype=np.uint8)
                blk[0] = np.uint8(rng.integers(120, 134))
                blks.append(blk)
            all_blocks[(n_tile, k_tile)] = blks
    C_ref = cpu_ref(A_bf16, all_blocks)
    print(f"  C_ref shape={C_ref.shape} |C_ref|={np.linalg.norm(C_ref):.4f}")

    B_packed = pack_full_B(all_blocks)
    print(f"  B_packed {B_packed.nbytes}B (expected {CALLS * B_TILE_BYTES})")

    C_npu, npu_status = npu_run(A_bf16, B_packed)
    cos = -1.0
    if C_npu is not None:
        a = C_npu.reshape(-1); b = C_ref.reshape(-1)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  NPU vs CPU cosine: {cos:.5f}")
        verified = cos > 0.99
    else:
        verified = False
        print(f"  NPU cosine not measured ({npu_status}).")

    print("\n=== SUMMARY ===")
    print(f"  addressing check:  {'PASS' if ok1 else 'FAIL'}")
    print(f"  NPU cosine:        {cos if cos >= 0 else 'not measured'}")
    print(f"  verified (>0.99):  {verified}")
    return ok1 and verified


if __name__ == "__main__":
    sys.exit(0 if main() else 1)