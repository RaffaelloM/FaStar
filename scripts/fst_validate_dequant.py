#!/usr/bin/env python3
"""Correctness gate for the MXFP4 dequant kernel (fst_dequant_v4).

Packs a synthetic expert (gate [2048,4096], up [2048,4096], down [4096,2048])
with the converter's byte-level packing (_hf_fp4_to_dense_blocks), runs the
NPU dequant kernel, and compares each projection's output to the float32
reference dequant_v4_fp4 (cast to BF16).  Confirms:
  * the dequant kernel's FP4_LUT + E8M0 scale math matches the converter, and
  * the output orientation is B[N, K] row-major (out-major), matching what the
    transposed-B GEMM consumes.
"""
import os
import sys

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault(
    "PEANO_INSTALL_DIR",
    os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"),
)

import numpy as np
import torch
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import NPU2
from aie.utils import set_current_device
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

set_current_device(NPU2())

from fst_converter import (
    DENSE_BLOCK_ELEMS,
    _hf_fp4_to_dense_blocks,
    dequant_v4_fp4,
)
from fst_dequant_v4 import fst_dequant_v4, TOTAL_BLOCKS

OUT_BYTES = TOTAL_BLOCKS * 17            # 13,369,344
OUT_ELEMS = TOTAL_BLOCKS * DENSE_BLOCK_ELEMS  # 25,165,824 BF16


def make_proj(out_dim, in_dim, rng):
    """Return (w_raw, w_shape, s_raw, s_shape) + float32 reference [out, in]."""
    num_groups = in_dim // DENSE_BLOCK_ELEMS   # in/32
    in_half = in_dim // 2
    # Random FP4 nibbles 0..15, packed 2/byte low-first.
    nibbles = rng.integers(0, 16, size=(out_dim, in_dim), dtype=np.uint8)
    w_raw = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)  # [out, in/2]
    # E8M0 scales: realistic exponents for normalized weights (~110..130 =>
    # 2^-17 .. 2^3).  Avoid the 1..254 full range: very large exponents make
    # 6.0 * 2^(e-127) overflow float32 in BOTH the kernel and the reference,
    # which is a degenerate test (inf == inf).  Real DeepSeek scales live in
    # this small band.  Skip 0/255 (kernel maps them to 0).
    s_raw = rng.integers(108, 128, size=(out_dim, num_groups), dtype=np.uint8)
    ref = dequant_v4_fp4(w_raw, s_raw).astype(np.float32)  # [out, in] = B[N, K]
    return (w_raw.tobytes(), (out_dim, in_half), s_raw.tobytes(), (out_dim, num_groups)), ref


def bf16_round(x_f32):
    """Cast float32 -> BF16 -> float32 (round-to-nearest-even via torch)."""
    return torch.from_numpy(x_f32).to(torch.bfloat16).to(torch.float32).numpy()


def main():
    rng = np.random.default_rng(7)

    # Three projections: gate/up [2048,4096], down [4096,2048].
    gate, ref_g = make_proj(2048, 4096, rng)
    up,   ref_u = make_proj(2048, 4096, rng)
    down, ref_d = make_proj(4096, 2048, rng)

    packed = _hf_fp4_to_dense_blocks(*gate) + _hf_fp4_to_dense_blocks(*up) + _hf_fp4_to_dense_blocks(*down)
    assert len(packed) == OUT_BYTES, (len(packed), OUT_BYTES)

    in_t = XRTTensor.from_torch(torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).copy()), device="npu")
    out_t = iron.zeros((OUT_ELEMS,), dtype=bfloat16, device="npu")

    fst_dequant_v4(in_t, out_t, total_blocks=TOTAL_BLOCKS)
    out_np = out_t.numpy().astype(np.float32)

    # Split into 3 projections (8,388,608 BF16 each = out*in).
    def split(arr, out_dim, in_dim):
        return arr.reshape(out_dim, in_dim)

    proj_elems = 2048 * 4096
    npu_g = split(out_np[0:proj_elems], 2048, 4096)
    npu_u = split(out_np[proj_elems:2 * proj_elems], 2048, 4096)
    npu_d = split(out_np[2 * proj_elems:2 * proj_elems + 4096 * 2048], 4096, 2048)

    # Reference cast to BF16 (what the kernel produces).
    ref_g_b = bf16_round(ref_g)
    ref_u_b = bf16_round(ref_u)
    ref_d_b = bf16_round(ref_d)

    results = []
    for name, npu, ref in [("gate", npu_g, ref_g_b), ("up", npu_u, ref_u_b), ("down", npu_d, ref_d_b)]:
        err = np.max(np.abs(npu - ref))
        maxv = float(np.max(np.abs(ref))) + 1e-6
        ok = err <= 0.02 * maxv + 1e-3   # BF16-exact: kernel does scale*nib in BF16, ref in f32->bf16
        print(f"[dequant {name}] shape={npu.shape} max|W|={maxv:.3f} max|err|={err:.4f} => {'PASS' if ok else 'FAIL'}")
        results.append(ok)
        if not ok:
            # Show where it diverges
            diff = np.abs(npu - ref)
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"    worst @ {idx}: npu={npu[idx]:.4f} ref={ref[idx]:.4f}")

    print("\n==== dequant SUMMARY ====", "ALL PASS" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()