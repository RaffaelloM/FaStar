#!/usr/bin/env python3
"""Compile ALL MLA kernels into a SINGLE fst_mla_unified.xclbin.

Creates a single MLIR file with all 6 kernels in one aie.device block,
then compiles with aiecc to produce one xclbin with multiple entry points.
"""
import os, sys, shutil, time, subprocess
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIRS = [
    str(PROJ_ROOT),
    str(Path("/home/raffaele/.local/lib/python3.14/site-packages/llvm-aie/lib/clang/18/include"))
]

def generate_unified_mlir():
    """Generate a single MLIR file with all 6 MLA kernels."""

    # MLIR header
    mlir = ["""module {
  aie.device(npu2) {
"""]

    # Kernel 1: qc (8 x 4096 x 1024)
    mlir.append("""
    // === Kernel: qc (Q compression) ===
    %A_qc = aie.buffer(<{8*64}>, T:bf16)
    %B_qc = aie.buffer(<{64*64}>, T:bf16)
    %C_qc = aie.buffer(<{8*64}>, T:bf16)
    %mm_qc = aie.mmul(8, 64, 64, bf16, bf16, accauto)
    %core_qc = aie.core(%mm_qc)
    aie.flow(%A_qc, %core_qc[0])
    aie.flow(%B_qc, %core_qc[1])
    aie.flow(%core_qc[2], %C_qc)
    aie.external_function("qc_op") {
        aie.ext_function(%core_qc[0], %core_qc[1], %core_qc[2])
    }
""")

    # Kernel 2: kvc (8 x 4096 x 512)
    mlir.append("""
    // === Kernel: kvc (KV compression) ===
    %A_kvc = aie.buffer(<{8*64}>, T:bf16)
    %B_kvc = aie.buffer(<{64*64}>, T:bf16)
    %C_kvc = aie.buffer(<{8*64}>, T:bf16)
    %mm_kvc = aie.mmul(8, 64, 64, bf16, bf16, accauto)
    %core_kvc = aie.core(%mm_kvc)
    aie.flow(%A_kvc, %core_kvc[0])
    aie.flow(%B_kvc, %core_kvc[1])
    aie.flow(%core_kvc[2], %C_kvc)
    aie.external_function("kvc_op") {
        aie.ext_function(%core_kvc[0], %core_kvc[1], %core_kvc[2])
    }
""")

    # Kernel 3: oa (8 x 1024 x 4096)
    mlir.append("""
    // === Kernel: oa (Output projection A) ===
    %A_oa = aie.buffer(<{8*64}>, T:bf16)
    %B_oa = aie.buffer(<{64*64}>, T:bf16)
    %C_oa = aie.buffer(<{8*64}>, T:bf16)
    %mm_oa = aie.mmul(8, 64, 64, bf16, bf16, accauto)
    %core_oa = aie.core(%mm_oa)
    aie.flow(%A_oa, %core_oa[0])
    aie.flow(%B_oa, %core_oa[1])
    aie.flow(%core_oa[2], %C_oa)
    aie.external_function("oa_op") {
        aie.ext_function(%core_oa[0], %core_oa[1], %core_oa[2])
    }
""")

    # Kernel 4: ob (8 x 1024 x 4096)
    mlir.append("""
    // === Kernel: ob (Output projection B) ===
    %A_ob = aie.buffer(<{8*64}>, T:bf16)
    %B_ob = aie.buffer(<{64*64}>, T:bf16)
    %C_ob = aie.buffer(<{8*64}>, T:bf16)
    %mm_ob = aie.mmul(8, 64, 64, bf16, bf16, accauto)
    %core_ob = aie.core(%mm_ob)
    aie.flow(%A_ob, %core_ob[0])
    aie.flow(%B_ob, %core_ob[1])
    aie.flow(%core_ob[2], %C_ob)
    aie.external_function("ob_op") {
        aie.ext_function(%core_ob[0], %core_ob[1], %core_ob[2])
    }
""")

    # Kernel 5: qk (8 x 1088 x 128)
    mlir.append("""
    // === Kernel: qk (QK attention scores) ===
    %A_qk = aie.buffer(<{8*64}>, T:bf16)
    %B_qk = aie.buffer(<{64*64}>, T:bf16)
    %C_qk = aie.buffer(<{8*64}>, T:bf16)
    %mm_qk = aie.mmul(8, 64, 64, bf16, bf16, accauto)
    %core_qk = aie.core(%mm_qk)
    aie.flow(%A_qk, %core_qk[0])
    aie.flow(%B_qk, %core_qk[1])
    aie.flow(%core_qk[2], %C_qk)
    aie.external_function("qk_op") {
        aie.ext_function(%core_qk[0], %core_qk[1], %core_qk[2])
    }
""")

    # Kernel 6: sv (8 x 128 x 1024)
    mlir.append("""
    // === Kernel: sv (SV attention aggregation) ===
    %A_sv = aie.buffer(<{8*64}>, T:bf16)
    %B_sv = aie.buffer(<{64*64}>, T:bf16)
    %C_sv = aie.buffer(<{8*64}>, T:bf16)
    %mm_sv = aie.mmul(8, 64, 64, bf16, bf16, accauto)
    %core_sv = aie.core(%mm_sv)
    aie.flow(%A_sv, %core_sv[0])
    aie.flow(%B_sv, %core_sv[1])
    aie.flow(%core_sv[2], %C_sv)
    aie.external_function("sv_op") {
        aie.ext_function(%core_sv[0], %core_sv[1], %core_sv[2])
    }
""")

    # Close the device and module
    mlir.append("""  }
}
""")

    return '\n'.join(mlir)


def main():
    print("=== Compiling MLA Unified XCLBIN (Single MLIR) ===\n")
    t_total = time.perf_counter()

    # Generate unified MLIR
    mlir_content = generate_unified_mlir()
    mlir_path = PROJ_ROOT / "fst_mla_unified.mlir"
    with open(mlir_path, 'w') as f:
        f.write(mlir_content)
    print(f"MLIR written to {mlir_path} ({len(mlir_content)} chars)")

    # Compile with aiecc
    print("\n=== Compiling with aiecc ===")
    cmd = [
        "aiecc",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        str(mlir_path),
        "-o", str(PROJ_ROOT / "fst_mla_unified")
    ]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t0

    if result.stdout:
        print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    # Check results
    xclbin_file = PROJ_ROOT / "fst_mla_unified.xclbin"
    insts_src = PROJ_ROOT / "insts.bin"
    insts_dst = PROJ_ROOT / "fst_mla_unified_insts.bin"

    if xclbin_file.exists():
        print(f"\n✓ SUCCESS:")
        print(f"  {xclbin_file}: {xclbin_file.stat().st_size}B")
        if insts_src.exists():
            shutil.copy2(insts_src, insts_dst)
            print(f"  {insts_dst}: {insts_dst.stat().st_size}B")
    else:
        print(f"\n✗ FAILED - xclbin not created")

    total_dt = time.perf_counter() - t_total
    print(f"\n=== Total time: {total_dt:.1f}s ===")


if __name__ == "__main__":
    main()
