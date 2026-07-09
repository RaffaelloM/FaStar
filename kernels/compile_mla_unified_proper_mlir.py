#!/usr/bin/env python3
"""Compile ALL MLA kernels into a SINGLE fst_mla_unified.xclbin.

Creates a proper MLIR file with all 6 kernels, each with unique names.
"""
import os, sys, shutil, time, subprocess
from pathlib import Path
import re

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Read individual MLIR files
def read_kernel_mlir(name, mlir_file):
    """Extract the kernel body from an individual MLIR file."""
    with open(mlir_file, 'r') as f:
        content = f.read()

    # Find the aie.device body
    device_start = content.find('aie.device(npu2) {')
    if device_start == -1:
        return None

    # Find the end of the device body (matching closing brace)
    depth = 0
    device_body_start = device_start + len('aie.device(npu2) {')
    for i in range(device_body_start, len(content)):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                device_body_end = i
                break

    kernel_body = content[device_body_start:device_body_end].strip()

    # Make SSA names unique by adding kernel name prefix
    lines = []
    ssa_map = {}
    ssa_counter = 0

    for line in kernel_body.split('\n'):
        if not line.strip():
            continue

        new_line = line
        # Replace SSA values
        def replace_ssa(match):
            ssa_name = match.group(1)
            # Skip if already has kernel suffix or is a special constant
            if name in ssa_name or ssa_name.startswith('c') or ssa_name.startswith('arg'):
                return match.group(0)
            # Create unique name
            if ssa_name not in ssa_map:
                ssa_map[ssa_name] = f'{ssa_name}_{name}'
            return f'%{ssa_map[ssa_name]}'

        new_line = re.sub(r'%(\d+|[a-zA-Z_][a-zA-Z0-9_]*)', replace_ssa, new_line)

        # Replace objectfifo names
        new_line = re.sub(r'@(\w+)', lambda m: f'@{m.group(1)}_{name}' if m.group(1) not in ['Ain', 'Bin', 'Cout'] else f'@{m.group(1)}_{name}', new_line)

        lines.append(new_line)

    return '\n'.join(lines)


def generate_unified_mlir_proper():
    """Generate unified MLIR by combining individual kernel MLIRs."""

    kernels = [
        ("qc", "qc_mlir.txt"),
        ("kvc", "kvc_mlir.txt"),
        ("oa", "oa_mlir.txt"),
        ("ob", "ob_mlir.txt"),
        ("qk", "qk_mlir.txt"),
        ("sv", "sv_mlir.txt"),
    ]

    # First, generate individual MLIR files for each kernel
    print("Generating individual MLIR files...")
    for name, _ in kernels:
        # We'll use the existing compiled xclbins approach
        # But we need the MLIR source
        pass

    # For now, let's create a simpler approach - manually construct the MLIR
    # based on the working template

    mlir = ["""module {
  aie.device(npu2) {
"""]

    # Each kernel needs its own section with unique names
    # For simplicity, let's use the approach of combining the device bodies

    # Read all kernel MLIR bodies
    kernel_bodies = {}
    for name, _ in kernels:
        mlir_file = f"{name}_mlir.txt"
        if os.path.exists(mlir_file):
            body = read_kernel_mlir(name, mlir_file)
            if body:
                kernel_bodies[name] = body

    if len(kernel_bodies) != 6:
        print(f"Warning: Only found {len(kernel_bodies)} MLIR files")
        print("Generating MLIR files first...")

        # Generate MLIR for each kernel
        gen_scripts = {
            "qc": ("8", "4096", "1024"),
            "kvc": ("8", "4096", "512"),
            "oa": ("8", "1024", "4096"),
            "ob": ("8", "1024", "4096"),
            "qk": ("8", "1088", "128"),
            "sv": ("8", "128", "1024"),
        }

        for name, (m, k, n) in gen_scripts.items():
            # Use a simpler approach - compile with IRON and capture MLIR
            pass

    # For now, use the working separate xclbins approach
    # The unified xclbin approach requires more investigation
    return None


def main():
    print("=== MLA Unified XCLBIN Strategy ===\n")
    print("Strategy: Use separate xclbins with permanent contexts")
    print("Each xclbin = 1 permanent hw_context")
    print("Total contexts needed: 5 (MLA) + 5 (FFN) + 3 (other) = 13")
    print("This exceeds the 8 context limit.")
    print("\nConclusion: Must merge xclbins to fit in 8 contexts.")

    print("\n=== Attempting MLIR Merge ===")

    # Try to combine MLIR files
    mlir_files = {
        "qc": "qc_mlir.txt",
        "kvc": "kvc_mlir.txt",
        "oa": "oa_mlir.txt",
        "ob": "ob_mlir.txt",
        "qk": "qk_mlir.txt",
        "sv": "sv_mlir.txt",
    }

    existing_files = [f for f in mlir_files.values() if os.path.exists(f)]
    print(f"Found {len(existing_files)} MLIR files")

    if len(existing_files) == 6:
        # Try to combine them
        combined = ["module {", "  aie.device(npu2) {"]
        for name, mlir_file in mlir_files.items():
            body = read_kernel_mlir(name, mlir_file)
            if body:
                combined.append(f"    // === Kernel: {name} ===")
                for line in body.split('\n'):
                    combined.append(f"    {line}")
        combined.extend(["  }", "}"])

        unified_mlir = '\n'.join(combined)
        mlir_path = PROJ_ROOT / "fst_mla_unified.mlir"
        with open(mlir_path, 'w') as f:
            f.write(unified_mlir)
        print(f"Wrote unified MLIR to {mlir_path}")

        # Try to compile
        cmd = ["aiecc", "--aie-generate-xclbin", "--aie-generate-npu-insts",
               str(mlir_path), "-o", str(PROJ_ROOT / "fst_mla_unified")]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ SUCCESS: Unified xclbin created")
        else:
            print("✗ FAILED:")
            print(result.stderr[-500:])
    else:
        print("Not all MLIR files available. Cannot proceed with merge.")
        print("\nFalling back to: Separate xclbins with LRU eviction (NOT ACCEPTABLE)")


if __name__ == "__main__":
    main()
