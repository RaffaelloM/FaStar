#!/usr/bin/env python3
"""Check FST file: parse header, list all TIDs present in shared directory."""

import struct
import sys
from pathlib import Path

PAGE = 4096
HEADER_FMT = "<4sIIIIIIIIIIIQffQQQQQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SHARED_ENTRY_FMT = "<IHHHHIQQQQQQ"
SHARED_ENTRY_SIZE = struct.calcsize(SHARED_ENTRY_FMT)

TID_NAMES = {
    0: "EMBED",
    1: "OUTPUT_NORM",
    2: "LM_HEAD",
    3: "INPUT_NORM",
    4: "POST_ATTN_NORM",
    5: "Q_PROJ",
    6: "K_PROJ",
    7: "V_PROJ",
    8: "O_PROJ",
    9: "ROUTER",
    10: "ROUTER_BIAS",
    11: "SHARED_GATE",
    12: "SHARED_UP",
    13: "SHARED_DOWN",
    14: "Q_NORM",
    15: "KV_NORM",
}

QTYPE_NAMES = {1: "Q8_0", 2: "BF16", 3: "F32"}


def check_fst(path):
    data = Path(path).read_bytes()
    fsize = len(data)
    print(f"File: {path}")
    print(f"Size: {fsize:,} bytes ({fsize / 1e9:.2f} GB)")

    if fsize < HEADER_SIZE:
        print("ERROR: file too small for header")
        return

    hdr = struct.unpack_from(HEADER_FMT, data, 0)
    magic = hdr[0]
    version = hdr[1]

    print(f"\n=== HEADER ===")
    print(f"Magic:   {magic}")
    print(f"Version: {version}")

    if magic != b"FST\x00":
        print(f"ERROR: bad magic (expected FST\\x00)")
        return

    field_names = [
        "hidden_dim", "num_layers", "num_experts", "top_k",
        "num_q_heads", "num_kv_heads", "head_dim", "expert_inter_dim",
        "num_shared_experts", "_pad0", "vocab_size",
        "rms_eps", "rope_freq_base",
        "shared_dir_offset", "shared_dir_count",
        "expert_bank_offset", "expert_block_bytes", "expert_block_stride",
        "expert_count_total", "_pad1", "_pad2",
    ]
    for i, name in enumerate(field_names):
        if name.startswith("_pad"):
            continue
        print(f"  {name:25s} = {hdr[i + 2]}")

    shared_dir_offset = hdr[15]
    shared_dir_count = hdr[16]
    expert_bank_offset = hdr[17]
    expert_block_bytes = hdr[18]
    expert_block_stride = hdr[19]
    expert_count_total = hdr[20]

    print(f"\n=== SHARED DIRECTORY ({shared_dir_count} entries) ===")
    if shared_dir_count == 0:
        print("  (no shared tensors)")
    elif shared_dir_offset + shared_dir_count * SHARED_ENTRY_SIZE > fsize:
        print(f"  ERROR: directory extends past EOF")
        print(f"  offset={shared_dir_offset}, count={shared_dir_count}, "
              f"end={shared_dir_offset + shared_dir_count * SHARED_ENTRY_SIZE}")
    else:
        tid_counts = {}
        for i in range(shared_dir_count):
            off = shared_dir_offset + i * SHARED_ENTRY_SIZE
            entry = struct.unpack_from(SHARED_ENTRY_FMT, data, off)
            tid, layer_id, sub_id, qtype, ndim = entry[0], entry[1], entry[2], entry[3], entry[4]
            shape0, shape1, shape2 = entry[6], entry[7], entry[8]
            data_off, data_size = entry[9], entry[10]
            tid_name = TID_NAMES.get(tid, f"UNKNOWN({tid})")
            qtype_name = QTYPE_NAMES.get(qtype, f"?{qtype}")

            in_bounds = data_off + data_size <= fsize if data_off > 0 and data_size > 0 else True

            tid_counts[tid] = tid_counts.get(tid, 0) + 1
            if i < 20 or tid < 3 or not in_bounds:
                print(f"  [{i:3d}] TID={tid:2d} ({tid_name:16s}) layer={layer_id:5d} "
                      f"sub={sub_id} qtype={qtype_name:4s} shape=[{shape0},{shape1},{shape2}] "
                      f"data_off={data_off:#x} data_size={data_size:,}"
                      + ("" if in_bounds else " *** OUT OF BOUNDS ***"))

        print(f"\n  ... ({shared_dir_count} total entries)")
        print(f"\n=== TID SUMMARY ===")
        for tid in sorted(tid_counts.keys()):
            name = TID_NAMES.get(tid, f"UNKNOWN({tid})")
            print(f"  TID {tid:2d} ({name:16s}): {tid_counts[tid]:5d} tensors")

        # Check for missing TIDs 0, 1, 2
        print(f"\n=== CRITICAL TID CHECK ===")
        for tid in [0, 1, 2]:
            name = TID_NAMES.get(tid, f"UNKNOWN({tid})")
            if tid in tid_counts:
                print(f"  TID {tid} ({name}): PRESENT ({tid_counts[tid]} tensors)")
            else:
                print(f"  TID {tid} ({name}): *** MISSING ***")

    print(f"\n=== EXPERT BANK ===")
    if expert_bank_offset == 0:
        print("  (no expert bank)")
    else:
        bank_size = fsize - expert_bank_offset
        n_blocks = bank_size // expert_block_stride if expert_block_stride > 0 else 0
        print(f"  expert_bank_offset = {expert_bank_offset:#x}")
        print(f"  expert_block_bytes = {expert_block_bytes:,}")
        print(f"  expert_block_stride = {expert_block_stride:,}")
        print(f"  expert_count_total = {expert_count_total}")
        print(f"  bank_size (to EOF) = {bank_size:,}")
        print(f"  blocks that fit   = {n_blocks}")
        if expert_block_stride > 0:
            expected = expert_count_total * expert_block_stride
            print(f"  expected bank size = {expected:,}")
            if expected != bank_size:
                print(f"  WARNING: bank size mismatch ({bank_size - expected:+,} bytes)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "deepseek_all.fst"
    check_fst(path)
