#!/usr/bin/env python3
"""Verify that Embedding (TID=0) and LM Head (TID=2) weights inside an FST
file are real (non-zero), not zero-initialized.

Reads the FST header + shared directory from raw bytes, locates the TID=0
and TID=2 entries, mmap's their BF16 data, and prints the first 10 values
as float32. Exits non-zero if either tensor is entirely zero.

Usage:  python3 verify_fst_weights.py [path.fst]
        (default: dspark_draft.fst)
"""

import struct
import sys
import mmap
from pathlib import Path

import numpy as np

PAGE = 4096
FST_MAGIC = b"FST\x00"

HEADER_FMT = "<4sIIIIIIIIIIIQffQQQQQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SHARED_ENTRY_FMT = "<IHHHHIQQQQQQ"
SHARED_ENTRY_SIZE = struct.calcsize(SHARED_ENTRY_FMT)

TID_EMBED = 0
TID_LM_HEAD = 2
TID_OUTPUT_NORM = 1
GLOBAL_LAYER = 0xFFFF
QTYPE_BF16 = 2


def read_header(f):
    raw = f.read(HEADER_SIZE)
    fields = struct.unpack(HEADER_FMT, raw)
    return {
        "magic": fields[0],
        "version": fields[1],
        "vocab_size": fields[12],
        "shared_dir_offset": fields[15],
        "shared_dir_count": fields[16],
    }


def read_shared_dir(f, offset, count):
    f.seek(offset)
    entries = []
    for _ in range(count):
        raw = f.read(SHARED_ENTRY_SIZE)
        e = struct.unpack(SHARED_ENTRY_FMT, raw)
        entries.append({
            "tid": e[0], "layer_id": e[1], "sub_id": e[2],
            "qtype": e[3], "ndim": e[4],
            "d0": e[6], "d1": e[7], "d2": e[8],
            "data_offset": e[9], "data_size": e[10],
        })
    return entries


def bf16_bytes_to_f32(raw, count):
    """Decode `count` BF16 values (little-endian uint16) to float32."""
    u16 = np.frombuffer(raw[:count * 2], dtype=np.uint16)
    f32 = np.zeros(u16.shape, dtype=np.float32)
    f32.view(np.uint32)[:] = np.uint32(u16) << 16
    return f32


def find_entry(entries, tid):
    for e in entries:
        if e["tid"] == tid and e["layer_id"] == GLOBAL_LAYER:
            return e
    return None


def main(path):
    p = Path(path)
    fsize = p.stat().st_size
    print(f"File: {p}  ({fsize:,} bytes)")

    with open(p, "rb") as f:
        hdr = read_header(f)
        if hdr["magic"] != FST_MAGIC:
            print(f"FAIL: bad magic {hdr['magic']!r}")
            return 1
        entries = read_shared_dir(f, hdr["shared_dir_offset"], hdr["shared_dir_count"])

    ok = True
    for tid, name in [(TID_EMBED, "Embedding"), (TID_LM_HEAD, "LM Head")]:
        e = find_entry(entries, tid)
        if e is None:
            print(f"\n{name} (TID={tid}): *** MISSING from shared directory ***")
            ok = False
            continue

        off, size = e["data_offset"], e["data_size"]
        dims = (e["d0"], e["d1"]) if e["ndim"] >= 2 else (e["d0"],)
        print(f"\n{name} (TID={tid}): layer={e['layer_id']} qtype={e['qtype']} "
              f"shape={dims} offset={off:#x} size={size:,}")

        with open(p, "rb") as f:
            f.seek(off)
            raw = f.read(min(size, 1 << 20))

        n = min(10, len(raw) // 2)
        vals = bf16_bytes_to_f32(raw, n)
        print(f"  first {n} values: {vals.tolist()}")

        nz = int(np.count_nonzero(np.frombuffer(raw, dtype=np.uint8)))
        total = len(raw)
        print(f"  non-zero bytes in first {total:,} data bytes: {nz}/{total}")
        if nz == 0:
            print(f"  *** {name} IS ALL ZEROS — model will not produce coherent text ***")
            ok = False
        else:
            print(f"  {name}: OK (non-zero)")

    print("\n" + "=" * 60)
    print("RESULT: " + ("PASS — weights are real" if ok else "FAIL — weights are zero/missing"))
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dspark_draft.fst"
    sys.exit(main(path))
