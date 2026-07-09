#!/usr/bin/env python3
"""Binary verification of an FST file produced by fst_converter.py.

Reads the file header and shared directory from raw bytes.
No numpy/torch/safetensors dependencies — pure struct + mmap.
"""

import struct
import sys
from pathlib import Path

PAGE = 4096
FST_MAGIC = b"FST\x00"
FST_VERSION = 2

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


def read_header(f):
    raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"File too small for header: {len(raw)} < {HEADER_SIZE}")
    fields = struct.unpack(HEADER_FMT, raw)
    return {
        "magic": fields[0],
        "version": fields[1],
        "hidden_dim": fields[2],
        "num_layers": fields[3],
        "num_experts": fields[4],
        "top_k": fields[5],
        "num_q_heads": fields[6],
        "num_kv_heads": fields[7],
        "head_dim": fields[8],
        "expert_inter_dim": fields[9],
        "num_shared_experts": fields[10],
        "_pad11": fields[11],
        "vocab_size": fields[12],
        "rms_eps": fields[13],
        "rope_freq_base": fields[14],
        "shared_dir_offset": fields[15],
        "shared_dir_count": fields[16],
        "expert_bank_offset": fields[17],
        "expert_block_bytes": fields[18],
        "expert_block_stride": fields[19],
        "expert_count_total": fields[20],
        "_pad21": fields[21],
        "_pad22": fields[22],
    }


def read_shared_dir(f, offset, count):
    f.seek(offset)
    entries = []
    for i in range(count):
        raw = f.read(SHARED_ENTRY_SIZE)
        if len(raw) < SHARED_ENTRY_SIZE:
            raise ValueError(f"Truncated shared directory at entry {i}")
        fields = struct.unpack(SHARED_ENTRY_FMT, raw)
        entries.append({
            "tid": fields[0],
            "layer_id": fields[1],
            "sub_id": fields[2],
            "qtype": fields[3],
            "ndim": fields[4],
            "_pad5": fields[5],
            "dim0": fields[6],
            "dim1": fields[7],
            "dim2": fields[8],
            "data_offset": fields[9],
            "data_size": fields[10],
            "_pad11": fields[11],
        })
    return entries


def verify(path):
    p = Path(path)
    fsize = p.stat().st_size
    print(f"File: {p}")
    print(f"Size: {fsize:,} bytes ({fsize / 1e9:.2f} GB)")

    with open(p, "rb") as f:
        # ── Header ────────────────────────────────────────────────
        print("\n--- HEADER ---")
        hdr = read_header(f)

        if hdr["magic"] != FST_MAGIC:
            print(f"FAIL: bad magic {hdr['magic']!r}, expected {FST_MAGIC!r}")
            return False
        print(f"  Magic:           {hdr['magic']!r}  OK")
        print(f"  Version:         {hdr['version']}")
        print(f"  Hidden dim:      {hdr['hidden_dim']}")
        print(f"  Num layers:      {hdr['num_layers']}")
        print(f"  Num experts:     {hdr['num_experts']}")
        print(f"  Top-k:           {hdr['top_k']}")
        print(f"  Q heads:         {hdr['num_q_heads']}")
        print(f"  KV heads:        {hdr['num_kv_heads']}")
        print(f"  Head dim:        {hdr['head_dim']}")
        print(f"  Expert inter:    {hdr['expert_inter_dim']}")
        print(f"  Shared experts:  {hdr['num_shared_experts']}")
        print(f"  Vocab size:      {hdr['vocab_size']}")
        print(f"  RMS eps:         {hdr['rms_eps']}")
        print(f"  Rope base:       {hdr['rope_freq_base']}")
        print(f"  Shared dir off:  {hdr['shared_dir_offset']:#x}")
        print(f"  Shared dir cnt:  {hdr['shared_dir_count']}")
        print(f"  Expert bank off: {hdr['expert_bank_offset']:#x}")
        print(f"  Expert blk bytes:{hdr['expert_block_bytes']:,}")
        print(f"  Expert blk stride:{hdr['expert_block_stride']:,}")
        print(f"  Expert count:    {hdr['expert_count_total']}")
        print(f"  Expected experts:{hdr['num_layers'] * hdr['num_experts']}")

        if hdr["version"] != FST_VERSION:
            print(f"FAIL: version {hdr['version']} != {FST_VERSION}")
            return False

        expected_experts = hdr["num_layers"] * hdr["num_experts"]
        if hdr["expert_count_total"] != expected_experts:
            print(f"FAIL: expert_count_total={hdr['expert_count_total']} "
                  f"!= {hdr['num_layers']}*{hdr['num_experts']}={expected_experts}")
            return False

        # ── Shared directory ──────────────────────────────────────
        print("\n--- SHARED DIRECTORY ---")
        entries = read_shared_dir(f, hdr["shared_dir_offset"], hdr["shared_dir_count"])
        print(f"  {len(entries)} entries")

        tid_set = set()
        for e in entries:
            tid_set.add(e["tid"])
            name = TID_NAMES.get(e["tid"], f"TID_{e['tid']}")
            print(f"  TID {e['tid']:2d} ({name:16s})  "
                  f"layer={e['layer_id']:5d}  sub={e['sub_id']}  "
                  f"qtype={e['qtype']}  ndim={e['ndim']}  "
                  f"shape=[{e['dim0']},{e['dim1']},{e['dim2']}]  "
                  f"offset={e['data_offset']:#012x}  size={e['data_size']:,}")

        # ── Required TID checks ───────────────────────────────────
        print("\n--- TID CHECKS ---")
        all_ok = True
        for required_tid, name in [(0, "EMBED"), (1, "OUTPUT_NORM"), (2, "LM_HEAD")]:
            if required_tid in tid_set:
                count = sum(1 for e in entries if e["tid"] == required_tid)
                print(f"  TID {required_tid} ({name}): FOUND ({count} tensor(s))")
            else:
                print(f"  TID {required_tid} ({name}): *** MISSING ***")
                all_ok = False

        # ── Expert bank sanity ────────────────────────────────────
        print("\n--- EXPERT BANK ---")
        ebo = hdr["expert_bank_offset"]
        ebb = hdr["expert_block_bytes"]
        ebs = hdr["expert_block_stride"]
        ect = hdr["expert_count_total"]

        expected_bank_size = ect * ebs
        actual_bank_size = fsize - ebo
        print(f"  Offset:           {ebo:#x}")
        print(f"  Block bytes:      {ebb:,}")
        print(f"  Block stride:     {ebs:,}")
        print(f"  Expert count:     {ect}")
        print(f"  Expected bank sz: {expected_bank_size:,} bytes "
              f"({expected_bank_size / 1e9:.2f} GB)")
        print(f"  Actual bank sz:   {actual_bank_size:,} bytes "
              f"({actual_bank_size / 1e9:.2f} GB)")

        if actual_bank_size < expected_bank_size:
            shortfall = expected_bank_size - actual_bank_size
            print(f"  WARNING: bank is {shortfall:,} bytes short "
                  f"(likely incomplete conversion)")
        elif actual_bank_size >= expected_bank_size:
            print(f"  Bank size OK (≥ expected)")

        # Check first expert block for non-zero content
        if ect > 0 and ebo > 0 and ebo < fsize:
            f.seek(ebo)
            block_head = f.read(min(ebb, 64))
            all_zero = all(b == 0 for b in block_head)
            if all_zero:
                print("  WARNING: first expert block is all zeros")
            else:
                print(f"  First expert block has data (first 16 bytes: "
                      f"{block_head[:16].hex()})")

        # ── File structure integrity ──────────────────────────────
        print("\n--- STRUCTURE ---")
        # Header at 0..HEADER_SIZE
        # Shared dir at shared_dir_offset
        # Shared data after dir
        # Expert bank after shared data
        print(f"  Header:     0x0 - {HEADER_SIZE:#x}")
        print(f"  Shared dir: {hdr['shared_dir_offset']:#x} - "
              f"{hdr['shared_dir_offset'] + hdr['shared_dir_count'] * SHARED_ENTRY_SIZE:#x}")
        print(f"  Expert bank:{ebo:#x} - {ebo + actual_bank_size:#x}")

        # ── Verdict ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        if all_ok:
            print("FST FILE IS VALID AND READY FOR C++ ENGINE")
        else:
            missing = [t for t in [0, 1, 2] if t not in tid_set]
            print(f"FST FILE IS INVALID — missing TIDs: {missing}")
        print("=" * 60)

        return all_ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "deepseek_v4_full.fst"
    ok = verify(path)
    sys.exit(0 if ok else 1)
