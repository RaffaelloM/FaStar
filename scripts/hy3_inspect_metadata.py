#!/usr/bin/env python3
"""Inspect Hunyuan-3.0 GGUF metadata WITHOUT downloading the full ~145 GB file.

Strategy: HTTP Range-fetch only the header (KV metadata + tensor-info table, which
precedes tensor data in the GGUF layout), write to a temp partial file, and parse it
with gguf.GGUFReader. We never access tensor data slices, so a truncated file is fine
as long as it covers the complete header (tensor data_offset is the header boundary).

Prints:
  * header (magic / version / n_tensors / n_kv)
  * every metadata KV field (key = value, type)  -> architecture hyperparams
  * every tensor name with shape / ggml type / byte size / data_offset
  * a per-prefix grouping of tensor names (blk. / nextn. / token_embd / output / ...)
  * projected .fst expert storage size (for disk-budget assessment)
"""
import os
import sys
import requests
import gguf

URL = "https://huggingface.co/satgeze/Hy3-1M-GGUF/resolve/main/hy3-1M-MTP-Q3_K_M.gguf"
PARTIAL = "/tmp/hy3_head.bin"
INITIAL = 16 * 1024 * 1024   # 16 MB header window; grow if needed
MAXIMUM  = 256 * 1024 * 1024 # 256 MB cap


def fetch_header(nbytes):
    """Range-fetch the first nbytes bytes; return bytes fetched."""
    headers = {"Range": f"bytes=0-{nbytes - 1}"}
    with requests.get(URL, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        chunk = bytearray()
        for b in r.iter_content(8 * 1024 * 1024):
            chunk.extend(b)
            if len(chunk) >= nbytes:
                break
    data = bytes(chunk[:nbytes])
    with open(PARTIAL, "wb") as f:
        f.write(data)
    return len(data)


def main():
    # --- HEAD: confirm reachability + total size ---
    h = requests.head(URL, allow_redirects=True, timeout=60)
    print(f"HTTP HEAD status={h.status_code} len={h.headers.get('Content-Length')} accept-ranges={h.headers.get('Accept-Ranges')}")
    total = int(h.headers.get("Content-Length", "0"))
    if total:
        print(f"Total GGUF size: {total/1e9:.2f} GB ({total} bytes)")

    # --- Range fetch + parse, growing the window if the header is larger ---
    nbytes = INITIAL
    reader = None
    while nbytes <= MAXIMUM:
        got = fetch_header(nbytes)
        print(f"\nFetched header window: {got/1e6:.1f} MB -> {PARTIAL}")
        try:
            reader = gguf.GGUFReader(PARTIAL)
            break
        except Exception as e:
            print(f"  GGUFReader on {got} bytes failed: {e!r}; doubling window")
            nbytes *= 2
    if reader is None:
        print("ERROR: could not parse header within 256 MB window", file=sys.stderr)
        sys.exit(1)

    hdr = reader.header
    print(f"\n=== GGUF HEADER ===")
    print(f"  magic={hdr.magic!r} version={hdr.version} n_tensors={hdr.n_tensors} n_kv={hdr.n_kv}")

    # --- metadata KV fields (architecture) ---
    print(f"\n=== METADATA KV ({hdr.n_kv} fields) ===")
    for key, field in reader.fields.items():
        # field.parts is a list of np arrays; field.types the gguf type codes
        val = field.parts
        try:
            # Most scalars: single-part array; strings: uint8 array -> decode
            import numpy as np
            if len(val) == 1:
                arr = val[0]
                if arr.dtype == np.uint8:
                    s = bytes(arr).decode("utf-8", "replace")
                    print(f"  {key} = {s!r}   [str]")
                else:
                    print(f"  {key} = {arr.tolist()}   [{arr.dtype}]")
            else:
                # multi-part (e.g. arrays of ints)
                flat = [a.tolist() for a in val]
                print(f"  {key} = {flat}   [array/{len(val)}parts]")
        except Exception as e:
            print(f"  {key} = <unparseable: {e!r}>")

    # --- tensors ---
    print(f"\n=== TENSORS ({hdr.n_tensors}) ===")
    from collections import defaultdict
    groups = defaultdict(list)
    total_expert_bytes = 0
    total_attn_bytes = 0
    total_other_bytes = 0
    expert_keys = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shard", "ffn_up_shard", "ffn_down_shard")
    attn_keys = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
    for i, t in enumerate(reader.tensors):
        name = t.name
        shape = list(t.shape)
        ttype = str(t.tensor_type) if hasattr(t.tensor_type, "__str__") else t.tensor_type
        nbytes_t = int(t.n_bytes)
        prefix = name.split(".")[0] if "." in name else name
        groups[prefix].append(name)
        if any(k in name for k in expert_keys):
            total_expert_bytes += nbytes_t
        elif any(k in name for k in attn_keys):
            total_attn_bytes += nbytes_t
        else:
            total_other_bytes += nbytes_t
        if i < 80:
            print(f"  [{i:4d}] {name:48s} shape={shape} type={ttype} bytes={nbytes_t} off={t.data_offset}")

    print(f"\n=== TENSOR PREFIX GROUPS ===")
    for prefix, names in sorted(groups.items()):
        print(f"  {prefix:18s}: {len(names)} tensors  (e.g. {names[:3]})")

    # Also list nextn.* (MTP) explicitly
    nextn = [t.name for t in reader.tensors if t.name.startswith("nextn")]
    print(f"\n=== MTP / nextn.* tensors ({len(nextn)}) ===")
    for n in nextn:
        print(f"  {n}")

    # router + shared expert + per-head norm explicit
    router = [t.name for t in reader.tensors if "ffn_gate_inp" in t.name or "ffn_exp_probs_b" in t.name]
    shared = [t.name for t in reader.tensors if "shard" in t.name.lower() and "ffn" in t.name]
    qknorm = [t.name for t in reader.tensors if "q_norm" in t.name or "k_norm" in t.name]
    print(f"\n=== ROUTER tensors ({len(router)}) (first 8) ===")
    for n in router[:8]: print(f"  {n}")
    print(f"\n=== SHARED-EXPERT (shard) tensors ({len(shared)}) (first 8) ===")
    for n in shared[:8]: print(f"  {n}")
    print(f"\n=== PER-HEAD Q/K NORM tensors ({len(qknorm)}) (first 8) ===")
    for n in qknorm[:8]: print(f"  {n}")

    # --- disk budget projection ---
    print(f"\n=== DISK BUDGET PROJECTION (from GGUF tensor sizes) ===")
    print(f"  expert (gate/up/down exps+shard) bytes in GGUF : {total_expert_bytes/1e9:.2f} GB")
    print(f"  attention bytes in GGUF                       : {total_attn_bytes/1e9:.3f} GB")
    print(f"  other (router/norms/embd/output/mtp) in GGUF  : {total_other_bytes/1e9:.3f} GB")
    # MXFP4 = 4 bits + 8-bit scale per 16 elems => 9 bytes / 16 elems = 4.5 bits/elem
    # Need element counts, not GGUF bytes (GGUF is Q3_K ~3.4 bits). Derive elems from expert bytes*8/3.4 approx.
    approx_expert_elems = total_expert_bytes * 8 / 3.4
    mxfp4_bytes = approx_expert_elems * 4.5 / 8
    print(f"  approx expert elements (Q3_K~3.4bpe)           : {approx_expert_elems/1e9:.2f} Gelems")
    print(f"  projected MXFP4 expert storage in .fst        : {mxfp4_bytes/1e9:.2f} GB")
    print(f"  peak disk (GGUF {total/1e9:.0f} + .fst ~{mxfp4_bytes/1e9:.0f})  : {(total+mxfp4_bytes)/1e9:.0f} GB")
    print("\nDONE.")


if __name__ == "__main__":
    main()