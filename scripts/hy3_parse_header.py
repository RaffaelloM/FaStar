#!/usr/bin/env python3
"""Parse the Hunyuan-3.0 GGUF *header* manually (no full download, no GGUFReader).

GGUFReader eagerly materializes every tensor's data view, so it fails on a partial
file. The header (magic / version / counts / KV metadata / tensor-info table) lives
entirely before the tensor data region and is small (~MB), so we parse it directly
per the GGUF spec. We never touch tensor data -> a few-MB Range fetch is enough.

Outputs every metadata KV field (architecture), every tensor name/shape/type/offset,
prefix grouping, MTP/router/shared/QK-norm tensor lists, and a disk-budget projection.
"""
import os
import sys
import struct
import requests
import gguf  # only for enum names

URL = "https://huggingface.co/satgeze/Hy3-1M-GGUF/resolve/main/hy3-1M-MTP-Q3_K_M.gguf"
PARTIAL = "/tmp/hy3_head.bin"
WINDOW = 8 * 1024 * 1024  # 8 MB header window (header is << 1 MB)

# ggml type id -> (name, block_elems, block_bytes) for n_bytes computation
GGML = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 30: ("BF16", 1, 2),
}
# gguf value type id -> (name, read function given a cursor reader)
def _str(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off); off += 8  # GGUF string len is uint64
    s = buf[off:off+n].decode("utf-8", "replace"); off += n
    return s, off
def _scalar(buf, off, fmt, size):
    (v,) = struct.unpack_from(fmt, buf, off); off += size
    return v, off
VT = {
    0: ("UINT8", "<B", 1), 1: ("INT8", "<b", 1), 2: ("UINT16", "<H", 2),
    3: ("INT16", "<h", 2), 4: ("UINT32", "<I", 4), 5: ("INT32", "<i", 4),
    6: ("FLOAT32", "<f", 4), 7: ("BOOL", "<?", 1), 8: ("STRING", None, None),
    9: ("ARRAY", None, None), 10: ("UINT64", "<Q", 8), 11: ("INT64", "<q", 8),
    12: ("FLOAT64", "<d", 8),
}


def ensure_window():
    if os.path.exists(PARTIAL) and os.path.getsize(PARTIAL) >= WINDOW:
        return
    print(f"Range-fetching {WINDOW/1e6:.0f} MB header window...")
    h = requests.head(URL, allow_redirects=True, timeout=60)
    print(f"  HEAD {h.status_code} len={h.headers.get('Content-Length')} ranges={h.headers.get('Accept-Ranges')}")
    r = requests.get(URL, headers={"Range": f"bytes=0-{WINDOW-1}"}, stream=True, timeout=120, allow_redirects=True)
    r.raise_for_status()
    data = bytearray()
    for b in r.iter_content(4 * 1024 * 1024):
        data.extend(b)
        if len(data) >= WINDOW:
            break
    with open(PARTIAL, "wb") as f:
        f.write(bytes(data[:WINDOW]))
    print(f"  saved {WINDOW} bytes -> {PARTIAL}")


def read_value(buf, off, vt):
    name, fmt, size = VT[vt]
    if name == "STRING":
        return _str(buf, off)
    if name == "ARRAY":
        (et,) = struct.unpack_from("<I", buf, off); off += 4
        (cnt,) = struct.unpack_from("<Q", buf, off); off += 8
        etname = VT.get(et, (f"?{et}",))[0]
        vals = []
        for _ in range(cnt):
            v, off = read_scalar(buf, off, et)
            vals.append(v)
        return (f"ARRAY<{etname}>[{cnt}]", vals), off
    # scalar: unpack and advance off
    (v,) = struct.unpack_from(fmt, buf, off); off += size
    return v, off


def read_scalar(buf, off, vt):
    name, fmt, size = VT[vt]
    if name == "STRING":
        return _str(buf, off)
    return _scalar(buf, off, fmt, size)


def tensor_nbytes(shape, ttype):
    nelems = 1
    for d in shape:
        nelems *= d
    if ttype not in GGML:
        return nelems, None
    _, be, bb = GGML[ttype]
    return nelems, (nelems // be) * bb


def main():
    ensure_window()
    with open(PARTIAL, "rb") as f:
        buf = f.read()
    off = 0
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", buf, off); off += 24
    print(f"=== GGUF HEADER ===")
    print(f"  magic={bytes(struct.pack('<I',magic))!r} version={version} n_tensors={n_tensors} n_kv={n_kv}")

    print(f"\n=== METADATA KV ({n_kv}) ===")
    fields = {}
    for _ in range(n_kv):
        key, off = _str(buf, off)
        (vt,) = struct.unpack_from("<I", buf, off); off += 4
        v, off = read_value(buf, off, vt)
        vtname = VT.get(vt, (f"?{vt}",))[0]
        if isinstance(v, tuple) and isinstance(v[0], str) and v[0].startswith("ARRAY"):
            label, vals = v
            # summarize arrays that are long (e.g. per-layer)
            if len(vals) > 24:
                print(f"  {key} = {label} first8={vals[:8]} ... last4={vals[-4:]}")
            else:
                print(f"  {key} = {label} {vals}")
        elif vt == 8:  # STRING
            print(f"  {key} = {v!r}   [STRING]")
        else:
            print(f"  {key} = {v}   [{vtname}]")
        fields[key] = v

    print(f"\n=== TENSOR INFOS ({n_tensors}) ===")
    tensors = []
    for _ in range(n_tensors):
        name, off = _str(buf, off)
        (ndims,) = struct.unpack_from("<I", buf, off); off += 4
        dims = list(struct.unpack_from(f"<{ndims}Q", buf, off)); off += 8 * ndims
        (ttype,) = struct.unpack_from("<I", buf, off); off += 4
        (doff,) = struct.unpack_from("<Q", buf, off); off += 8
        tname = GGML.get(ttype, (f"?{ttype}",))[0]
        nelems, nb = tensor_nbytes(dims, ttype)
        tensors.append((name, dims, ttype, tname, nelems, nb, doff))

    for i, (name, dims, ttype, tname, nelems, nb, doff) in enumerate(tensors):
        if i < 100:
            print(f"  [{i:4d}] {name:46s} dims={dims} type={tname} elems={nelems} bytes={nb} off={doff}")

    # grouping
    from collections import defaultdict
    groups = defaultdict(list)
    for (name, *_rest) in tensors:
        groups[name.split(".")[0] if "." in name else name].append(name)
    print(f"\n=== TENSOR PREFIX GROUPS ===")
    for p, ns in sorted(groups.items()):
        print(f"  {p:18s}: {len(ns)}  e.g. {ns[:3]}")

    def names(pred):
        return [t[0] for t in tensors if pred(t[0])]

    nextn = names(lambda n: n.startswith("nextn"))
    router = names(lambda n: "ffn_gate_inp" in n or "ffn_exp_probs_b" in n)
    shared = names(lambda n: "ffn" in n and "shard" in n.lower())
    qknorm = names(lambda n: "q_norm" in n or "k_norm" in n)
    experts = names(lambda n: "ffn_gate_exps" in n or "ffn_up_exps" in n or "ffn_down_exps" in n)
    for label, lst in [("MTP/nextn", nextn), ("ROUTER", router), ("SHARED-EXPERT(shard)", shared),
                       ("PER-HEAD Q/K NORM", qknorm), ("EXPERT(exps)", experts)]:
        print(f"\n=== {label} ({len(lst)}) ===")
        for n in lst[:12]:
            print(f"  {n}")
        if len(lst) > 12:
            print(f"  ... +{len(lst)-12} more")

    # disk budget
    print(f"\n=== DISK BUDGET PROJECTION ===")
    def sum_bytes(pred):
        tot_elems = tot_bytes = 0
        unknown = 0
        for (name, dims, ttype, tname, nelems, nb, doff) in tensors:
            if pred(name):
                tot_elems += nelems
                if nb is None:
                    unknown += 1
                else:
                    tot_bytes += nb
        return tot_elems, tot_bytes, unknown
    e_elems, e_bytes, e_unk = sum_bytes(lambda n: "ffn" in n and ("exps" in n or "shard" in n.lower()))
    a_elems, a_bytes, a_unk = sum_bytes(lambda n: any(k in n for k in ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")))
    o_elems, o_bytes, o_unk = sum_bytes(lambda n: not any(k in n for k in ("attn_", "ffn_")) and not n.startswith("nextn"))
    print(f"  experts  : {len(experts)+len(shared)} tensors, elems={e_elems/1e9:.2f}G, GGUF-bytes={e_bytes/1e9:.2f}G (unknown types={e_unk})")
    print(f"  attention: elems={a_elems/1e6:.1f}M, GGUF-bytes={a_bytes/1e6:.1f}M (unknown={a_unk})")
    print(f"  other    : elems={o_elems/1e6:.1f}M, GGUF-bytes={o_bytes/1e6:.1f}M (unknown={o_unk})")
    mxfp4_bytes = e_elems * 4.5 / 8  # 4 bits + 8-bit scale per 16 elems
    bf16_attn = a_elems * 2
    print(f"  projected .fst: experts MXFP4 ~{mxfp4_bytes/1e9:.2f} GB + attn/router/norms BF16 ~{(bf16_attn+o_elems*2)/1e9:.3f} GB")
    total_gguf = 144.95e9
    print(f"  peak disk (GGUF {total_gguf/1e9:.0f} + .fst ~{(mxfp4_bytes+bf16_attn+o_elems*2)/1e9:.0f}) = {(total_gguf+mxfp4_bytes+bf16_attn+o_elems*2)/1e9:.0f} GB (free was 343 GB)")

    # ---- HY3-specific structural analysis ----
    import re as _re
    from collections import defaultdict as _dd
    print(f"\n=== DISTINCT TENSOR PATTERNS (blk.N normalized) ===")
    pats = _dd(int)
    pat_sample = {}
    for (name, dims, ttype, tname, nelems, nb, doff) in tensors:
        pat = _re.sub(r"blk\.\d+\.", "blk.N.", name)
        pats[pat] += 1
        pat_sample.setdefault(pat, (dims, tname))
    for pat in sorted(pats):
        ds, tn = pat_sample[pat]
        print(f"  {pats[pat]:4d}  {pat:38s} dims={ds} type={tn}")

    print(f"\n=== LAST BLOCK blk.80.* (candidate MTP/NextN) ===")
    for (name, dims, ttype, tname, nelems, nb, doff) in tensors:
        if name.startswith("blk.80."):
            print(f"  {name:42s} dims={dims} type={tname} bytes={nb}")

    print(f"\n=== CASE-INSENSITIVE MTP search (nextn/mtp/predict/eh_proj/enorm/hnorm/shared_head/head_norm/tblk) ===")
    hits = [t for t in tensors if _re.search(r"nextn|mtp|predict|eh_proj|enorm|hnorm|shared_head|head_norm|tblk", t[0], _re.I)]
    print(f"  {len(hits)} hits")
    for (name, dims, ttype, tname, nelems, nb, doff) in hits[:30]:
        print(f"  {name:42s} dims={dims} type={tname}")

    shexp = [t for t in tensors if "shexp" in t[0]]
    print(f"\n=== shexp (shared expert) tensors: {len(shexp)} (first 4) ===")
    for (name, dims, ttype, tname, nelems, nb, doff) in shexp[:4]:
        print(f"  {name:42s} dims={dims} type={tname} bytes={nb}")
    epb = [t for t in tensors if "exp_probs_b" in t[0]]
    print(f"=== exp_probs_b (router bias) tensors: {len(epb)} (first 2) ===")
    for (name, dims, ttype, tname, nelems, nb, doff) in epb[:2]:
        print(f"  {name:42s} dims={dims} type={tname}")

    # dense vs MoE: a layer is MoE if it has ffn_gate_inp or exp_probs_b
    has_router = {}
    for (name, *_x) in tensors:
        m = _re.match(r"blk\.(\d+)\.", name)
        if m:
            li = int(m.group(1))
            has_router.setdefault(li, False)
            if "ffn_gate_inp" in name or "exp_probs_b" in name:
                has_router[li] = True
    dense = sorted([li for li, v in has_router.items() if not v])
    moe = sorted([li for li, v in has_router.items() if v])
    print(f"\n=== DENSE layers (no router): {dense}  -> count {len(dense)} ===")
    print(f"=== MoE layers (have router): count {len(moe)} (range {min(moe)}..{max(moe)}) ===")
    # does dense layer 0 have ffn_gate/up/down (dense FFN, inter=13312)?
    l0 = [t for t in tensors if t[0].startswith("blk.0.ffn_")]
    print(f"=== blk.0.ffn_* (dense L0 FFN) ===")
    for (name, dims, ttype, tname, nelems, nb, doff) in l0:
        print(f"  {name:42s} dims={dims} type={tname} bytes={nb}")

    print("\nDONE.")


if __name__ == "__main__":
    main()