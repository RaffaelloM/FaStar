#!/usr/bin/env python3
"""diag_bf16.py — inspect HF shard dtypes/shapes vs both .fsts for L0."""
import json, struct, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwopus_full_ref import FST, GLOBAL, QTYPE_F32, QTYPE_BF16, QTYPE_MXFP4

HF = "models/Qwopus3.6-27B-Coder-BF16"
T_Q35_SSM_CONV1D, T_Q35_SSM_A = 67, 68
T_Q35_SSM_DT_BIAS, T_Q35_SSM_NORM = 71, 72
T_INPUT_NORM, T_POST_ATTN_NORM = 3, 4
T_OUTPUT_NORM = 1

def shard_header(name):
    idx = json.load(open(os.path.join(HF, "model.safetensors.index.json")))
    shard = idx["weight_map"][name]
    sp = os.path.join(HF, shard)
    with open(sp, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return sp, 8 + hlen, hdr, shard

def load_hf(name):
    sp, ds, hdr, _ = shard_header(name)
    e = hdr[name]
    o0, o1 = e["data_offsets"]
    with open(sp, "rb") as f:
        f.seek(ds + o0); raw = f.read(o1 - o0)
    print(f"  HF {name}: dtype={e['dtype']} shape={tuple(e['shape'])} bytes={len(raw)}")
    if e["dtype"] == "BF16":
        u = np.frombuffer(raw, np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"]).copy()
    if e["dtype"] == "F32":
        return np.frombuffer(raw, np.float32).reshape(e["shape"]).copy()
    return np.frombuffer(raw, np.uint8)

f4 = FST("qwopus.fst"); fb = FST("qwopus_bf16.fst")

def fst_vec(fst, tid, lid):
    raw, qt, nd, sh = fst._raw(tid, lid)
    if qt == QTYPE_F32:
        return np.frombuffer(raw, np.float32).copy(), qt, sh
    if qt == QTYPE_BF16:
        u = np.frombuffer(raw, np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).copy(), qt, sh
    return np.zeros(0), qt, sh

print("=== input_norm L0 ===")
a4, q4, sh4 = fst_vec(f4, T_INPUT_NORM, 0)
ab, qb, shb = fst_vec(fb, T_INPUT_NORM, 0)
print(f"  Q4 .fst: qt={q4} shape={sh4} n={a4.size} first5={a4[:5]}")
print(f"  BF .fst: qt={qb} shape={shb} n={ab.size} first5={ab[:5]}")
hn = load_hf("model.language_model.layers.0.input_layernorm.weight")
print(f"  HF input_layernorm flat first5={hn.reshape(-1)[:5]}")

print("\n=== ssm_a L0 ===")
a4, q4, _ = fst_vec(f4, T_Q35_SSM_A, 0)
ab, qb, _ = fst_vec(fb, T_Q35_SSM_A, 0)
print(f"  Q4 .fst: qt={q4} n={a4.size} first5={a4[:5]}")
print(f"  BF .fst: qt={qb} n={ab.size} first5={ab[:5]}")
alog = load_hf("model.language_model.layers.0.linear_attn.A_log")
print(f"  HF A_log -> -exp first5={(-np.exp(alog.reshape(-1).astype(np.float32)))[:5]}")

print("\n=== conv1d L0 ===")
a4, q4, sh4 = fst_vec(f4, T_Q35_SSM_CONV1D, 0)
ab, qb, shb = fst_vec(fb, T_Q35_SSM_CONV1D, 0)
print(f"  Q4 .fst: qt={q4} shape={sh4} n={a4.size} first8={a4[:8]}")
print(f"  BF .fst: qt={qb} shape={shb} n={ab.size} first8={ab[:8]}")
cv = load_hf("model.language_model.layers.0.linear_attn.conv1d.weight")
print(f"  HF conv1d flat first8={cv.reshape(-1)[:8]}")
print(f"  HF conv1d flat[0:4] (chan0 taps)={cv.reshape(-1)[:4]}")
print(f"  HF conv1d as [CDIM,CK] chan0={cv.reshape(cv.shape[0],-1)[0]}")

print("\n=== dt_bias L0 ===")
a4, _, _ = fst_vec(f4, T_Q35_SSM_DT_BIAS, 0)
ab, _, _ = fst_vec(fb, T_Q35_SSM_DT_BIAS, 0)
print(f"  Q4 n={a4.size} first5={a4[:5]}")
print(f"  BF n={ab.size} first5={ab[:5]}")
dt = load_hf("model.language_model.layers.0.linear_attn.dt_bias")
print(f"  HF dt_bias first5={dt.reshape(-1)[:5]}")

print("\n=== output_norm ===")
a4, q4, _ = fst_vec(f4, T_OUTPUT_NORM, GLOBAL)
ab, qb, _ = fst_vec(fb, T_OUTPUT_NORM, GLOBAL)
print(f"  Q4 qt={q4} n={a4.size} first5={a4[:5]}")
print(f"  BF qt={qb} n={ab.size} first5={ab[:5]}")
on = load_hf("model.language_model.norm.weight")
print(f"  HF norm first5={on.reshape(-1)[:5]}")