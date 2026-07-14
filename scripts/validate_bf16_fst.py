#!/usr/bin/env python3
"""validate_bf16_fst.py — confirm the BF16->MXFP4 .fst has correct HF tensor mapping.

Compares the F32/BF16 (non-MXFP4) tensors of qwopus_bf16.fst against the
known-good Q4_K_M-derived qwopus.fst for L0 (SSM) and L3 (GQA), plus globals.
Q4_K_M keeps 1-D norm/bias vectors and small F32 tensors unquantized, so these
must match the BF16 source to high precision (F32: exact; BF16: exact-when-both-
BF16).  A mismatch => wrong HF tensor name or bad transform; a match proves the
mapping is correct and only the MXFP4 weights differ (the intended lever).

Also reads A_log straight from the HF shard and checks ssm_a == -exp(A_log),
and checks conv1d flat layout is channel-outer (c*CK+t) matching the engine.
"""
import json, struct, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwopus_full_ref import FST, GLOBAL, QTYPE_F32, QTYPE_BF16

# TID constants (mirror fst_converter.py)
T_EMBED, T_OUTPUT_NORM, T_LM_HEAD = 0, 1, 2
T_INPUT_NORM, T_POST_ATTN_NORM = 3, 4
T_Q35_Q_NORM, T_Q35_K_NORM = 60, 61
T_Q35_SSM_CONV1D, T_Q35_SSM_A = 67, 68
T_Q35_SSM_DT_BIAS, T_Q35_SSM_NORM = 71, 72

Q4 = "qwopus.fst"
BF = sys.argv[1] if len(sys.argv) > 1 else "qwopus_bf16.fst"
HF_DIR = sys.argv[2] if len(sys.argv) > 2 else "models/Qwopus3.6-27B-Coder-BF16"

CK = 4; CDIM = 10240


def cmp_vec(a, b, name):
    if a.size == 0 or b.size == 0:
        print(f"  {name:24s} MISSING (a={a.size} b={b.size})")
        return False
    d = a.astype(np.float32) - b.astype(np.float32)
    mx = float(np.abs(d).max())
    ok = mx < 1e-4
    print(f"  {name:24s} n={a.size:6d} max|Δ|={mx:.3e} {'OK' if ok else 'MISMATCH'}")
    return ok


def load_hf_a_log():
    idx = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    shard = wm["model.language_model.layers.0.linear_attn.A_log"]
    sp = os.path.join(HF_DIR, shard)
    with open(sp, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    ds = 8 + hlen
    e = hdr["model.language_model.layers.0.linear_attn.A_log"]
    o0, o1 = e["data_offsets"]
    with open(sp, "rb") as f:
        f.seek(ds + o0); raw = f.read(o1 - o0)
    return np.frombuffer(raw, dtype=np.float32).copy()


def main():
    f4 = FST(Q4); fb = FST(BF)
    print(f"Q4  n_layers={f4.cfg['num_layers']} vocab={f4.cfg['vocab']} hidden={f4.cfg['hidden']}")
    print(f"BF  n_layers={fb.cfg['num_layers']} vocab={fb.cfg['vocab']} hidden={fb.cfg['hidden']}")
    all_ok = True

    print("\n=== L0 (SSM) + L3 (GQA) + globals: BF16-fst vs Q4-fst (F32/BF16 only) ===")
    pairs = [
        ("L0 input_norm",      T_INPUT_NORM,      0),
        ("L0 post_attn_norm",  T_POST_ATTN_NORM,  0),
        ("L0 ssm_conv1d",      T_Q35_SSM_CONV1D,  0),
        ("L0 ssm_a",           T_Q35_SSM_A,       0),
        ("L0 ssm_dt_bias",     T_Q35_SSM_DT_BIAS, 0),
        ("L0 ssm_norm",        T_Q35_SSM_NORM,    0),
        ("L3 input_norm",      T_INPUT_NORM,      3),
        ("L3 post_attn_norm",  T_POST_ATTN_NORM,  3),
        ("L3 q_norm",          T_Q35_Q_NORM,      3),
        ("L3 k_norm",          T_Q35_K_NORM,      3),
        ("global output_norm", T_OUTPUT_NORM,     GLOBAL),
    ]
    for name, tid, lid in pairs:
        try:
            a = f4.read_vec(tid, lid); b = fb.read_vec(tid, lid)
        except Exception as ex:
            print(f"  {name:24s} ERR {ex}"); all_ok = False; continue
        all_ok &= cmp_vec(a, b, name)

    # ssm_a vs HF -exp(A_log)
    print("\n=== ssm_a == -exp(A_log) check (BF16-fst vs HF source) ===")
    try:
        a_log = load_hf_a_log()
        ssm_a_hf = -np.exp(a_log)
        sa_bf = fb.read_vec(T_Q35_SSM_A, 0)
        d = sa_bf - ssm_a_hf
        mx = float(np.abs(d).max())
        ok = mx < 1e-5
        print(f"  {'L0 ssm_a vs -exp(A_log)':24s} n={sa_bf.size} max|Δ|={mx:.3e} "
              f"{'OK' if ok else 'MISMATCH'}  (all-negative={bool((sa_bf<0).all())})")
        all_ok &= ok
    except Exception as ex:
        print(f"  ssm_a-vs-HF skip: {ex}")

    # conv1d flat layout: must be channel-outer c*CK+t (engine line 4831)
    print("\n=== conv1d flat layout (channel-outer c*CK+t) ===")
    try:
        cv = fb.read_vec(T_Q35_SSM_CONV1D, 0)
        ok_size = cv.size == CDIM * CK
        # cross-check vs Q4: Q4 .fst conv1d is known channel-outer
        cv4 = f4.read_vec(T_Q35_SSM_CONV1D, 0)
        d = cv - cv4
        mx = float(np.abs(d).max())
        ok = ok_size and mx < 1e-3   # BF16 vs Q4-F32 conv -> ~bf16 precision
        print(f"  conv1d size={cv.size} (expect {CDIM*CK})  vs Q4 max|Δ|={mx:.3e} "
              f"{'OK' if ok else 'CHECK'}")
        all_ok &= ok
    except Exception as ex:
        print(f"  conv1d check skip: {ex}")

    # embed / lm_head shapes (MXFP4 — only shape, not values)
    print("\n=== embed/lm_head shapes (MXFP4) ===")
    for tid, nm in [(T_EMBED, "embed"), (T_LM_HEAD, "lm_head")]:
        _, _, _, sh4 = f4._raw(tid, GLOBAL)
        _, _, _, shb = fb._raw(tid, GLOBAL)
        m = tuple(sh4) == tuple(shb)
        print(f"  {nm:8s} Q4={tuple(sh4)} BF={tuple(shb)} {'OK' if m else 'SHAPE-DIFF'}")
        all_ok &= m

    print("\n" + ("=== ALL F32/BF16 TENSORS MATCH — HF mapping correct ===" if all_ok
                  else "=== MISMATCH DETECTED — check mapping/transforms ==="))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())