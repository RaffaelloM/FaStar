#!/usr/bin/env python3
"""Per-layer residual bisection: compare engine eng_h_L{n}.f32 (FST_TRUNK_DUMP)
against reference ref_h_L{n}.f32 (qwopus_full_ref.py dump) at the dumped layers.

Gradual growth of max|Δ| with layer index = bf16/NPU accumulation noise.
A sudden jump at one layer = a composition bug at that layer.
"""
import os, sys, numpy as np

eng = sys.argv[1] if len(sys.argv) > 1 else "logs/trunk_dump"
ref = sys.argv[2] if len(sys.argv) > 2 else "logs/ref_dump_9419"

layers = sorted({int(f.split("_L")[1].split(".")[0])
                 for f in os.listdir(eng) if f.startswith("eng_h_L")})
print(f"layers dumped by engine: {layers}")
print(f"{'L':>3} {'eng_norm':>10} {'ref_norm':>10} {'max|Δ|':>12} {'rel_max':>10} {'cos':>8}")
prev = None
for L in layers:
    ep = os.path.join(eng, f"eng_h_L{L}.f32")
    rp = os.path.join(ref, f"ref_h_L{L}.f32")
    if not os.path.exists(rp):
        print(f"{L:3d}  (no ref dump)")
        continue
    e = np.fromfile(ep, np.float32)
    r = np.fromfile(rp, np.float32)
    d = np.abs(e - r)
    mx = float(d.max())
    denom = np.maximum(np.maximum(np.abs(e), np.abs(r)), 1e-6)
    rel = float((d / denom).max())
    cos = float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-20))
    en = float(np.linalg.norm(e)); rn = float(np.linalg.norm(r))
    tag = ""
    if prev is not None and mx > prev * 3.0 and mx > 1.0:
        tag = "  <-- JUMP"
    print(f"{L:3d} {en:10.3f} {rn:10.3f} {mx:12.4e} {rel:10.3e} {cos:8.5f}{tag}")
    prev = mx