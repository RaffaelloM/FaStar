#!/usr/bin/env python3
"""Compare the full-trunk numpy reference (logs/ref_dump_9419) against the C++
engine head dump (logs/head_dump) for token 9419 ("Hello").

Reports max|Δ| / rel for: final pre-norm residual h_prenorm, post-norm hn, and
the full logits vector + top-5 agreement.  This is the end-to-end trunk test.
"""
import os, sys, numpy as np

ref = sys.argv[1] if len(sys.argv) > 1 else "logs/ref_dump_9419"
eng = sys.argv[2] if len(sys.argv) > 2 else "logs/head_dump"


def cmp(name, rp, ep):
    r = np.fromfile(rp, np.float32)
    e = np.fromfile(ep, np.float32)
    if r.shape != e.shape:
        print(f"  !! {name}: SHAPE MISMATCH ref={r.shape} eng={e.shape}")
        return None
    d = np.abs(r - e)
    mx = float(d.max()) if d.size else 0.0
    denom = np.maximum(np.maximum(np.abs(r), np.abs(e)), 1e-6)
    rel = float((d / denom).max()) if d.size else 0.0
    tag = "OK " if mx < 1e-2 else "XX "
    print(f"  {tag}{name:16s} n={r.size:7d} max|Δ|={mx:.4e} rel={rel:.4e}")
    return mx


def topk(logits, k=5):
    idx = np.argsort(logits)[::-1][:k]
    return list(zip(idx.tolist(), logits[idx].tolist()))


print(f"== ref {ref}  vs  eng {eng} ==")
cmp("h_prenorm", os.path.join(ref, "ref_h_prenorm.f32"),
    os.path.join(eng, "head_p0_h_prenorm.f32"))
cmp("hn", os.path.join(ref, "ref_hn.f32"),
    os.path.join(eng, "head_p0_hn.f32"))
cmp("logits", os.path.join(ref, "ref_logits.f32"),
    os.path.join(eng, "head_p0_logits.f32"))

rl = np.fromfile(os.path.join(ref, "ref_logits.f32"), np.float32)
el = np.fromfile(os.path.join(eng, "head_p0_logits.f32"), np.float32)
print("\n  ref top-5:", [(int(i), round(float(v), 4)) for i, v in topk(rl)])
print("  eng top-5:", [(int(i), round(float(v), 4)) for i, v in topk(el)])
print(f"  ref logit[9419]={float(rl[9419]):.4f}   eng logit[9419]={float(el[9419]):.4f}")
rarg = int(np.argmax(rl)); earg = int(np.argmax(el))
print(f"  ref argmax={rarg}  eng argmax={earg}  (input token=9419)")