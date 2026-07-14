#!/usr/bin/env python3
"""Compare engine pos-17 (templated prefill-last) vs reference pos-17."""
import numpy as np, os, sys

D = "logs/ref_dump_tmpl"
def load(p):
    return np.fromfile(p, dtype=np.float32)

eng_hp = load("logs/head_tmpl/head_p0_h_prenorm.f32")
eng_hn = load("logs/head_tmpl/head_p0_hn.f32")
eng_lg = load("logs/head_tmpl/head_p0_logits.f32")
ref_hp = load(os.path.join(D, "ref_h_prenorm.f32"))
ref_hn = load(os.path.join(D, "ref_hn.f32"))
ref_lg = load(os.path.join(D, "ref_logits.f32"))

def cmp(a, b, name):
    d = a - b
    print(f"{name:14s} eng_norm={float(np.linalg.norm(a)):.4f} ref_norm={float(np.linalg.norm(b)):.4f} "
          f"max|Δ|={float(np.abs(d).max()):.4f} rel={float(np.abs(d).max()/max(1e-9,np.abs(b).max())):.4f} "
          f"cos={float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-20)):.5f}")

print("=== pos-17 (templated prefill-last) ref vs engine ===")
cmp(eng_hp, ref_hp, "h_prenorm")
cmp(eng_hn, ref_hn, "hn")
cmp(eng_lg, ref_lg, "logits")

et = np.argsort(eng_lg)[::-1][:8]
rt = np.argsort(ref_lg)[::-1][:8]
print("\neng top-8:", [(int(t), round(float(eng_lg[t]),3)) for t in et])
print("ref top-8:", [(int(t), round(float(ref_lg[t]),3)) for t in rt])
print(f"\neng argmax={int(et[0])}  ref argmax={int(rt[0])}  "
      f"{'MATCH' if int(et[0])==int(rt[0]) else 'DIVERGE'}")
print("\nlogit deltas at eng top-8:")
for t in et:
    print(f"  id={int(t):6d} eng={float(eng_lg[t]):.4f} ref={float(ref_lg[t]):.4f} Δ={float(eng_lg[t]-ref_lg[t]):.4f}")
print("DONE")