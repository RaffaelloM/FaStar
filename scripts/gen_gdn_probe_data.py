#!/usr/bin/env python3
"""gen_gdn_probe_data.py — emit a deterministic GDN test-vector file for the hardware
probe (tools/gdn_probe.cpp).

Uses the EXACT same data + reference as scripts/compare_kernel_twin.py (rng seed 7,
standard_normal q/k/v, g_logit=-1.3, beta=0.37, S0=zeros) and the bit-correct numpy
reference gdn_delta_rule from scripts/qwopus_ssm_ref.py.  Writes gdn_probe_data.bin:

  layout (all fp32, little-endian):
    qn[128]      pre-normed q: l2norm(q)*SCALE  (kernel contract input)
    kn[128]      pre-normed k: l2norm(k)
    v[128]
    gdec(1)      = exp(g_logit)
    beta(1)      = 0.37 (already sigmoid'd)
    S0[128*128]  row-major, zeros
    ref_y[128]   = gdn_delta_rule(...) output y
    ref_S[128*128] = gdn_delta_rule(...) output S
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from qwopus_ssm_ref import gdn_delta_rule, _l2norm_np, SCALE, HEAD_K, HEAD_V

rng = np.random.default_rng(7)
q = rng.standard_normal(HEAD_K).astype(np.float32)
k = rng.standard_normal(HEAD_K).astype(np.float32)
v = rng.standard_normal(HEAD_V).astype(np.float32)
g_logit, beta = -1.3, 0.37

qn  = (_l2norm_np(q) * SCALE).astype(np.float32)
kn  = _l2norm_np(k).astype(np.float32)
gdec = np.float32(np.exp(g_logit))
# NON-ZERO S0: exposes any +element drain shift in passA's a,b (which are S0ᵀ@kn,
# S0ᵀ@qn — zero when S0=0, masking a drain shift).  Small magnitude keeps fp32 clean.
S0  = (0.01 * rng.standard_normal((HEAD_K, HEAD_V))).astype(np.float32)

# pure delta rule = the kernel's contract (upstream l2norm/scale/exp already applied)
ref_y, ref_S = gdn_delta_rule(qn, kn, v, float(gdec), float(beta), S0.copy())

out = np.concatenate([
    qn.reshape(-1), kn.reshape(-1), v.reshape(-1),
    np.array([gdec, beta], np.float32),
    S0.reshape(-1),
    ref_y.reshape(-1).astype(np.float32),
    ref_S.reshape(-1).astype(np.float32),
]).astype(np.float32)

path = sys.argv[1] if len(sys.argv) > 1 else "gdn_probe_data.bin"
out.tofile(path)
print(f"wrote {path} ({out.nbytes} B = {out.size} fp32)")
print(f"  qn|kn|v|gdec,beta|S0|ref_y|ref_S = 128+128+128+2+16384+128+16384 = {128+128+128+2+16384+128+16384}")
print(f"  ref_y[:4] = {ref_y[:4]}")
print(f"  ref_S[0,:4] = {ref_S[0,:4]}")
print(f"  gdec={float(gdec):.9g} beta={beta} (|ref_y|_max={np.max(np.abs(ref_y)):.6g}, |ref_S|_max={np.max(np.abs(ref_S)):.6g})")