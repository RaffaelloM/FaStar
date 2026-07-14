#!/usr/bin/env python3
"""gdn_chunkwise_probe_ref.py — reference + compare for the chunkwise M=K GDN
kernel probe (Stage 1.3 standalone, toy data).

Mode "gen" (default): generate deterministic toy input (S0 + K=8 params for 48
v-heads), write kernels/inp_s0.bin + kernels/inp_par.bin, and compute the
EXPECTED outputs (y per token, S_final) with the SAME math as the C kernel
(bf16 S storage, fp32 compute, bf16 round at every passB store), writing
kernels/ref_y.bin + kernels/ref_sf.bin.

Mode "cmp": read the probe's kernels/out_y.bin + kernels/out_sf.bin and compare
to ref_y.bin + ref_sf.bin.  Report max|Δy|, max|ΔS|, and an argmax proxy.

The bf16 round is applied to S after every token's passB (matching the kernel's
acc.to_vector<bfloat16>() at the passB store).  fp32 compute on bf16-rounded S
is the A/B-validated lossless mitigation (FST_Q35_GDN_BF16S, 39/39 argmax).

Layout (must match gen_gdn_chunkwise.py / the C kernel):
  inp_s0 : [NVH, HV, HV]  fp32   (S0 per v-head, row-major)
  inp_par: [NVH, K, PKT_PAR] fp32  ([qn(128)|kn(128)|v(128)|gdec(1)|beta(1)|pad(6)])
  out_y  : [NVH, K, HV]   fp32   (y per token per v-head)
  out_sf : [NVH, HV, HV]  fp32   (S_final per v-head)
"""
import os, sys, numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

KDIR = Path(__file__).resolve().parent.parent / "kernels"
HV, K, NVH = 128, 8, 48
PKT_PAR = 392
PAR_GDEC, PAR_BETA = 384, 385

def gen_input():
    rng = np.random.RandomState(20260713)
    # S0: small values in ~[-0.05, 0.05]; bf16-representable after round.
    s0 = (rng.rand(NVH, HV, HV).astype(np.float32) - 0.5) * 0.1
    par = np.zeros((NVH, K, PKT_PAR), np.float32)
    for v in range(NVH):
        for t in range(K):
            qn = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1
            kn = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1
            vv = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1
            par[v, t, 0:HV]            = qn
            par[v, t, HV:2*HV]         = kn
            par[v, t, 2*HV:3*HV]       = vv
            par[v, t, PAR_GDEC]        = 0.9 + 0.001 * (t % 5)
            par[v, t, PAR_BETA]        = 0.5
    return s0, par

def ref_chunkwise(s0, par):
    """bf16 S, fp32 compute, K=8 sequential tokens.  Returns (y, sf) fp32."""
    y = np.zeros((NVH, K, HV), np.float32)
    sf = np.zeros((NVH, HV, HV), np.float32)
    for v in range(NVH):
        S = s0[v].astype(bfloat16)            # initial state (bf16)
        for t in range(K):
            qn = par[v, t, 0:HV].astype(np.float32)
            kn = par[v, t, HV:2*HV].astype(np.float32)
            vv = par[v, t, 2*HV:3*HV].astype(np.float32)
            gdec = float(par[v, t, PAR_GDEC])
            beta = float(par[v, t, PAR_BETA])
            Sf = S.astype(np.float32)           # bf16 -> fp32 for compute
            a = Sf.T @ kn                        # a[j] = Σ_i S[i,j] kn[i]
            b = Sf.T @ qn                        # b[j] = Σ_i S[i,j] qn[i]
            c = float(kn @ qn)
            kvm = gdec * a
            delta = (vv - kvm) * beta
            y[v, t] = gdec * b + delta * c
            # passB: S[i,j] = gdec*S[i,j] + kn[i]*delta[j]  (fp32 compute, round to bf16)
            S2 = gdec * Sf + np.outer(kn, delta) # [HV,HV]
            S = S2.astype(bfloat16)
        sf[v] = S.astype(np.float32)
    return y, sf

def mode_gen():
    s0, par = gen_input()
    y, sf = ref_chunkwise(s0, par)
    s0.tofile(KDIR / "inp_s0.bin")
    par.tofile(KDIR / "inp_par.bin")
    y.tofile(KDIR / "ref_y.bin")
    sf.tofile(KDIR / "ref_sf.bin")
    print(f"gen: inp_s0 {s0.shape} {s0.dtype} ({s0.nbytes}B), inp_par {par.shape} ({par.nbytes}B)")
    print(f"     ref_y {y.shape}, ref_sf {sf.shape}")
    print(f"     ref y range [{y.min():.6f}, {y.max():.6f}], S_final range [{sf.min():.6f}, {sf.max():.6f}]")

def mode_cmp():
    out_y = np.fromfile(KDIR / "out_y.bin", np.float32).reshape(NVH, K, HV)
    out_sf = np.fromfile(KDIR / "out_sf.bin", np.float32).reshape(NVH, HV, HV)
    ref_y = np.fromfile(KDIR / "ref_y.bin", np.float32).reshape(NVH, K, HV)
    ref_sf = np.fromfile(KDIR / "ref_sf.bin", np.float32).reshape(NVH, HV, HV)
    dy = np.abs(out_y - ref_y)
    ds = np.abs(out_sf - ref_sf)
    print(f"y : max|Δ|={dy.max():.6g}  mean|Δ|={dy.mean():.6g}  (ref |y|max={np.abs(ref_y).max():.6g})")
    print(f"Sf: max|Δ|={ds.max():.6g}  mean|Δ|={ds.mean():.6g}  (ref |S|max={np.abs(ref_sf).max():.6g})")
    # argmax proxy: per (v,t) the max-lane index of y
    oa = out_y.reshape(-1, HV).argmax(1).reshape(NVH, K)
    ra = ref_y.reshape(-1, HV).argmax(1).reshape(NVH, K)
    agree = (oa == ra).mean()
    print(f"argmax agreement: {agree*100:.1f}%  ({(oa==ra).sum()}/{oa.size})")
    # relative error on the largest-magnitude y entries
    scale = np.maximum(np.abs(ref_y), 1e-6)
    rel = (dy / scale)
    print(f"y relative error: max={rel.max():.4g}  p99={np.percentile(rel,99):.4g}")
    ok = dy.max() < 5e-3 and agree >= 0.99
    print("PASS" if ok else "FAIL", "(gate: max|Δy|<5e-3 AND argmax≥99%)")
    return 0 if ok else 1

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    sys.exit(mode_cmp() if mode == "cmp" else mode_gen())