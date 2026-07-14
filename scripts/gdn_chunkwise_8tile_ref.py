#!/usr/bin/env python3
"""gdn_chunkwise_8tile_ref.py — reference + pack/unpack for the 8-TILE
column-split chunkwise M=K GDN kernel probe (Stage 1.2-v2).

The numpy recurrence math is IDENTICAL to gdn_chunkwise_probe_ref.py (bf16 S,
fp32 compute, bf16 round at every passB) — and that is the point: column-split
makes the recurrence INDEPENDENT per tile, so the 8-tile assembled output must
equal the full-128-col reference.  Only the DMA packing differs.

Mode "gen": deterministic toy input (same RandomState as the 1-tile ref),
  compute the reference (y, sf), and PACK the unified f_in BO:
    per chain (A=cols 0..63, B=cols 64..127), per v-head:
      128 S0 packets  (328 fp32: chain cols at [0:64], rest pad)
      8   par packets (328 fp32: [kn(128)|qn(128)|v(64 chain slice)|gdec@320|beta@321|pad])
  writes kernels/inp_8t.bin  ([chain A | chain B], each 48*136*328 fp32)
          kernels/ref_y.bin, kernels/ref_sf.bin  (full 128-col ref).

Mode "cmp": read kernels/out_8t.bin (raw 328-packet drain), UNPACK:
    per chain per v-head: 8 y-packets (chain cols at [0:64]) then 128 sf-packets.
  assemble out_y [NVH,K,HV] (chain A -> cols 0..63, chain B -> cols 64..127),
  out_sf [NVH,HV,HV]; compare to ref_y/ref_sf.  PASS gate: max|Δy|<5e-3 AND
  argmax>=99%.

NOTE on packet field order: the C kernel reads kn@0, qn@128 (kn FIRST); the
1-tile ref stored qn first then kn.  Here we repack kn-first to match the
8-tile kernel.
"""
import os, sys, numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

KDIR = Path(__file__).resolve().parent.parent / "kernels"
HV, K, NVH, PKT = 128, 8, 48, 328
PER_VH = HV + K          # 136 packets/v-head/stream
CHAIN  = 64              # cols per chain
GDEC_O, BETA_O, V_O = 320, 321, 256   # offsets in a param packet
N_CHAIN = NVH * PER_VH   # 6528 packets/chain
N_FLT   = 2 * N_CHAIN * PKT


def gen_input():
    rng = np.random.RandomState(20260713)   # same seed as the 1-tile ref
    s0 = (rng.rand(NVH, HV, HV).astype(np.float32) - 0.5) * 0.1
    # Reconstruct the SAME par the 1-tile ref produced (qn@0, kn@128, v@256,
    # gdec@384, beta@385) so the reference matches byte-for-byte.
    par = np.zeros((NVH, K, 392), np.float32)
    for v in range(NVH):
        for t in range(K):
            qn = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1
            kn = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1
            vv = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1
            par[v, t, 0:HV]       = qn
            par[v, t, HV:2*HV]    = kn
            par[v, t, 2*HV:3*HV]  = vv
            par[v, t, 384]        = 0.9 + 0.001 * (t % 5)
            par[v, t, 385]        = 0.5
    return s0, par


def ref_chunkwise(s0, par):
    """bf16 S, fp32 compute, K=8 sequential. Returns (y, sf) fp32, full 128 cols."""
    y = np.zeros((NVH, K, HV), np.float32)
    sf = np.zeros((NVH, HV, HV), np.float32)
    for v in range(NVH):
        S = s0[v].astype(bfloat16)
        for t in range(K):
            qn = par[v, t, 0:HV].astype(np.float32)
            kn = par[v, t, HV:2*HV].astype(np.float32)
            vv = par[v, t, 2*HV:3*HV].astype(np.float32)
            gdec = float(par[v, t, 384]); beta = float(par[v, t, 385])
            Sf = S.astype(np.float32)
            a = Sf.T @ kn; b = Sf.T @ qn
            c = float(kn @ qn)
            kvm = gdec * a; delta = (vv - kvm) * beta
            y[v, t] = gdec * b + delta * c
            S = (gdec * Sf + np.outer(kn, delta)).astype(bfloat16)
        sf[v] = S.astype(np.float32)
    return y, sf


def pack_f_in(s0, par):
    """Build the unified f_in BO: [chain A | chain B], each 48*136*328 fp32."""
    bo = np.zeros(N_FLT, np.float32)
    off = {0: 0, 1: N_CHAIN * PKT}      # chain A at 0, chain B at N_CHAIN*PKT
    for ch in (0, 1):
        o = off[ch]
        for v in range(NVH):
            base = o + v * PER_VH * PKT
            # 128 S0 packets: chain cols at [0:64].
            for i in range(HV):
                p = base + i * PKT
                bo[p:p + CHAIN] = s0[v, i, ch*CHAIN:(ch+1)*CHAIN]
            # 8 param packets: [kn|qn|v(64)|gdec@320|beta@321|pad]
            for t in range(K):
                p = base + (HV + t) * PKT
                bo[p:p + HV]            = par[v, t, HV:2*HV]               # kn
                bo[p + HV:p + 2*HV]     = par[v, t, 0:HV]                  # qn
                bo[p + V_O:p + V_O + CHAIN] = par[v, t, 2*HV + ch*CHAIN:2*HV + (ch+1)*CHAIN]  # v slice
                bo[p + GDEC_O] = par[v, t, 384]
                bo[p + BETA_O] = par[v, t, 385]
    return bo


def unpack_f_out(bo):
    """Unpack the raw f_out BO -> (out_y, out_sf) full 128 cols."""
    out_y = np.zeros((NVH, K, HV), np.float32)
    out_sf = np.zeros((NVH, HV, HV), np.float32)
    off = {0: 0, 1: N_CHAIN * PKT}
    for ch in (0, 1):
        o = off[ch]
        for v in range(NVH):
            base = o + v * PER_VH * PKT
            for t in range(K):                                   # 8 y packets first
                p = base + t * PKT
                out_y[v, t, ch*CHAIN:(ch+1)*CHAIN] = bo[p:p + CHAIN]
            for i in range(HV):                                  # then 128 sf packets
                p = base + (K + i) * PKT
                out_sf[v, i, ch*CHAIN:(ch+1)*CHAIN] = bo[p:p + CHAIN]
    return out_y, out_sf


def mode_gen():
    s0, par = gen_input()
    y, sf = ref_chunkwise(s0, par)
    bo = pack_f_in(s0, par)
    bo.tofile(KDIR / "inp_8t.bin")
    y.tofile(KDIR / "ref_y.bin")
    sf.tofile(KDIR / "ref_sf.bin")
    print(f"gen: inp_8t {bo.shape} {bo.dtype} ({bo.nbytes/1e6:.1f}MB)")
    print(f"     ref_y {y.shape}, ref_sf {sf.shape}")
    print(f"     ref y range [{y.min():.6f}, {y.max():.6f}], S_final range [{sf.min():.6f}, {sf.max():.6f}]")


def mode_cmp():
    bo = np.fromfile(KDIR / "out_8t.bin", np.float32)
    if bo.size != N_FLT:
        print(f"out_8t size {bo.size} != expected {N_FLT}"); return 1
    out_y, out_sf = unpack_f_out(bo)
    ref_y  = np.fromfile(KDIR / "ref_y.bin",  np.float32).reshape(NVH, K, HV)
    ref_sf = np.fromfile(KDIR / "ref_sf.bin", np.float32).reshape(NVH, HV, HV)
    dy = np.abs(out_y - ref_y); ds = np.abs(out_sf - ref_sf)
    print(f"y : max|Δ|={dy.max():.6g}  mean|Δ|={dy.mean():.6g}  (ref |y|max={np.abs(ref_y).max():.6g})")
    print(f"Sf: max|Δ|={ds.max():.6g}  mean|Δ|={ds.mean():.6g}  (ref |S|max={np.abs(ref_sf).max():.6g})")
    oa = out_y.reshape(-1, HV).argmax(1).reshape(NVH, K)
    ra = ref_y.reshape(-1, HV).argmax(1).reshape(NVH, K)
    agree = (oa == ra).mean()
    print(f"argmax agreement: {agree*100:.1f}%  ({(oa==ra).sum()}/{oa.size})")
    ok = dy.max() < 5e-3 and agree >= 0.99
    print("PASS" if ok else "FAIL", "(gate: max|Δy|<5e-3 AND argmax>=99%)")
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    sys.exit(mode_cmp() if mode == "cmp" else mode_gen())