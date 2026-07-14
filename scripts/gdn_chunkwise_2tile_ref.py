#!/usr/bin/env python3
"""gdn_chunkwise_2tile_ref.py — reference + pack/unpack for the 2-TILE
column-split single-call stack-S chunkwise M=K GDN kernel probe (Stage 1.2-v3).

The recurrence math matches the kernel: bf16 S0, bf16 kn/qn/v/gdec/beta inputs
(bf16 unified packets to fit the ~64KB tile data memory), fp32 compute, bf16
round at every passB store, bf16 y output.  Column-split ⇒ the 2-tile assembled
output (tile0 cols 0..63 + tile1 cols 64..127) equals the full-128-col reference.

Mode "gen": deterministic toy input (same RandomState as the 1-tile/8-tile refs),
compute the reference (y, sf) with bf16 inputs, PACK the bf16 f_in BO:
  per tile (0=cols 0..63, 1=cols 64..127), per v-head: ONE in-packet [10768 bf16]
  = [S0(128×64) | par(8×322 = kn|qn|v|gdec|beta)].
  writes kernels/inp_2t.bin  ([tile0 | tile1], each 48*10768 bf16)
          kernels/ref_y.bin, kernels/ref_sf.bin  (full 128-col ref, fp32 for compare).

Mode "cmp": read kernels/out_2t.bin (bf16 drain), UNPACK per tile per v-head:
  out-packet [8704 bf16] = [y(8×64) | sf(128×64)]; assemble out_y [NVH,K,HV],
  out_sf [NVH,HV,HV] (tile0->cols 0..63, tile1->cols 64..127); compare to ref.
  PASS gate: max|Δy|<5e-3 AND argmax>=99%.
"""
import os, sys, numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

KDIR = Path(__file__).resolve().parent.parent / "kernels"
HV, K, NVH, COLS, IN_PAR = 128, 8, 48, 64, 322
IN_PKT  = HV * COLS + K * IN_PAR     # 10768
OUT_PKT = K * COLS + HV * COLS       # 8704
PAR_V, PAR_GDEC, PAR_BETA = 256, 320, 321
HALF   = NVH * IN_PKT
HALF_O = NVH * OUT_PKT
N_IN, N_OUT = 2 * HALF, 2 * HALF_O


def gen_input():
    rng = np.random.RandomState(20260713)   # same seed
    s0 = (rng.rand(NVH, HV, HV).astype(np.float32) - 0.5) * 0.1
    par = np.zeros((NVH, K, 392), np.float32)   # [qn(128)|kn(128)|v(128)|gdec@384|beta@385]
    for v in range(NVH):
        for t in range(K):
            par[v, t, 0:HV]      = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1   # qn
            par[v, t, HV:2*HV]   = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1   # kn
            par[v, t, 2*HV:3*HV] = (rng.rand(HV).astype(np.float32) - 0.5) * 0.1   # v
            par[v, t, 384]       = 0.9 + 0.001 * (t % 5)
            par[v, t, 385]       = 0.5
    return s0, par


def ref_chunkwise(s0, par):
    """bf16 S, bf16 kn/qn/v inputs, fp32 compute, bf16 y output. Returns (y, sf) fp32."""
    y = np.zeros((NVH, K, HV), np.float32)
    sf = np.zeros((NVH, HV, HV), np.float32)
    for v in range(NVH):
        S = s0[v].astype(bfloat16)
        for t in range(K):
            qn  = par[v, t, 0:HV].astype(bfloat16).astype(np.float32)
            kn  = par[v, t, HV:2*HV].astype(bfloat16).astype(np.float32)
            vv  = par[v, t, 2*HV:3*HV].astype(bfloat16).astype(np.float32)
            gdec = float(np.array(par[v, t, 384]).astype(bfloat16))
            beta = float(np.array(par[v, t, 385]).astype(bfloat16))
            Sf = S.astype(np.float32)
            a = Sf.T @ kn; b = Sf.T @ qn
            c = float(kn @ qn)
            kvm = gdec * a; delta = (vv - kvm) * beta
            y[v, t] = (gdec * b + delta * c).astype(bfloat16).astype(np.float32)
            S = (gdec * Sf + np.outer(kn, delta)).astype(bfloat16)
        sf[v] = S.astype(np.float32)
    return y, sf


def pack_f_in(s0, par):
    """bf16 f_in BO: [tile0 | tile1], each 48*10768 bf16."""
    bo = np.zeros(N_IN, bfloat16)
    for tile in (0, 1):
        off = tile * HALF
        for v in range(NVH):
            base = off + v * IN_PKT
            # S0: 128 rows × 64 cols (this tile's col slice) -> bf16
            s0slice = s0[v, :, tile*COLS:(tile+1)*COLS].astype(bfloat16)
            bo[base:base + HV*COLS] = s0slice.reshape(-1)
            # par: 8 tokens × 322 = [kn(128)|qn(128)|v(64)|gdec|beta] bf16
            for t in range(K):
                p = base + HV*COLS + t * IN_PAR
                bo[p:p+HV]         = par[v, t, HV:2*HV].astype(bfloat16)            # kn
                bo[p+HV:p+2*HV]    = par[v, t, 0:HV].astype(bfloat16)               # qn
                bo[p+PAR_V:p+PAR_V+COLS] = par[v, t, 2*HV+tile*COLS:2*HV+(tile+1)*COLS].astype(bfloat16)  # v slice
                bo[p+PAR_GDEC] = np.array(par[v, t, 384]).astype(bfloat16)
                bo[p+PAR_BETA] = np.array(par[v, t, 385]).astype(bfloat16)
    return bo


def unpack_f_out(bo):
    """bf16 f_out BO -> (out_y, out_sf) full 128 cols, fp32."""
    out_y = np.zeros((NVH, K, HV), np.float32)
    out_sf = np.zeros((NVH, HV, HV), np.float32)
    for tile in (0, 1):
        off = tile * HALF_O
        for v in range(NVH):
            base = off + v * OUT_PKT
            for t in range(K):                                   # 8 y rows first
                p = base + t * COLS
                out_y[v, t, tile*COLS:(tile+1)*COLS] = bo[p:p+COLS].astype(np.float32)
            for i in range(HV):                                  # then 128 sf rows
                p = base + K*COLS + i * COLS
                out_sf[v, i, tile*COLS:(tile+1)*COLS] = bo[p:p+COLS].astype(np.float32)
    return out_y, out_sf


def mode_gen():
    s0, par = gen_input()
    y, sf = ref_chunkwise(s0, par)
    bo = pack_f_in(s0, par)
    bo.tofile(KDIR / "inp_2t.bin")
    y.tofile(KDIR / "ref_y.bin")
    sf.tofile(KDIR / "ref_sf.bin")
    print(f"gen: inp_2t {bo.shape} {bo.dtype} ({bo.nbytes/1e6:.1f}MB)")
    print(f"     ref_y {y.shape}, ref_sf {sf.shape}")
    print(f"     ref y range [{y.min():.6f}, {y.max():.6f}], S_final range [{sf.min():.6f}, {sf.max():.6f}]")


def mode_cmp():
    bo = np.fromfile(KDIR / "out_2t.bin", bfloat16)
    if bo.size != N_OUT:
        print(f"out_2t size {bo.size} != expected {N_OUT}"); return 1
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