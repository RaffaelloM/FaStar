#!/usr/bin/env python3
"""cmp_fused_ab.py — check the fused GDN kernel's passA a/b at a given call index.

A3 v2 layout (1 MM2S + 1 S2MM).  FST_GDN_DUMP_SCR writes:
  callN_out.bin : [nV*129*1024] drain order per v-head = [128 S2 rows, 1 y-pkt].
                  The y-pkt is the held scratch [a|b|delta|y]: a@0, b@128,
                  delta@256, y@384.  y-pkt for v at (v*129+128)*1024.
  callN_spkt.bin: [nV*259*136] per v-head = [pkt_v(136)][pkt_kn(136)][pkt_qn(136)]
                  [128 passA rows(136)][128 passB rows(136)].
                  passA row i at [v, 3+i, :]: S0[i,j]@[0:128], kn_i@128, qn_i@129.

passA computes  a[v][j] = Σ_i S0[i,j]*kn_v[i] ;  b[v][j] = Σ_i S0[i,j]*qn_v[i].
We recompute the reference a/b straight from the spkt passA rows and compare to
the engine's drained a/b (from the held y-pkt) — isolating passA from delta/passB.

Usage: python3 cmp_fused_ab.py /tmp/scr_dump 48      # call48 = pos1 layer0
"""
import sys, numpy as np

HV, NV, NBLK, BLK = 128, 48, 128, 136
OUTPKT = 1024
NPPARAM = 3
NPKT_VH = 259          # 3 + 128 + 128
STRIDE = 2 * NBLK + 1  # 129 output packets per v-head


def load(p, n):
    a = np.fromfile(p, dtype=np.float32, count=n)
    if a.size != n:
        raise SystemExit(f"{p}: expected {n}, got {a.size}")
    return a


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scr_dump"
    call = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    out = load(f"{d}/call{call}_out.bin", NV * STRIDE * OUTPKT).reshape(NV, STRIDE, OUTPKT)
    spk = load(f"{d}/call{call}_spkt.bin", NV * NPKT_VH * BLK).reshape(NV, NPKT_VH, BLK)

    ypkt = out[:, NBLK, :]                          # [NV, 1024] held scratch (a|b|delta|y)
    eng_a = ypkt[:, 0:HV].copy()
    eng_b = ypkt[:, HV:2 * HV].copy()
    passA = spk[:, NPPARAM:NPPARAM + NBLK, :]       # [NV, 128, 136]
    S0   = passA[:, :, 0:HV]                        # [NV, 128, 128]  S0[i,j]
    kn_i = passA[:, :, HV + 0]                      # [NV, 128]  kn_v[i]
    qn_i = passA[:, :, HV + 1]                      # [NV, 128]  qn_v[i]
    ref_a = np.einsum("vij,vi->vj", S0, kn_i)
    ref_b = np.einsum("vij,vi->vj", S0, qn_i)

    print(f"call{call} (passA a/b vs reference from spkt):")
    print(f"  max|S0|={np.abs(S0).max():.4e}")
    worst = []
    for v in range(NV):
        da = float(np.abs(ref_a[v] - eng_a[v]).max())
        db = float(np.abs(ref_b[v] - eng_b[v]).max())
        worst.append((max(da, db), v, da, db))
    worst.sort(reverse=True)
    nok = sum(1 for m, *_ in worst if m < 1e-3)
    print(f"  {nok}/{NV} v-heads a/b match (<1e-3). Worst 5:")
    for m, v, da, db in worst[:5]:
        ra = float(np.abs(ref_a[v]).max())
        print(f"    vh{v:2d} max|Δa|={da:.4e} max|Δb|={db:.4e}  (|ref_a|max={ra:.4f})")
    print(f"  vh0 eng_a[:4]={eng_a[0,:4]}  ref_a[:4]={ref_a[0,:4]}")


if __name__ == "__main__":
    main()