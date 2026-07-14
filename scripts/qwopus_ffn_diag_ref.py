#!/usr/bin/env python3
"""qwopus_ffn_diag_ref.py — host ground-truth for the diag FFN dump (Phase 3b).

Reproduces the EXACT deterministic data of tools/qwopus_ffn_probe (h, W) and
computes, for the first packet (tile0 nc=0 kc=0 => outputs 0..15, K=0..1023,
i.e. the first k_chunk = G=32 groups):

  (a) fp32-precise  y[n]  = sum_k h[k] * dq(W[n,k])           (the probe's ref)
  (b) bf16-precise  y[n]  = sum_k bf16_trunc(h[k]) * dq(W[n,k])  (the NPU's math:
                          bf16 h + exact FP4*scale dequant + fp32 accumulate)
  (c) A@B^T variant  (in case the emulated mmul transposes B)

Compares (a)/(b) to the NPU's dumped cv0[0..7] (outputs 0..7) and cv1[0..7]
(outputs 8..15).  If (b) matches the NPU => the mmul is CORRECT and the only
bug is the extraction layout (t-fast).  If (c) matches => the mmul transposes
B and B_buf must be laid out N-fast.  If neither matches => a deeper operand
bug (A broadcast concat order, etc).
"""
import struct

FP4 = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
       0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]

HV_K=5120; GROUPS=HV_K//32; G=32; N=17408

def bf16_trunc(x):
    u=struct.unpack('<I',struct.pack('<f',x))[0]
    b=u>>16                      # truncate (keep top 16 bits) — engine's bf16
    return struct.unpack('<f',struct.pack('<I', b<<16))[0]

h=[(((k*37)%1000)-500)/500.0 for k in range(HV_K)]   # [-1,1], matches probe

W=[[None]*GROUPS for _ in range(N)]                  # W[n][g] = 17-byte block (list)
for n in range(N):
    for g in range(GROUPS):
        sc=120+((n*7+g*13)%21)
        if (n+g)%23==0: sc=0
        nibs=[(n*3+g*5+i*11)&0xFF for i in range(16)]
        W[n][g]=[sc]+nibs

def dq(n,k):   # dequant W[n,k] (fp32 exact; FP4*scale is exact in bf16 too)
    g=k//32; i=k%32
    blk=W[n][g]; sc=blk[0]
    if sc==0: return 0.0
    scale=2.0**(sc-127)
    byte=blk[1+(i>>1)]
    nib=(byte>>4)&0xF if (i&1) else byte&0xF
    return FP4[nib]*scale

# first k_chunk: K=0..1023 (g=0..31), outputs n=0..15
K0=0; K1=G*32  # 1024
def y_fp32(n): return sum(h[k]*dq(n,k) for k in range(K0,K1))
def y_bf16(n): return sum(bf16_trunc(h[k])*dq(n,k) for k in range(K0,K1))

print("NPU cv0[0..7] (outputs 0..7):")
print("  145699 65644.7 48035 30380.1 -1472.99 -18731.4 -163173 2480.19")
print("NPU cv1[0..7] (outputs 8..15):")
print("  184317 -42728.5 -2819.41 -48673.1 -64077.9 -1950.26 114293 15581.3")
print()
print("(a) fp32-precise y[0..7] :", [round(y_fp32(n),1) for n in range(8)])
print("(a) fp32-precise y[8..15]:", [round(y_fp32(n),1) for n in range(8,16)])
print("(b) bf16-precise y[0..7] :", [round(y_bf16(n),1) for n in range(8)])
print("(b) bf16-precise y[8..15]:", [round(y_bf16(n),1) for n in range(8,16)])

# (c) A@B^T hypothesis: the emulated mmul<4,8,8> computes C[r,t]=sum_s A[r,s]*B[t,s]
# with B laid out s-fast (B[s,t]=dq(W[output_t, K_sub*8+s])).  If the mmul instead
# reads B as B[t,s] (transposed), then per K-sub ks, per N-tile nt:
#   C[0,t] = sum_s h[K_sub*8+s] * B[t,s] = sum_s h[K_sub*8+s] * dq(W[output_s, K_sub*8+t])
# i.e. the s index (which should select K) selects the OUTPUT row, and t selects K.
# Replicate the exact K-sub/K-tile accumulation the kernel does:
colA=4; colB=2
def y_abT(n):
    # n is the output index 0..15; nt=n>>3, t=n&7
    nt=n>>3; t=n&7
    acc=0.0
    for g in range(G):                       # K-tile = group g (K=g*32..g*32+31)
        for ks in range(colA):               # K-sub within group (K=g*32+ks*8+s)
            for s in range(8):
                hval=bf16_trunc(h[g*32+ks*8+s])
                # B[t,s] under the s-fast layout = dq(W[output_s, K_sub*8+t])
                #   where output_s = nt*8 + s, K_sub*8+t = ks*8+t  (K within group)
                output_s = nt*8 + s
                kk = g*32 + ks*8 + t          # K index = ks*8 + t within group g
                acc += hval * dq(output_s, kk)
    return acc
print("(c) A@B^T      y[0..7] :", [round(y_abT(n),1) for n in range(8)])
print("(c) A@B^T      y[8..15]:", [round(y_abT(n),1) for n in range(8,16)])

# (d) SINGLE-GROUP (g=0 only, fresh accumulator — matches the new diag kernel):
#   y[n] = sum_{i=0..31} bf16_trunc(h[i]) * dq(W[n, g=0, i])   (one K-tile, no accum)
def y_g0(n):
    return sum(bf16_trunc(h[i])*dq(n,i) for i in range(32))
print()
print("NPU cv0[0..7] single-g : [237.916, 481.261, -323.982, 443.708, 501.821, 608.828, -262.226, 836.587]")
print("NPU cv1[0..7] single-g : [-531.294, -545.263, -643.515, -925.888, -696.589, 522.793, 983.621, 28.1237]")
print("(d) STD A@B   g0 y[0..7] :", [round(y_g0(n),2) for n in range(8)])
print("(d) STD A@B   g0 y[8..15]:", [round(y_g0(n),2) for n in range(8,16)])

# (e) A@B^T single-group: the mmul transposes B.  With B_buf[s,t]=dq(W[output_t,K+s])
# (s=K-fast, t=output), A@B^T gives C[r,t]=sum_s A[r,s]*B[t,s]=sum_s h[K+s]*dq(W[output_s,K+t]).
# So cv[n=nt*8+t] = sum_{ks,s} bf16h[ks*8+s] * dq(W[nt*8+s, ks*8+t]).
def y_g0_BT(n):
    nt=n>>3; t=n&7
    acc=0.0
    for ks in range(4):
        for s in range(8):
            acc += bf16_trunc(h[ks*8+s]) * dq(nt*8+s, ks*8+t)
    return acc
print("(e) A@B^T     g0 y[0..7] :", [round(y_g0_BT(n),2) for n in range(8)])
print("(e) A@B^T     g0 y[8..15]:", [round(y_g0_BT(n),2) for n in range(8,16)])

# (e') A@B^T but output mapping swapped (nt<->t) — try a couple index perms
def y_g0_BT2(n):
    nt=n&7; t=n>>3   # swap: low 3 bits = nt, high = t  (unlikely for 16-wide)
    acc=0.0
    for ks in range(4):
        for s in range(8):
            acc += bf16_trunc(h[ks*8+s]) * dq(nt*8+s, ks*8+t)
    return acc
print("(e')A@B^Tswap g0 y[0..7] :", [round(y_g0_BT2(n),2) for n in range(8)])
print("(e')A@B^Tswap g0 y[8..15]:", [round(y_g0_BT2(n),2) for n in range(8,16)])

# (e'') maybe the C-tile t maps to output but the K-sub t vs s swapped:
#   cv[n=nt*8+t] = sum_{ks,s} bf16h[ks*8+t] * dq(W[n, ks*8+s])   (s<->t in h vs W)
def y_g0_BT3(n):
    nt=n>>3; t=n&7
    acc=0.0
    for ks in range(4):
        for s in range(8):
            acc += bf16_trunc(h[ks*8+t]) * dq(nt*8+t, ks*8+s) if False else 0.0
    return acc