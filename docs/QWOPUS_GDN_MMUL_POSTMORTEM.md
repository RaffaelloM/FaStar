# QWOPUS GDN Slab MMUL Re-Investigation — MEASURED POSTMORTEM

**Date:** 2026-07-14
**Status:** Step 2 of the "do not accept 0.20 tok/s ceiling" re-investigation.
**Verdict:** The prior "bank-contention / accept 0.20 ceiling" verdict was
**partly wrong in mechanism and unmeasured on the MMUL lever**. Re-measured
thoroughly: the slab is **still dead**, but now on **measured** ground. The
real blocker is **BFP16 quantization amplified by the S-feedback recurrence**
(Not "bank contention"), plus a **mul-instruction contention cost** the nomul
canary exposed.

## What the prior verdict got wrong
The prior postmortem (docs/QWOPUS_GDN_8KSLAB_ENGINE_POSTMORTEM.md) declared the
slab dead at 1180 ms with root cause "concurrent s0/snew/yd DMA bank-contention
with stack-S RMW" and concluded "MMUL can't help (memory-bound)." That bound was
**never measured** — it used the *uncontended* synthetic-probe math time
(79.73 ms) as the MAC cost. The user correctly rejected this as an unverified
assumption and demanded the MMUL lever be built and measured.

## The measurements (all 4-tile, K=8, start=0, best of 5, same dump)

| variant | passA math | latency | argmax | max\|S\| (ref 18.22) |
|---|---|---|---|---|
| scalar (shipped-style slab) | fp32 scalar-broadcast MAC | **1180 ms** | 99.5% | 17.88 |
| **nomul** canary | MACs→adds (loads preserved) | **520 ms** | (wrong, canary) | — |
| **vecmul** canary | `mul(sv,ones)` vec×vec, no bcast | **973 ms** | (wrong, canary) | — |
| bfp16 `<8,8,8>` mmul | passA mmul, passB scalar | **733 ms** | 83.6% ❌ | 50.75 ❌ explode |
| native `<8,8,4>` mmul | passA mmul, passB scalar | **886 ms** | **99.5% ✅** | **17.88 ✅** |
| fp32 `<8,8,8>` mmul | — | **compile fail** | — | — |

Gate = 720 ms (90 ms/tok × K=8). Shipped 3-pass = 720 ms for 8 tok.

### 1. nomul canary — MACs are 56%, NOT memory-bound
`fst_gdn_8kslab_nomul`: identical memory structure (same loads, same DMA, same
packet layout) but `a += sv` instead of `a += sv*kn`, `S = gdec*S + delta`
instead of `+kn*delta`. **520 ms** vs 1180 ms scalar ⇒ the MACs are
**660 ms (56%)**. The slab is NOT purely memory-bound; attacking the MACs
(MMUL) is worth doing. The prior "memory-bound, MMUL can't help" bound was
wrong (it used the uncontended probe's 79.73 ms).

### 2. vecmul canary — decomposes the 660 ms
`fst_gdn_8kslab_vecmul`: `mul(sv_f, ones_f)` — a **vector×vector** mul (no
per-row scalar broadcast), math wrong (latency canary). **973 ms**.
- `aie::mul` **instruction itself** = 973 − 520 = **453 ms** under DMA
  contention, even with no broadcast.
- per-row **scalar broadcast** = 1180 − 973 = **207 ms** extra.
Both contribute; the mul instruction is the larger cost. MMUL (a different,
matrix-mac instruction) attacks exactly this.

### 3. bfp16 `<8,8,8>` mmul — fast but EXPLODES the recurrence
`fst_gdn_8kslab_mmul` (passA → `aie::mmul<8,8,8,bfloat16>`). **733 ms**
(recovered 447 ms!). But **broken**: argmax 83.6%, max|S| = 50.75 (2.8× ref),
per-token Δ grows 0.002 → 1.48 — **S explodes**.

Root cause (verified in `mmul_bf16_bf16.hpp`): `<8,8,8,bfloat16,bfloat16>` is
**UNCONDITIONALLY** BFP16-converted — line 112 `to_v64bfp16ebs8(acc_a)` +
`mac_8x8_8x8T_conf`. The `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` flag only
gates `<4,8,8>` (line 63), **not** `<8,8,8>`. BFP16 = block-floating-point
(shared exponent per 8-element block) ⇒ ~1e-3 per-product quantization. This
is fine for a **non-recurrent** FFN (single matvec — that's why the FFN's bf16
mmul worked), but the GDN **S-feedback loop amplifies that bias over K=8
tokens ⇒ explosion**. The recurrence demands **fp32-exact products**.

Proof it's bfp16 (not a layout bug): `mul`-then-`mac` (explicit zeroing) gave
identical results (not a stale accumulator); emulate-flag-ON vs OFF gave
identical results (both bfp16 — there is **no exact bf16 mmul at `<8,8,8>`**);
tk0 correct (0.002) with growth only tk1+ (error accumulation through S
feedback, not a constant layout bug); delta/passB byte-identical to the working
99.5% scalar slab ⇒ isolated to passA.

### 4. native `<8,8,4>` mmul — CORRECT, but over the gate
`fst_gdn_8kslab_mmuln4`: `aie::mmul<8,8,4,bfloat16>` uses `mac_8x8_8x4_bf16` —
**native bf16, NO bfp16 conversion** ⇒ exact products. **886 ms, 99.5% argmax,
max|S| = 17.88 (no explosion), flat per-token Δ** — same correctness as the
scalar slab. So native `<8,8,4>` **is** the exact-MMUL path and it works
numerically.

But 886 ms = **110.8 ms/tok > 90 ms gate** (1.23× slower than shipped 720 ms).
N=4 (the max native N; no native N=8 exists) forces **2× the mac instructions**
of N=8 ⇒ macs = 366 ms vs bfp16's 213 ms. aie_api has no 4-element bfloat16
vector (`extract<4>` undefined — min width 8), so B is packed via scalar 4-copy
and C scattered via a 32-float temp buffer; that overhead is negligible (886 ≈
733 + 153 extra-mac-instructions) — the 2× macs is the cost, not the packing.

### 5. fp32 mmul — unsupported at the needed size
`aie::mmul<8,8,8,float,float>`: **compile error** — no `accum_type` (no fp32
8×8×8 mmul on AIE2P); `transpose<float,64>` also undefined. fp32 mmul exists
only at tiny shapes (`<4,4,8>`, `<4,2,8>` — M≤4, K≤4) ⇒ 2× the macs AND 2×
operand bytes vs bf16 `<8,8,8>` ⇒ guaranteed > 733 ms. Dead for the gate.

## Decisive bound
**memory floor 520 ms (nomul) + fastest-possible macs 213 ms (bfp16
`<8,8,8>`) = 733 ms > 720 ms gate.** The gate is **unreachable even with the
broken macs**. The correct native macs (366 ms) ⇒ 886 ms. **No MMUL variant can
make the slab beat the shipped 3-pass GDN.**

The 520 ms floor is **per-token on-tile stack access** (S RMW 128×32×2×K =
6 MB stack traffic), NOT DMA (S0+snew DMA ≈ 3 ms) ⇒ **larger K does NOT
amortize it** — 520/8 = 65 ms/tok stack + 46 ms/tok native mac = 111 ms/tok,
flat in K.

## Step 2.1 — "is S DMA'd per-token inside the K-loop?" (user's concern)
**Already satisfied.** `gdn_8kslab_vhead` loads S0 → 8 KB stack-S **once**,
loops K=8 `recur_core` (stack-S RMW, **no S DMA in the loop**), stores snew
**once**. Only `yd` (out) and `par` (in) are per-token. S is fully isolated on
the stack during the K-recurrence. The canary (S += 1×K, same DMA) proved stack
S **persists** across the K-loop (snew = S0+8 exactly).

## Correction to the Step-3 (speculative decoding) premise
The user's plan states SD with M=4 "hinges on fixing the GDN M=K kernel." That
premise is **incorrect**: the GDN recurrence is **sequential over tokens
(M=1 always, even under SD)** — the S-state update is token-by-token, so GDN
cannot use M≥4 regardless of SD. SD's win is **FFN-only** (native FFN MMUL at
M≥4 beats CPU: 1.89 ms vs 10.7 ms, per the FFN postmortem), and it is
**independent** of GDN. GDN stays M=1 / shipped-3-pass under SD. So Step 3
(FFN-under-SD) can proceed **without** fixing GDN M=K (which is impossible
anyway).

## Conclusion
The slab M=K GDN is dead on **measured** ground:
- bfp16 mmul (fastest, 733 ms) **explodes** the recurrence — BFP16 quantization
  amplified by the S feedback.
- native `<8,8,4>` mmul is **correct** (99.5%) but 886 ms > 720 ms gate
  (N=4 ⇒ 2× macs; no native N=8).
- fp32 mmul unsupported at `<8,8,8>`; would be slower anyway.
- memory floor (520 ms) + fastest macs (213 ms) = 733 ms > 720 ms gate ⇒
  unreachable even with broken macs.

FST_Q35_GDN_8KSLAB remains **never wired** (shipped path untouched, nothing to
revert). The real GDN lever = speculative decoding (dispatch amortization on
the shipped 3-pass path), **not** on-tile recurrence MMUL speed.

## Artifacts
- `kernels/fst_gdn_8kslab_{nomul,vecmul,mmul,mmulnat,mmuln4}_kernel.cc`
  (`mmulfp` failed to compile — fp32 `<8,8,8>` unsupported)
- `kernels/gen_gdn_8kslab.py` (FST_GDN_SRC/SUFFIX env)
- `tools/gdn_8kslab_engine_probe.cpp` (argv[4] = stem)
- `logs/gdn_8kslab_{nomul,vecmul,mmul,mmulnat,mmuln4}.log`