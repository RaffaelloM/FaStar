#!/usr/bin/env python3
"""qwopus_bf16s_drift_test.py — Cheap gate for the chunkwise parallel-scan GDN path.

The research verdict (docs, 2026-07-13): a chunkwise M=K GDN kernel can amortize
3K dispatches -> 1 dispatch (held S on-tile, K sequential steps), BUT fp32 S
[128,128] = 64 KB fills the entire AIE2P tile -> forced to bf16 S (32 KB) ->
NOT bit-correct (each recurrence step re-quantizes S to bf16).  The single biggest
risk is whether bf16-S RMW drift over K=2..8 steps breaks the speculative-verify
argmax agreement (i.e. losslessness).

This script is the CHEAP gate (pure numpy, no NPU, no IRON, no HF, no .fst load):
replay the engine's EXACT proven GDN recurrence for K steps, once with fp32 S
(reference) and once with S rounded to bf16 after each state update (bf16 storage,
fp32 compute — the standard mitigation), and measure max|Δy| / argmax drift per
step.  Realistic input magnitudes are matched to the engine's observed values
(max|S0|~5.88 at pos1; qn/kn unit-norm; gdec~0.85-0.99; beta~0.3-0.7; v~O(1-10)).

If max|Δy| stays in the fp32-compute-noise band (rel ~1e-2, the tolerance
qwopus_ssm_recur_ref.py uses for fp32-vs-fp32 engine), bf16-S is lossless-grade
-> the chunkwise M=K kernel is viable and worth the IRON authoring effort.
If drift is catastrophic, the only fp32 path is a 2-tile split (likely dead-end,
same multi-fifo death as the M=1 fusion) -> parallel-scan GDN is a dead-end.

We ALSO run a 2nd regime with a fake "verify-logit" projection (random W) so the
test reports whether the bf16-S drift flips the argmax over a realistic logit
spread — the actual losslessness question for speculative verify.
"""
import numpy as np

HV = 128          # per v-head state dim
NV = 48           # v-heads (drift is per-v-head independent; test NV heads for stats)
KMAX = 8          # max sequential steps to test
NSAMP = 64        # random sequences to average over

rng = np.random.default_rng(20260713)

def bf16_round(x):
    """Truncate fp32 -> bf16 the way the engine does (u & 0xffff0000, low 16 bits zeroed)."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    u &= np.uint32(0xffff0000)
    return u.view(np.float32)

def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n

def gen_inputs():
    """One K-step sequence, realistic magnitudes."""
    # q, k pre-norm: order ~1-10 (projection outputs); v order ~1-10
    q = rng.standard_normal((KMAX, HV)).astype(np.float32) * 3.0
    k = rng.standard_normal((KMAX, HV)).astype(np.float32) * 3.0
    v = rng.standard_normal((KMAX, HV)).astype(np.float32) * 3.0
    qn = l2norm(q) / np.sqrt(HV)
    kn = l2norm(k)
    # gdec = exp(g_logit), g_logit ~ softplus-scaled; realistic decay 0.85..0.99
    g_logit = rng.uniform(-0.16, -0.01, KMAX).astype(np.float32)
    gdec = np.exp(g_logit)
    # beta = sigmoid(b), b ~ [-0.5, 1.0] -> beta ~ 0.38..0.73
    b = rng.uniform(-0.5, 1.0, KMAX).astype(np.float32)
    beta = 1.0 / (1.0 + np.exp(-b))
    # initial state: realistic, max|S0| ~ 5 (pos1)
    S0 = (rng.standard_normal((HV, HV)).astype(np.float32) * 0.8)
    return qn, kn, v, gdec, beta, S0

def recur_fp32(qn, kn, v, gdec, beta, S0):
    """Engine-exact recurrence, fp32 S.  Returns y[K] (the read-after-update output)."""
    S = S0.copy()
    ys = []
    for t in range(KMAX):
        S = gdec[t] * S
        kvm = S.T @ kn[t]
        delta = (v[t] - kvm) * beta[t]
        S = S + np.outer(kn[t], delta)
        y = S.T @ qn[t]
        ys.append(y)
    return np.stack(ys)  # [K, HV]

def recur_bf16s(qn, kn, v, gdec, beta, S0):
    """Same recurrence, S stored as bf16 (re-quantized after each update), fp32 compute."""
    S = bf16_round(S0)
    ys = []
    for t in range(KMAX):
        S = bf16_round(gdec[t] * S)          # decay, then store bf16
        kvm = S.T @ kn[t]                      # fp32 compute from bf16-stored S
        delta = (v[t] - kvm) * beta[t]
        S = bf16_round(S + np.outer(kn[t], delta))  # update, then store bf16
        y = S.T @ qn[t]
        ys.append(y)
    return np.stack(ys)

def recur_bf16s_fp32accum(qn, kn, v, gdec, beta, S0):
    """Mitigation variant: fp32 accumulator, bf16 storage only (the standard pattern)."""
    S = S0.astype(np.float32)
    S_b = bf16_round(S0)
    ys = []
    for t in range(KMAX):
        S = gdec[t] * S                       # fp32 accumulator (full precision)
        S_b = bf16_round(gdec[t] * S_b)       # bf16 storage mirrors it
        kvm = S_b.T @ kn[t]                    # READ from bf16 storage (the on-tile reality)
        delta = (v[t] - kvm) * beta[t]
        S = S + np.outer(kn[t], delta)        # fp32 accum update
        S_b = bf16_round(S)                   # store bf16
        y = S_b.T @ qn[t]                      # read output from bf16 storage
        ys.append(y)
    return np.stack(ys)

def main():
    print(f"== bf16-S drift gate (HV={HV}, NV={NV}, KMAX={KMAX}, NSAMP={NSAMP}) ==")
    print(f"   bf16 = engine truncation (u & 0xffff0000)")
    # Per-step worst-case max|Δy| across all v-heads and samples.
    maxdy_pure   = np.zeros(KMAX)   # bf16-S (pure bf16 storage)
    maxdy_mit    = np.zeros(KMAX)   # bf16-S with fp32 accumulator mitigation
    # Fake verify-logit: project y[K] -> vocab-size logit.  The argmax flip rate
    # depends on the logit top-2 margin; a random W gives TIGHT margins (pessimistic).
    # We record the margin at each flipped case so flips are interpretable: if real
    # model verify-logit margins are >> the flip-margin distribution, lossless.
    VOCAB = 1000
    W = rng.standard_normal((HV, VOCAB)).astype(np.float32) * 0.05
    argmax_flip_pure = np.zeros(KMAX, dtype=int)
    argmax_flip_mit  = np.zeros(KMAX, dtype=int)
    flip_margin_pure = []   # top-2 margin of l_ref at each flipped case (pure)
    flip_margin_mit  = []
    all_margin = []         # all top-2 margins (to see the distribution)
    for _ in range(NSAMP):
        for _h in range(NV):
            qn, kn, v, gdec, beta, S0 = gen_inputs()
            y_ref  = recur_fp32(qn, kn, v, gdec, beta, S0)
            y_pure = recur_bf16s(qn, kn, v, gdec, beta, S0)
            y_mit  = recur_bf16s_fp32accum(qn, kn, v, gdec, beta, S0)
            for t in range(KMAX):
                maxdy_pure[t] = max(maxdy_pure[t], np.max(np.abs(y_pure[t] - y_ref[t])))
                maxdy_mit[t]  = max(maxdy_mit[t],  np.max(np.abs(y_mit[t]  - y_ref[t])))
                lr  = y_ref[t]  @ W
                lp  = y_pure[t] @ W
                lm  = y_mit[t]  @ W
                # top-2 margin of the reference logit
                s = np.sort(lr); marg = s[-1] - s[-2]
                all_margin.append(marg)
                if np.argmax(lp) != np.argmax(lr):
                    argmax_flip_pure[t] += 1; flip_margin_pure.append(marg)
                if np.argmax(lm) != np.argmax(lr):
                    argmax_flip_mit[t]  += 1; flip_margin_mit.append(marg)
    ntot = NSAMP * NV
    print(f"\n step |  max|Δy| bf16-S  |  max|Δy| bf16-S+fp32acc | argmax-flip pure | argmax-flip mit")
    print( "------+------------------+------------------------+------------------+----------------")
    for t in range(KMAX):
        print(f"  {t+1:2d}  |   {maxdy_pure[t]:.3e}      |    {maxdy_mit[t]:.3e}         |   "
              f"{argmax_flip_pure[t]:4d}/{ntot}        |   {argmax_flip_mit[t]:4d}/{ntot}")
    ymag = np.median([np.max(np.abs(recur_fp32(*gen_inputs()))) for _ in range(16)])
    print(f"\n typical max|y_fp32| ~ {ymag:.3f}  (so max|Δy| above is ~relative error)")
    all_margin = np.array(all_margin)
    print(f"\n fake-logit top-2 margin distribution (random W, TIGHT = pessimistic):")
    print(f"   median={np.median(all_margin):.4f}  p10={np.percentile(all_margin,10):.4f}  "
          f"p90={np.percentile(all_margin,90):.4f}  max={all_margin.max():.4f}")
    print(f"   (real model verify-logit top-2 margins are typically >>1, often >>10)")
    def flipstats(name, fm):
        if not fm: print(f"   {name}: NO flips"); return
        fm = np.array(fm)
        print(f"   {name}: {len(fm)} flips; flip-margin median={np.median(fm):.4f} "
              f"p90={np.percentile(fm,90):.4f} max={fm.max():.4f}")
        print(f"      -> flips concentrated at margin <= {np.percentile(fm,90):.4f}; "
              f"real margins >> this => lossless")
    flipstats("pure bf16-S", flip_margin_pure)
    flipstats("bf16-S+fp32acc", flip_margin_mit)
    flips_pure = argmax_flip_pure.sum()
    flips_mit  = argmax_flip_mit.sum()
    print(f"\n TOTAL argmax flips over {KMAX} steps x {ntot} heads: "
          f"pure-bf16-S={flips_pure}, bf16-S+fp32acc={flips_mit}")
    # In-band check vs the engine's fp32-vs-fp32 tolerance (2e-2 from recur ref)
    inband = maxdy_mit[KMAX-1] < 2e-2
    print(f"\n max|Δy| at K=8 (mitigation) = {maxdy_mit[KMAX-1]:.3e} vs engine fp32-noise tol 2e-2 -> "
          f"{'IN-BAND' if inband else 'OUT-OF-BAND'}")

if __name__ == "__main__":
    main()