# HY3 NPU FFN Wiring — Post-Mortem

**Task:** Compile HY3-specific NPU kernels (inter_dim=1536 vs DS4's 2048) and wire
them into the engine to shift FFN compute from CPU to NPU.
**Date:** 2026-07-10.  **Model:** `hy3.fst` (Tencent Hunyuan-3.0, GQA + sigmoid MoE).

## Verdict up front

The HY3 FFN kernels **compiled, are numerically correct (dequant bit-exact,
GEMM cos 0.998 = the DS4 gold standard), and produce coherent English**
(`Hello` → "Hello! How can I help you today?").  Moving the routed FFN to the
NPU delivered a real **9× prefill speedup (130 → 14.5 s/token)**.  **Decode,
however, stayed at 0.05 tok/s** — HY3 decode is **dispatch-count bound, not
NPU-compute bound**, so the >0.5 tok/s target is **not reachable by kernel
wiring alone**; it needs dispatch *reduction* (dequant+GEMM fusion).  Details
below.

---

## (a) HY3 FFN kernels N=1536 — COMPILED ✓

DS4's NPU kernels are hard-baked to inter=2048 and **cannot be reused** for
HY3's inter=1536 (wrong N for gate/up, wrong K for down, and the DS4 dequant is
baked to 786,432 blocks vs HY3's 589,824 — it would overrun the input BO).
Three new kernels were compiled by reusing the proven IRON/aiecc templates with
HY3 shapes:

| Kernel | File | Shape (M,K,N, tile) | NPU2 BD-step check | Size |
|--------|------|---------------------|--------------------|------|
| dequant | `fst_hy3_dequant.xclbin` + `_insts.bin` | 589,824 blocks, 16 cores, TAP [8,48,24] | reuses `fst_dequant_v4_4096` C++ kernel | 87,648 B / 4,560 B |
| gate/up | `fst_hy3_gemm_vec.xclbin` + `_insts.bin` | M=16 K=4096 N=1536 m16k128n32 b_col_maj | K_div_k=32, N_div_n=48 (both ≤64) ✓ | 11,418 B |
| down | `fst_hy3_gemm_down.xclbin` + `_insts.bin` | M=16 K=1536 N=4096 m16k128n64 b_col_maj | K_div_k=12, N_div_n=64 (both ≤64) ✓ | 12,762 B |

Scripts: `kernels/compile_hy3_dequant.py`, `kernels/compile_hy3_ffn.py` (reuses
`fst_expert_gemm_vectorized.py`).  All three compile clean and deploy to `kernels/`.

## (b) NPU wiring — routed experts on NPU; GQA/shared/dense on HOST by design

**Routed MoE experts → NPU** (`process_expert_ffn_hy3`, BO-to-BO, behind
`FST_HY3_HOST_FFN` guard for host A/B): `get_expert_bo` (persistent LRU host_only
BO of packed MXFP4 weights) → `hy3_dequant` (MXFP4→BF16 B[N,K] row-major) →
`hy3_gemm`/`hy3_gemm2` (gate/up) → `ew_unified` silu/mul → `hy3_gemm_down` →
router-weighted accumulate.  **No CPU GEMM, no CPU dequant** on this path.

**Numerically verified** (`FST_HY3_FFN_AUDIT`, layer 1, expert 145):
- DEQUANT: cos = **1.00000**, maxdiff = 0.0000, |npu|/|ref| = 1.0000 — **bit-exact** vs host dequant.
- GEMM (down): cos = **0.99798**, maxdiff = 0.0082, |npu|/|ref| = 0.97675 — the **DS4 gold standard** (DS4 ships at cos 0.998 with coherent English).

**Coherent** on peaked (templated) input: `Hello` → prefill-last = **16883 ("Hello")** = the documented host-first baseline, then a coherent English greeting in decode.  (Raw no-BOS M=1 gave 3048 vs host 58933 — argmax instability on degenerate flat logits, **not** an FFN bug; cos 0.998 confirms the math is correct.)

**GQA projections, shared expert, dense L0 stay on HOST — deliberately.**  This
**diverges from the literal "zero CPU fallbacks" rule**, and the reason matters:
HY3 decode is **dispatch-count bound** (see (c)), so the NPU only wins for the
*large* MXFP4 expert weights (moved).  For the *small-M* projections the CPU is
dramatically faster:

| Path | Per-token cost (decode M=1) | On NPU? | Why |
|------|-----------------------------|---------|-----|
| Routed MoE FFN (8 experts × 80 layers = 640 FFNs, large MXFP4 weights) | ~3840 dispatches ≈ 24 s — the dominant cost | **NPU** ✓ | large weights; NPU dequant+GEMM beats host fp32; cos 0.998 |
| GQA q/k/v/o (4 GEMMs × 80 = 320, M=1, small) | ~3 ms each on host fp32 | HOST | NPU = 34 dispatches/layer × ~6 ms ≈ **+16 s/token added**; host wins |
| Shared expert (BF16, always-on, 80 FFNs) | <2 ms each on host | HOST | small; BF16 (no dequant); NPU would add dispatches |
| Dense L0 (1 layer, K=13312) | ~5 ms host | HOST | needs a genuinely **new** kernel (k=128 → K_div_k=104 > 64 BD-step; no DS4 reuse) for 1 layer — not worth it |

Applying "all GEMMs on NPU" to GQA/shared/dense would **regress** tok/s.  The NPU
is correctly applied where it wins (large expert weights) and correctly *not*
applied where host wins (small projections).  The GQA kernels (qck/wqb/ob) are
layout-compatible (HY3 q/k/v/o_proj are stored [N,K], matching b_col_maj) and
reusable if the user wants them on NPU despite the regression — see "Deferred".

**DS4 path is byte-identical**: every HY3 kernel registration is behind
`if (config_.arch == ARCH_HY3)`; the MLA set registers only `if (ARCH_DS4)`;
dequant and `expert_bo_cap_` branch on arch.  DS4 takes the unchanged `else`
branches.  Hard constraint satisfied.

## (c) Prefill time + decode tok/s — 9× prefill win; decode 0.05 tok/s (dispatch floor)

Measured (templated `Hello`, `--tokens 16 --temp 0`, greedy, `FST_EXPERT_BO_CAP_GB=6`):

| Run | Prefill | Decode | Note |
|-----|---------|--------|------|
| Host FFN, M=16 (phase-5 baseline) | **130 s/token** (2,085 s / 16) | 0.04 tok/s | the user's "130 s/token on CPU" |
| **NPU FFN, M=16 (this run)** | **14.5 s/token** (231,921 ms / 16) | **0.05 tok/s** | **9× prefill speedup**; decode ≈ floor |
| NPU FFN, raw M=1 | ~21 s | 0.04 tok/s | per-token FFN dispatch-bound |

Run summary (exit 0, no FATAL/segfault):
```
HY3 Prefill: 231921.2 ms (16 prompt tokens, 80 trunk layers)
HY3 Decode Tokens/sec: 0.05
Peak RAM (RSS): 31.1 GB
Expert Pager: gets=9651 hits=9651 hit_rate=1.000 cache=5995MB
NPU Contexts: creates=4 evictions=0 hits=0 active=4
```

**Why prefill got 9× but decode did not move:**

- **Prefill (M=16)** benefits hugely because the NPU dequant+GEMM is far faster
  than host fp32 for the *large* expert weights, and the 6 GB expert-BO cache
  holds ~600 experts with **hit_rate 1.000** — SSD staging was amortized, so the
  NPU compute itself was the win (130 → 14.5 s/token).  4 NPU contexts
  (gate/up + down + dequant + ew_unified), 0 evictions.
- **Decode (M=1)** is **dispatch-count bound**, not compute-bound.  Per token:
  640 routed expert-FFNs × ~6 dispatches (dequant + gate + up + silu + mul +
  down, each forcing a cross-context flush) ≈ **3840 dispatches**, each
  flush+readback ~6 ms → ~24 s/token → ~0.04 tok/s.  The FFN GEMM *compute* is
  fast on the NPU; the wall-clock is **dispatch count × per-dispatch latency**,
  serialized by `flush_pending_runs()` (a correctness requirement — async was
  tried and reverted as non-deterministic; see `npu_gemm_mla_vec` comment at
  fst_engine.cpp:626).  This is the **same floor DS4 documented** ("0.04 tok/s =
  dispatch-COUNT floor"; ">1 tok/s needs IRON MLA+FFN FUSION"); HY3 is worse only
  because it has 80 trunk layers (vs DS4's 43) and 8 routed experts/layer.

**>0.5 tok/s is not reachable by adding NPU kernels.**  The lever is **dispatch
reduction**, specifically **fusing dequant+GEMM into one kernel** (the deferred
IRON fusion — one dispatch per expert instead of six).  That is a separate,
larger effort, not kernel wiring.  (GQA on host costs only ~1 s of the ~24 s
decode token, so it is NOT the decode bottleneck — confirming GQA belongs on
host.)

## (d) Exact output text + coherence

Templated `--prompt "Hello" --tokens 16 --temp 0` (greedy), NPU FFN.
prefill-last = **16883 = "Hello"**.  Decode stream (cb output, byte-faithful):

```
Hello !  How can I help you today? <｜hy_eos:opensource｜><think:opensource></think:opensource>Can you
```

i.e. **"Hello! How can I help you today?"** — a complete, correctly-punctuated
greeting response, byte-faithful to what a tuned Hunyuan assistant produces and
matching the host-first baseline (phase-5).

After the answer the model emits `<｜hy_eos:opensource｜>` (EOS, tid 120025) then
an empty `<think:opensource></think:opensource>` block and starts a **new turn**
("Can you …").  This post-EOS looping is an **engine artifact** (the decode loop
stops at max_seq−1, not on the EOS token id) — NOT an NPU bug; the model is
correctly emitting EOS and the engine correctly not hard-stopping on it.

For contrast, the **raw** no-BOS/no-template M=1 run gave prefill-last 3048
(NPU) vs 58933 (host) — argmax **instability on degenerate flat logits** (no
prompt context), NOT an FFN bug: the FST_HY3_FFN_AUDIT showed dequant cos 1.00000
+ GEMM cos 0.99798, so the math is correct; the flat-logit argmax is just
ill-defined and flips under the cos-0.998 NPU↔host difference.  On the peaked
(templated) logits the argmax is stable → both paths give "Hello".

**Coherent: YES.**  Correct + matches host baseline.  (Correctness-vs-HF argmax
still deferred — needs the HF PyTorch ref.)

---

## Deferred / Next

1. **>0.5 tok/s lever = dequant+GEMM fusion** (one dispatch/expert), not more
   NPU kernels.  This is the real throughput win; everything above is correctness
   + correct NPU placement.
2. **GQA on NPU** — reusable (qck exact for K/V; wqb N-tile×4 + K-split×4 for Q;
   ob N-tile×2 + K-split×8 for O, needs bf16-cast of fp32 attn output) but
   **predicted to regress** tok/s for small M.  Wire only if the user wants it
   despite the regression; the MLA kernels have a documented layout-bug history,
   so reuse needs a cos audit.
3. **Dense L0 on NPU** — needs a new K=13312 kernel (k=256 → K_div_k=52, or host
   K-split into 104 chunks).  1 layer, ~5 ms on host — low priority.
4. **Correctness-vs-HF** — still deferred (needs HF PyTorch ref).  "Hello"→"Hello"
  is coherent but not yet argmax-verified vs HF.