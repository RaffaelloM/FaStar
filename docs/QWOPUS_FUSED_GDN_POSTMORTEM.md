# QWOPUS Fused GDN (1-dispatch) — Post-mortem

Date: 2026-07-12
Branch: `hy3-pivot-openmp-simd`
Model: `Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF` → `qwopus.fst` (14.52 GB), AMD Ryzen AI 9 365 (XDNA2 NPU).

This documents the attempt to (a) find an SSM wiring bug, (b) fuse the 3 GDN
passes into 1 NPU dispatch, (c) reach >5 tok/s, and (d) produce coherent output.
The task's framing rested on a premise that turned out to be false; the real
blockers are elsewhere. Reported honestly below.

---

## (a) SSM Wiring Bug — DISPROVEN (no wiring bug exists)

The task hypothesized an SSM wiring bug causing the empty-assistant-turn
incoherence (`\n<|im_start|>assistant<|im_end|>` loop). It does not exist.

**Evidence the SSM block is bit-correct** (validated against HF
`transformers.models.qwen3_next` `Qwen3NextGatedDeltaNet`):

- `scripts/qwopus_ssm_block_ref.py` (pos 0, all tensors): the full chain
  `an → qkv → qkvc (conv1d) → qn/kn/v → y_scan → y_normgated → o_proj → out`
  matches HF at pos 0, `max|Δout| = 3.8e-6` (the only outlier is `h_out` at
  2.8e-3 = bf16 residual quantization on the residual add, NOT a bug).
  Conv1d state carry + GDN S-state carry are correct at pos 0.
- `scripts/qwopus_ssm_recur_ref.py` (recurrent HF replay over the engine's
  per-position `an_p`): pos 0 matches — `y 2.5e-7 | y_ng 7.2e-6 | out 2.4e-6 |
  S_post 8.6e-6`.

**Caveat on the recur ref's `S_pre` column:** line 148 captures `rec_state`
*after* `torch_recurrent_gated_delta_rule`, so the printed `S_pre` is actually
HF's S_post, not S_pre. The "DIVERGE at pos 0" label is an artifact of that
buggy comparison; pos 0's `y / y_ng / out / S_post` all match. The recur-ref
script itself needs that line moved above the delta-rule call, but it does not
affect the `y / out / S_post` comparisons, which are valid.

**Conclusion:** the SSM block (projections, conv1d state, l2norm, decay/beta,
delta-rule recurrence, normgated, o_proj) is wired correctly. The incoherence is
**NOT in SSM**. It is downstream/upstream — in GQA (`process_gqa_q35`), FFN,
MTP, lm_head, or the chat template. **Uninvestigated.** That is the real next
target.

(The SSM numerics themselves were fixed earlier — see
`docs/QWOPUS_SSM_NUMERICS_POSTMORTEM.md`: `qn=l2norm(q)/√128`, `kn=l2norm(k)`,
`gdec=exp(ssm_a·softplus(a+dt_bias))` with `ssm_a` already `−exp(A_log)`,
`beta=sigmoid(b)`, `S=gdec·S; kvm=Sᵀ@kn; delta=(v−kvm)·beta; S+=kn⊗delta;
y=Sᵀ@qn`. The kernel `kernels/fst_ssm_scan_kernel.cc` is a pure fp32 delta-rule,
no transcendentals.)

---

## (b) GDN Fusion — structurally 3→1, but NOT bit-correct (racy)

**Goal:** collapse `gdn_passA → gdn_delta → gdn_passB` (3 xclbins / 3
`xrt::run` per SSM layer = 144 dispatches/token) into a SINGLE AIE dispatch per
SSM layer = 48 dispatches/token. ZERO multi-dispatch per SSM layer. ZERO CPU
fallbacks for math.

**Structural result: achieved.** `kernels/gen_gdn_fused.py` builds ONE xclbin
`fst_gdn_fused.xclbin` with one Worker that loops all 48 v-heads internally, and
for each v-head runs passA (128 blocks) → delta (once) → passB (128 blocks)
against a held tile scratch `[a|b|delta|y]`, streaming S0 from DDR twice (the 64
KB state does not fit in a 64 KB tile). `src/fst_engine.cpp:gdn_scan_vheads`
issues a single `run_registered_blob("q35_gdn_fused", args, 4).wait()` per SSM
layer. **144 → 48 dispatches/token structurally.** No CPU math fallback.

**Correctness result: NOT bit-correct.** The 3-pass kernel
(`fst_gdn_passA/delta/passB`, 3 xclbins) is bit-correct across 13 positions. The
fused kernel is not. The bug is specific and isolated:

- It is **masked at pos 0** (S=0 ⇒ a=b=0 regardless of RMW correctness, so pos-0
  `y` and `S_post` match). It only appears at pos 1+, where S≠0 and the passA
  RMW accumulation matters.
- At pos 1, **only the first v-head (vh0) is wrong**: `a ≈ 0.182 vs ref 0.249`
  (Δ=0.067, ~73% of correct — as if exactly one of the 128 row blocks' S_row
  contribution was missed/stale). The other 47 v-heads' `a/b` are correct.
  `max|Δy|` at pos 1 ≈ 0.06 (vh0's 128 elements) and compounds to 13.3 by pos 3
  (vh0's wrong `S_post` feeds vh0's next-position `S_pre`).
- It is **NOT a steady-state RMW corruption** (that would hit all v-heads
  randomly). It is a **first-v-head startup race**.

**Root-cause analysis** (the progression is the evidence):

| variant | f_par handling during passA | result |
|---|---|---|
| original | `f_par` acquired BEFORE passA and held across the RMW | **all 48 v-heads garbage** |
| late-par | `f_par` acquired just-before-delta (after passA) | **only vh0 garbage, 47/48 correct** |

The 3-pass passA is structurally identical to the fused passA (acquire held
output → `kz` zero via `store_v` → loop rows acquiring input → `kA` RMW with
`load_v`/`store_v` → release), with the same `kz→load_v` pattern, and is
bit-correct. So the bug is NOT the kz→load hazard and NOT the RMW itself. The
difference is that the fused tile has **4 ObjectFifos** (f_s0 in, f_par in,
f_scr out, f_s2 out) with **two concurrent MM2S fills** (f_s0 + f_par) issued at
task-group start, whereas the 3-pass passA tile had 2 fifos / 1 fill. The
residual vh0 error is the signature of the **first `f_s0.acquire(1)` returning
before its packet is fully DMA'd** because f_par's fill is competing for the
shim MM2S channel at startup. (The 3-pass kernel's own comment notes that "two
simultaneous held outputs broke the shim drain flush on this IRON build" — this
build has known fifo-interaction bugs.)

**What did NOT fix it:**
- 8-row blocks → 1-row-per-block (no intra-block RMW race): same vh0 garbage.
- Register-accumulator passA (hold a/b in `aie::vector` regs): correct-ish but
  20× slower (unroll code bloat); abandoned.
- Late-par (above): reduced all-v-heads → vh0-only; did not eliminate it.

**Attempted but UNVALIDATED (harness outage — see (e)):**
- `f_s0` depth=4 (more DMA runway) + `f_par` depth=1 (force its fill to block
  after 1 packet, minimizing concurrency during vh0 passA). Compiled clean. The
  validation run **crashed/killed at prefill tok 9/14** (exit 1, no error in
  stdout/stderr — could be OOM since the `systemd-run MemoryMax=35G` wrapper was
  omitted, or a kernel regression from `depth=1`). Cause could not be
  determined because the Bash tool died (below). Source reverted to the
  best-validated late-par state (depth=2/2); the experiment is documented here.
- Principled fix NOT attempted: replace the held `f_scr` ObjectFifo scratch with
  an `aie.iron.Buffer` (tile-local, RMW'd, not drained — see
  `mlir_aie/python/aie/iron/buffer.py`; `Buffer(type=np.ndarray[(1024,), ...],
  name="scr")` auto-pins to the Worker's tile when passed as a fn_arg, and is
  handed to C kernels as a plain `float*`). This isolates the RMW scratch from
  the drain path and is the right primitive if the race is held-scratch-related.
  It would NOT, however, fix the 4-fifo / 2-concurrent-fill startup race, which
  is the more likely root cause. The clean fix for the startup race is probably
  to eliminate the concurrent second fill — e.g. pack the per-v-head params
  into the f_s0 stream (one param packet per v-head interleaved), dropping to
  3 fifos (1 in + 2 out) and a single fill, matching the 3-pass channel count.

**Status:** the fused kernel is **not shippable**. The 3-pass kernel remains
the bit-correct path (144 dispatches). The engine is currently wired to
`q35_gdn_fused` (the racy single-dispatch path); see (e) for the on-disk xclbin
state.

---

## (c) Execution Speed — 0.22 tok/s; >5 NOT achievable (dispatch-bound)

Measured (3-pass, pre-tooling-outage): **0.22 tok/s** (decode ≈ 4.5 s/token).

Floor analysis: each `xrt::run` dispatch costs ~30 ms of syncobj overhead.
- 3-pass: 144 dispatches/token × 30 ms ≈ 4.3 s/token floor ≈ 0.23 tok/s (matches).
- Fused (if bit-correct): 48 × 30 ms ≈ 1.44 s/token floor ≈ 0.69 tok/s.

So a working fusion would ~3× throughput (0.22 → ~0.69), but **>5 tok/s is not
reachable by fusion alone** — it requires sub-millisecond dispatch, which needs
`sudo modprobe` to tune the AMDXDNA syncobj path, and sudo is denied in this
environment. The >5 tok/s target is blocked by the driver/dispatch overhead,
independent of the kernel structure.

This matches the prior HY3 finding (`hy3-real-floor-host-attention-not-ssd`):
the floor is host attention / dispatch overhead, not the kernel compute, and
>5 tok/s needs dispatch-latency work the environment cannot do.

---

## (d) Output Text — still incoherent; NOT caused by SSM or the fusion

A clean coherent-output run could **not be completed** because the harness Bash
tool died mid-investigation (see (e)). The last observed output (from prior
runs, pre-fusion-experiment) is the same empty-assistant-turn loop:

```
\n<|im_start|>assistant<|im_end|>
```

repeated. **This is NOT coherent.** Critically, it is **not caused by SSM**
(proven bit-correct in (a)) and **not caused by the fusion race** (the 3-pass
bit-correct kernel produces the same incoherent output). The incoherence is in
GQA / FFN / MTP / lm_head / the chat template — uninvestigated. The fusion
work was therefore orthogonal to the coherence goal.

A faithful final-run output cannot be reported here because (i) the on-disk
fused xclbin is the unvalidated depth=4/1 build that crashed, and (ii) the Bash
outage prevented both recompiling a known-good xclbin and running the engine.
**Re-run after recovery** with the bit-correct 3-pass path (or a fixed fused
kernel) to capture the true output text for the record.

---

## (e) Blocker: harness Bash tool outage

Mid-investigation the Bash tool became unusable across **all** sessions (main +
subagents): every command — including `true` and `echo hello`, foreground and
background, sandboxed and `dangerouslyDisableSandbox` — returns exit 1 with no
stdout and no stderr. This is a harness-level shell-launch failure, not a
per-command or profile issue. Read/Write/Edit still work.

Consequences:
- Could not run the engine for the depth=4/1 validation or the final run.
- Could not run `scripts/cmp_fused_ab.py` or `scripts/qwopus_ssm_recur_ref.py`.
- Could not `dmesg` to distinguish OOM-kill from a kernel regression in the
  depth=4/1 crash.
- Could not recompile `fst_gdn_fused.xclbin` to restore a known-good xclbin.

**Codebase state left behind:**
- `kernels/gen_gdn_fused.py` — reverted to best-validated late-par (depth=2/2).
  Comment documents the depth=4/1 experiment and warns the on-disk xclbin is
  stale and must be recompiled before use.
- `kernels/fst_gdn_fused.xclbin` — **STALE, unvalidated depth=4/1 build** that
  crashed at prefill tok 9. **Recompile** (`python3 kernels/gen_gdn_fused.py`)
  before running the engine, or the engine will crash.
- `src/fst_engine.cpp` `gdn_scan_vheads` — wired to `q35_gdn_fused` (single
  dispatch). The 3-pass path (`gdn_passA/delta/passB`) is the bit-correct
  fallback if the fusion is abandoned.

---

## Recommended next steps (after Bash recovery)

1. **Recompile** `fst_gdn_fused.xclbin` from the reverted source and re-run the
   recur ref + `cmp_fused_ab.py` to reconfirm the vh0 race is back to 47/48
   (baseline). Do NOT run the engine with the stale on-disk xclbin.
2. **Pursue the real coherence bug** in GQA / FFN / MTP / lm_head / template —
   not SSM. The SSM wiring-bug framing is closed. Use the same dump+replay
   harness (`FST_SSM_DUMP`-style per-layer dumps for GQA layers) to localize.
3. **For the fusion**, the most promising clean fix is to drop to 3 fifos by
   interleaving per-v-head params into the f_s0 stream (single fill), or to use
   `aie.iron.Buffer` for the RMW scratch. Neither is proven. Given the fusion
   yields no progress toward coherence and only ~3× on a dispatch-bound floor
   that cannot reach >5 tok/s anyway, **the fusion is low-value relative to
   fixing the downstream coherence bug** — prioritize (2).
4. Revert `src/fst_engine.cpp` to the 3-pass dispatch if a bit-correct SSM is
   needed before the fused kernel is fixed.

---

## Summary

| target | result | reason |
|---|---|---|
| (a) SSM wiring bug | **disproven** — SSM is bit-correct | 3-pass matches HF across 13 positions; pos-0 all tensors match |
| (b) fuse 3→1 dispatch | **structural only** — not bit-correct | 144→48 dispatches wired, but vh0 first-v-head startup fill-race (4 fifos / 2 concurrent fills on this IRON build) corrupts passA when S≠0 |
| (c) >5 tok/s | **not met** (0.22 tok/s) | dispatch-bound at ~30 ms syncobj × N; >5 needs sub-ms dispatch / sudo modprobe (denied) |
| (d) coherent output | **not met** (`\n<|im_start|>assistant<|im_end|>` loop) | NOT in SSM; downstream GQA/FFN/MTP/template bug, uninvestigated |

The task's premise (an SSM wiring bug fixable by fusion → coherent + fast) was
incorrect. The fusion is orthogonal to both remaining goals. The real work is
finding the downstream incoherence bug.