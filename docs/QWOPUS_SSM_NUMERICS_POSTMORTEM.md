# QWOPUS SSM Numerics — GDN fix + AIE packaging blocker (2026-07-11)

## TL;DR
- **Numerics: SOLVED and validated bit-correct vs HF.** The qwen35 "SSM" layers are
  **Gated Delta Net (GDN)**, not Mamba2. The exact decode recurrence was extracted
  from `transformers.models.qwen3_next` and a numpy/torch reference was validated
  against `torch_recurrent_gated_delta_rule` (`max|Δy|=4.66e-9`, `max|ΔS|=5.96e-8`).
- **Kernel math: written and validated bit-correct** (`kernels/fst_ssm_scan_kernel.cc`,
  pure delta rule, fp32, vectorized; twin test `max|Δy|=1.16e-9`).
- **AIE compile: blocked by tile memory.** The [128,128] fp32 state (64 KB) cannot
  be an ObjectFifo packet (depth-2 × in+out = 266 KB ≫ 64 KB tile). Next step is a
  row-streaming raw-MLIR-AIE kernel with state in DDR. This is the expected hard
  blocker (memory note: "HARD BLOCKER = SSM kernels").

## What the "SSM" layers actually are
Qwen3.5-Next `qwen35` hybrid: 48 linear-attention layers + 16 GQA + 1 MTP. The 48
"SSM" layers are **Gated Delta Net** (Yang et al.), NOT Mamba2 selective scan:

- Matrix state `S [128,128]` **per value head** (not a 1-D vector per group).
- 48 v-heads (`time_step_rank`), 16 k-heads (`group_count`); q,k `repeat_interleave` 3×.
- `head_k_dim = head_v_dim = state_size = 128`; `value_dim = 6144`; `conv_dim = 10240`.

The previous `fst_ssm_scan_kernel.cc` implemented `s = a·s + b·x; y = c·s` — a
1-D vector scan over 16 groups. Structurally wrong (no matrix state, no delta rule).

## Exact recurrence (from transformers, validated)
Per v-head, per decode token, persistent state `S [128,128]` row-major fp32:

```
qn   = l2norm(q) * (1/sqrt(128))        # q,k RAW [128]; l2norm(x,eps=1e-6)=x*rsqrt((x*x).sum()+eps)
kn   = l2norm(k)
gdec = exp(g_logit)                      # g_logit = -exp(A_log)*softplus(a + dt_bias)   < 0
beta = sigmoid(b)                        # already sigmoid'd by the caller
S    = gdec * S
kvm  = Sᵀ @ kn        = sum_i S[i,:] * kn[i]        # [128]
delta= (v - kvm) * beta                              # [128]
S    = S + outer(kn, delta)                          # S[i,j] += kn[i]*delta[j]
y    = Sᵀ @ qn        = sum_i S[i,:] * qn[i]        # [128]
```
Source: `transformers/models/qwen3_next/modeling_qwen3_next.py`
`Qwen3NextGatedDeltaNet.forward` (lines 619-718) + `torch_recurrent_gated_delta_rule`
(lines 470-512). `use_qk_l2norm_in_kernel=True`; scale `1/sqrt(head_k_dim)`.

## Files produced this session
| File | Purpose |
|------|---------|
| `scripts/qwopus_ssm_ref.py` | numpy + torch reference; unit + stability + transformers cross-check (BIT-CORRECT) |
| `kernels/fst_ssm_scan_kernel.cc` | AIE2P kernel, pure delta rule, fp32, vectorized, NO transcendentals |
| `kernels/compile_ssm_scan.py` | IRON AOT compile; GDN geometry (48 v-heads / 16 cores = 3 seq, fp32) |
| `scripts/qwopus_ssm_kernel_twin.cc` | scalar twin (no aie_api) of the kernel recurrence |
| `scripts/compare_kernel_twin.py` | builds twin, diffs vs reference → `max|Δy|=1.16e-9` |

### Division of labor (zero-CPU)
The AIE scan kernel runs ONLY the pure delta rule — **no sqrt/exp**: the AIE scalar
unit has no libc math. `l2norm`/`scale`/`exp(g_logit)`/`sigmoid(b)` are elementwise
ops done **upstream on `ew_unified`** (which already implements RMSNorm =
sum-of-squares + rsqrt + scale, and SiLU = x·sigmoid(x)). Projections (qkv, gate,
out), conv1d, SiLU, RMSNorm-gated are separate NPU GEMM/ew_unified ops. The kernel
packet is `[S_in | qn | kn | v | gdec | beta] → [S_out | y]`, all fp32 (the model
keeps `recurrent_state` in fp32, so fp32 is bit-correct-capable, not bf16-approx).

## The AIE compile blocker (concrete)
`aie_api` C++ now compiles cleanly. The IRON `Resource allocation` fails:

```
aie.tile op allocated buffers exceeded available memory
in_15_cons_buff_0  : 67080 bytes   (IN_PER_CALL = 16770 fp32 = S + qn + kn + v + gdec + beta)
in_15_cons_buff_1  : 67080 bytes   (ObjectFifo depth 2)
out_15_buff_0      : 66048 bytes   (OUT_PER_CALL = 16512 fp32 = S + y)
out_15_buff_1      : 66048 bytes
(stack)            : 1024 bytes
total ≈ 261 KB ≫ 64 KB tile data memory
```

**Root cause:** every FaStar kernel tiles its work so each ObjectFifo packet is a
small tile (the FFN passes GEMM *tiles* through depth-2 fifos, not whole matrices).
The GDN [128,128] fp32 state is inherently 64 KB and cannot be a small packet. The
recurrence also spans the whole state across two passes (decay+kvm, then outer+y),
so it can't be naively sub-tiled without DDR staging of the decayed state.

## Next step: row-streaming raw-MLIR-AIE kernel
- State `S [128,128]` lives in a **DDR BO** (in/out), never in an ObjectFifo packet.
- Stream **one row (128 fp32 = 512 B)** per ObjectFifo packet (depth-2 = 1 KB — fits).
- **Pass 1** (128 row-acquires): `S_row *= gdec`; write decayed row back to a mid
  DDR buffer; accumulate `kvm` (128 fp32) in a **core-local stack array** (NOT .bss
  — AIE2P rule: never feed `aie::load_v` from static .bss; stack locals are fine).
- Compute `delta = (v - kvm) * beta` in-core (v, kvm, delta = 512 B each).
- **Pass 2** (128 row-acquires of the decayed state): `S_row += kn[i]*delta`;
  write updated row to the out DDR BO; accumulate `y` (128 fp32) in-core.
- `qn, kn, v, gdec, beta` are small (≤ 512 B) and fit a small per-call packet.
- 48 v-heads / 16 cores = 3 v-heads/core sequential; the per-core worker runs 2×128
  row-iters per v-head. Accumulators `kvm`/`y`/`delta` persist as stack locals across
  the row loop inside one v-head (reset between v-heads).
- This needs raw MLIR-AIE orchestration (4 small ObjectFifos/core, 2 phases) rather
  than a single ExternalFunction-with-state-packet. Substantial but well-scoped.

## Honesty note
The user's Phase-2 targets (zero-CPU, 1 dispatch/layer, >15 tok/s, coherent output)
are **not yet achieved** — none are claimed. What IS done: the SSM numerics are
fixed and bit-correct, removing the scientific blocker; the kernel body is written
and math-validated. The AIE packaging + ARCH_QWEN35 wiring + 3-tier memory + the
hardware run remain multi-session work. Related: [[qwopus36-qwen35-hybrid-ssm-not-gqa]].