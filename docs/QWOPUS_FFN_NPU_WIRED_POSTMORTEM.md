# QWOPUS NPU FFN Wired (Phase 3c) — POSTMORTEM

**Date:** 2026-07-14
**Result:** NEGATIVE on both axes. The NPU FFN is **3.26× slower** than the
shipped host FFN AND **incorrect** (diverges in prefill). The "move FFN to NPU
for speed" premise is **FALSIFIED** — the host FFN is OpenMP-parallelized
(~10.7 ms/matvec), not the 53 ms single-threaded strawman the Phase 3b probe
compared against. The NPU kernel (33 ms/matvec, compute-bound) cannot beat a
parallel CPU. **Recommendation: revert the gated NPU FFN wiring (dead — slower
+ wrong); keep the host FFN path (0.20 tok/s, proven correct).**

## What was wired (the Phase 3c diff)

Gate `FST_Q35_FFN_NPU` (runtime env-var, default OFF) replaces the 3 host
`mxfp4_matvec_f32` calls in `process_ffn_q35` with `q35_ffn_npu_matvec`
(3 MXFP4 matvecs: gate/up `[17408,5120]` + down `[5120,17408]`).

- `include/fst_engine.h`: declared `q35_ffn_npu_matvec(...)` + 3 input BOs
  (`q35_bo_ffn_in_gate_/up_/down_`, weights pre-packed once) + 1 shared output
  BO (`q35_bo_ffn_out_`) + 3 `*_packed_` flags.
- `src/fst_engine.cpp`:
  - `register_kernel("q35_ffn_npu", "fst_qwopus_ffn_insts.bin", "fst_qwopus_ffn.xclbin")`
    in the ARCH_QWEN35 block (hw_context total 7 < 9).
  - `q35_ffn_npu_matvec()`: packs `[hdr(32)|h_chunk(2048B bf16)|scales(512)|nibbles(8192)]`
    = 10784 B/pkt, NT=8 tiles × NPKT=680 pkts. Weights pre-packed once (static);
    per call only the h_chunk is re-stamped (bf16-truncate the fp32 input via
    `f2bf = u>>16`). ONE xclbin (compiled N_TILE=2176, NPKT=680) serves both
    shapes — down writes only the first 640/tile of the 2176-wide held output.
  - `process_ffn_q35`: `if (FST_Q35_FFN_NPU)` branch → 3 NPU matvecs + host
    SwiGLU (17408 elementwise ops, negligible, sits between matvecs) + residual.
    `else` branch = the proven host path (untouched).

The standalone probe (`tools/qwopus_ffn_probe.cpp`) validated the kernel on the
**gate/up shape `[17408,5120]`** with synthetic weights (sc 120-140): argmax
MATCH, 33 ms/matvec. The **down shape `[5120,17408]` was NOT probe-tested.**

## Measurement (real engine, `qwopus_bf16_mtp.fst`, prompt "What is 2+2?", 30 tok, temp 0)

| Path        | FFN / layer | / matvec | trunk (ssm+gqa+ffn)     | tok/s       | prefill-last (1st pred) | golden? |
|-------------|-------------|----------|------------------------|-------------|-------------------------|---------|
| Host (off)  | 2064 ms     | 10.7 ms  | 5074 ms (2766+244+2064)| **0.20**    | 248068                  | (ref)   |
| NPU (on)    | 6703 ms     | 34.9 ms  | 10074 ms (2970+401+6703)| **~0.10**¹ | 3280                    | DIVERGE |

¹ timed out (exit 124) at decode tok 18 (pos 32); ~18 tok / ~180 s ≈ 0.10 tok/s.

`FST_TRUNK_TIME` per-token: host `ffn=2064 ms` total / 64 layers = **32 ms/layer
= 10.7 ms/matvec** (the 3 matvecs, OpenMP). NPU `ffn=6703 ms` / 64 = **104.7
ms/layer = 34.9 ms/matvec** (3 × ~33 ms kernel + h-stamp). Per-call NPU profile:
`[q35_ffn_npu] L* 3-matvec=104 ms` (steady-state, matches).

## Finding 1 — SPEED: NPU FFN is 3.26× SLOWER than the OpenMP host (premise falsified)

The Phase 3 premise ("FFN is 40% on host = the lever to move to NPU") rested on
the Phase 3b probe's comparison: **33 ms NPU vs 53 ms scalar host → 1.6× speedup.**
That comparison was against a **strawman**: the probe's host ref
(`mxfp4_matvec_f32`) ran **single-threaded** (53 ms). The ENGINE's host path is
the SAME function but with `#pragma omp parallel for schedule(static)` over N
(`src/fst_engine.cpp:251`) → **10.7 ms/matvec across cores** (8.3 GFLOP/s
aggregate), **5× faster than the scalar strawman**.

So the real comparison is **33 ms NPU vs 10.7 ms OpenMP host → 3.26× SLOWER.**
The NPU's 33 ms is compute-bound (2.7 dequants/cycle; the 89 M-op MXFP4
dequant+matvec is compute-bound, NOT DMA-bound — per the Phase 3b finding that
"2 ms / DDR ceiling is physically impossible"). A compute-bound NPU matvec at
2.7 dequants/cycle loses to a parallel x86 at 8.3 GFLOP/s. The NPU's on-tile
bandwidth advantage is irrelevant because the kernel is NOT bandwidth-bound.

**Conclusion: the NPU cannot beat the OpenMP host FFN. Moving FFN off the CPU
does NOT help — the CPU is already doing it faster, in parallel.** This kills
the "FFN 40% lever" framing and supersedes the Phase 3b "1.6× over scalar"
result (which was real but vs the wrong baseline).

## Finding 2 — CORRECTNESS: diverges in prefill (NPU FFN output is wrong, not noisy)

- Per-prefill-token **input** ids match (17, 248045, 198) — confirms identical
  prompt, not correctness.
- `prefill-last = head_step(h after prefill trunk)`: **host=248068, NPU=3280** →
  the hidden state `h` after prefill differs → the NPU FFN diverged during
  prefill (every prefill token's FFN ran on the NPU).
- First decode token (pos 15): **host=198, NPU=192609** — unrelated tokens, not
  a borderline argmax flip. This is a **gross error**, not bfp16 precision noise
  (Fork D's "argmax must match" allowed borderline flips; 198↔192609 is not
  borderline).

Likely cause (not debugged — moot given Finding 1): the **down matvec
`[5120,17408]`** — the shape the probe never tested. The down path reuses the
gate/up xclbin with N_TILE=640 (vs 2176), reading the first 640/tile of the
2176-wide held output, KCHUNKS=17 (vs 5). A bug there (unpack indexing, the
17-chunk kc accumulation, or the N_TILE=640 mapping) would corrupt the down
output → wrong residual → h diverges from L0. Alternatively a real-weight edge
case (sc outside the probe's 120-140; sc=0 dead-block density) the synthetic
probe didn't exercise. The gate/up shape WAS probe-validated (argmax MATCH), so
if the bug is shape-specific it is in down.

Not worth debugging: even if corrected, the NPU FFN is 3.26× slower (Finding 1),
so a correct NPU FFN still loses. **The correctness bug is moot.**

## Standing-constraint check

- "No CPU math" — the NPU branch moves the 3 matvecs off the CPU (SwiGLU stays,
  17408 elementwise ops, negligible). But the GOAL of that constraint (speed via
  NPU) is unmet: the NPU is slower. The host path (kept as default) does the
  matvecs on CPU with OpenMP and is faster.
- "Do not touch proven C++ engine math" — respected: the `else` branch is the
  proven host `mxfp4_matvec_f32` path, byte-identical to before, just nested.
- Repo cleanliness — postmortem in `docs/`, logs in `logs/`, xclbins in
  `kernels/`. ✓.
- All kernels vectorial — the NPU kernel is (Phase 3b). ✓.
- The gate is OFF by default → the shipped 0.20 tok/s path is UNCHANGED (the
  host run above, gate off, reproduced 0.20 tok/s). No regression.

## Recommendation

**Revert the gated NPU FFN wiring.** It is dead on both axes (3.26× slower +
diverges). Keeping it adds dead code (helper + 3 BOs + registration + branch)
against the "prefer deletion over addition" principle. The host FFN (0.20
tok/s, OpenMP, proven correct) stays the shipped path. The kernel artifacts
(`kernels/fst_qwopus_ffn*.xclbin`) and the probe can stay (they're a valid
record of the compute-bound NPU FFN measurement; the probe's argmax-MATCH on
gate/up is still a valid de-risk of the dequant path).

## The real lever (unchanged)

The 0.20 tok/s floor is the trunk: ssm 2766 ms (54%) + ffn 2064 ms (40%) + gqa
244 ms (5%), all on host/NPU-GDN. FFN is 40% — but it's already as fast as the
NPU can make it (the NPU is slower). The ssm 54% (GDN, 144 dispatches × ~30 ms
syncobj floor) is the dominant floor, and it's a chain of dead-ends per the
memory (on-tile S, runtime chaining, delta-broadcast all dead). The only
un-tried lever is speculative decoding (M=K trunk via the chunkwise GDN — the
misty-snuggling-alpaca plan), which amortizes the dispatch floor across K
tokens rather than speeding up a single dispatch.