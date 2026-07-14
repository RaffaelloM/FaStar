# QWOPUS passB V-Head-Split — MEASURED POSTMORTEM

**Date:** 2026-07-14
**Status:** Step 1 of the "do not accept 0.20 tok/s ceiling" re-investigation.
**Verdict:** Multi-tile v-head-split passB is **MEASURED SLOWER** than the shipped
1-tile passB. The "8× shim bandwidth ⇒ <5 ms" thesis is **refuted by data**.

## The question
The shipped passB (`gen_gdn_scan.py::gdn_passB_op`) runs all 48 v-heads on ONE
tile / ONE worker / ONE shim pair (1 MM2S + 1 S2MM). It is DMA-bound on the
single shim's S0-read + S2-write. The earlier delta-broadcast fork
(`gen_gdn_passB_delta.py`) was **also 1-tile/1-shim** — it eliminated the 3.1 MB
replicated delta but the S0/S2 floor stayed, so it gave 0% gain. **Multi-tile
parallelism of the S0/S2 stream itself was never tested.** The hypothesis:
split the 48 v-heads across N tiles (6 v-heads/tile at N=8), each on its own
shim columns ⇒ N× aggregate DMA bandwidth ⇒ passB 33 ms → <5 ms.

## The measurement (same BO, same data, same machine, best of 5)
`tools/gdn_passB_vsplit_probe.cpp` packs one in-BO (768 packets, v-head-major /
block-major — identical global order for all variants) and runs the shipped
1-tile, 4-tile, and 8-tile kernels on it. Math is byte-identical
(`gdn_passB_block` reused verbatim — same 264-stride row packet, same 8-rows/
block row-streaming so S0 never sits on-tile as a 64 KB array).

```
variant           tiles  latency      S2 vs shipped S_post
1-tile SHIPPED      1    33.06 ms     max|Δ|=7.4e-9  (bit-identical)
4-tile v-split      4   263.86 ms     bit-identical
8-tile v-split      8    66.06 ms     bit-identical
cross-variant 4t-vs-8t max|Δ|=0.0 ; 1t-vs-8t max|Δ|=0.0   (all byte-identical)
```

passB moves ~9.5 MB (in-BO 6.5 MB + out-BO 3.1 MB). 1-tile ⇒ 9.5 MB / 33 ms =
**280 MB/s = full single-shim bandwidth** (passB already saturates one shim).
8-tile ⇒ 9.5 MB / 66 ms = **144 MB/s aggregate — less than one shim**.

## Why parallelism makes it worse
Concurrent multi-shim DDR access **contends**: 8 shims hitting DDR simultaneously
deliver *lower* aggregate bandwidth than one streaming shim. The "N× bandwidth"
assumption is false on NPU2 — parallelism *divides* bandwidth via DDR-controller
contention (and per-tile/per-stream setup), rather than multiplying it. The
non-monotonicity (4-tile 264 ms ≫ 8-tile 66 ms) confirms it is placement/
contention-driven, not a simple bandwidth-scaling regime.

This also fully explains the delta-broadcast 0%-gain: passB was already at the
single-shim streaming limit; neither removing the 3.1 MB replicated delta nor
multi-tiling helps, because the limit is shim/DDR *streaming efficiency*, not the
replicated-delta *volume*. The prior "structural on AIE2P 2+2 shim" framing was
right in conclusion but vague in mechanism — the measured mechanism is
single-shim saturation + multi-shim DDR contention.

## What this does / does not refute
- **Refutes:** 8-tile (or 4-tile) v-head-split passB as a speedup. passB's 33 ms
  is the floor; it cannot be dropped below ~33 ms by v-head parallelism.
- **Does not refute:** column-split passB (would hold S columns on-tile ⇒ the
  ≥16 KB slow regime, the slab's disease — not viable for row-streamed passB).
  v-head-split was the natural parallelization for self-contained v-heads.
- **Does not touch:** the slab M=K kernel (Step 2 — the recurrence math uses
  scalar-broadcast MAC, NOT `aie::mmul`; that is the live untested lever).

## Artifacts
`kernels/gen_gdn_passB_vsplit.py` (NTILE env), `kernels/fst_gdn_passB_vsplit_{4,8}t.
{cc-source is the reused fst_gdn_scan_kernel.cc::gdn_passB_block}.xclbin`,
`tools/gdn_passB_vsplit_probe.cpp`, `logs/gdn_passB_vsplit_ab.log`.

## Next
Step 2: the slab `recur_core` computes `a[c]=Σ_i S[i,c]·kn[i]` as 128
scalar-broadcast `aie::mul(sv_f, kn_i)` MACs (kn_i a scalar) — it does NOT use the
AIE2P hardware matrix multiplier (`aie::mmul`). That is the genuinely untested
lever; rewrite to `mmul` and re-measure the slab.