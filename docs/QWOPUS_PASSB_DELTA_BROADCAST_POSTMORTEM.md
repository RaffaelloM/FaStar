# passB Delta-Broadcast (Fork B / Phase 1) — POSTMORTEM

**Date:** 2026-07-13
**Verdict:** DEAD-END. Bit-identical to shipped passB but **0% speedup**. The
fork's premise (3.1 MB of replicated delta is the passB bottleneck) is
**falsified by measurement**. passB's ~34 ms/layer is structural and immovable
in the bit-identical constraint.

## What we tried

Shipped passB (`gdn_passB_block`, `kernels/fst_gdn_scan_kernel.cc`) packs the
128-float delta into EVERY input row: 48 v-heads × 128 rows × 128 fp32 = 3.1 MB
of replicated delta streamed through the shim per SSM layer, on top of 3.1 MB
S0 read + 3.1 MB S2 write = 9.4 MB total at ~280 MB/s/dir = 33.6 ms/layer
(33% of the token). The delta is identical across all 128 rows of a v-head, so
replicating it 128× is provably wasteful.

**Fork B design:** hold the per-v-head delta (128 fp32 = 512 B, well under the
8 KB fast on-tile-array threshold) via a 2nd MM2S shim, fed once per v-head;
drop delta from the per-row packet (264 → 136 fp32, `%8==0` stride). Math
byte-identical (`s2 = gdec*S + kn_i*delta`); only delta's source changes.

Artifacts: `kernels/fst_gdn_passB_delta_kernel.cc`, `kernels/gen_gdn_passB_delta.py`,
`tools/gdn_passB_delta_probe.cpp`.

## Measured results (3 variants, all bit-identical, maxdiff = 0.000e+00)

| variant | row pkt | data moved | packets | steady time |
|---|---|---|---|---|
| shipped `gdn_passB_block` | 264 (8-row) | 9.4 MB | 768 | 33.4–34.2 ms |
| delta-bcast 8-row | 136 (8-row) | 6.25 MB | 768 | 33.7 ms |
| delta-bcast 16-row | 136 (16-row) | 6.25 MB | 384 | 34.1 ms |

Probe: `tools/gdn_passB_delta_probe` loads only the 2 passB xclbins (4-xclbin
co-existence tripped the driver `qds_device::wait()` error; 2-xclbin works),
synthesizes 48 v-heads, runs both, compares S2 head-to-head + vs fp32 ref.

## Why it failed — the falsification

Two independent levers tested, both null:

1. **Data-volume lever (delta removal):** 9.4 MB → 6.25 MB (−34% data) → 0% time
   change. So passB is **not bandwidth-bound**. The 3.1 MB replicated delta was
   riding alongside the invariant S0-read + S2-write bidirectional transfer "for
   free" — removing it saved nothing.

2. **BD-count lever (16-row packets):** delta-bcast shrinks the row 264→136, so
   16 rows = 2176 words fits the 4096-word BD cap (shipped 264×16 = 4224 > 4096,
   capped at 8). That halves the packet count 768→384 → 0% time change. So
   passB is **not per-packet-BD-setup-bound** either (contrast the one-xclbin
   3-pass dead-end, where interleaved acquires WERE BD-bound — passB's flat
   packet stream already pipelines fine).

**The invariant:** S0-read (3.1 MB) + S2-write (3.1 MB) = 6.2 MB
bidirectional shim DMA at ~280 MB/s/dir. The GDN recurrence **must** read full
S and write full S every token — this is irreducible in the row-streaming
design. Both data volume and BD count were varied with zero effect → the floor
is the bidirectional S0/S2 transfer itself, structural on the AIE2P 2+2 shim.

## What this rules out / leaves

- **Bit-identical passB optimization: exhausted.** No row-packet reshape or
  delta re-sourcing moves the 34 ms. The report's "~0.24 tok/s via passB
  delta-broadcast → 33.6→18 ms" prediction is **wrong**; real passB is 34 ms,
  immovable bit-identically.
- **The only passB levers left are constraint relaxations (Fork D, sign-off):**
  - **bf16-S** (FST_Q35_GDN_BF16S, already A/B-proven lossless for argmax):
    halves S0/S2 volume to 3.1 MB total → potentially ~17 ms. NOT bit-identical.
  - **on-tile S** (chunkwise M=K): DEAD — the ≥16 KB on-tile-array wall
    (`qwopus-chunkwise-gdn-2tile-stack-deadend`).
- **passB stays 34 ms/layer = 1.61 s/token (33%)** in the shipped bit-identical
  path. This is a structural floor on AIE2P within the 2+2 shim + row-streaming
  GDN. Accept it, or relax the gate (bf16-S).

## Decision

Do NOT wire `FST_Q35_PASSB_DELTA` — it is bit-identical but gives no speedup
(no reason to ship a second kernel for 0% gain). Artifacts retained as
negative evidence. passB is no longer a viable bit-identical lever; the
remaining real lever is **Phase 2 / Fork A (NPU FFN re-stream-h)**, which
generalizes to all q35 projections (64% of the token).