# GDN Scan — Bit-Correct on NPU (Postmortem, Steps 1-2)

**Status:** Steps 1-2 of the raw-MLIR-AIE GDN task are **DONE and verified on hardware**.
Steps 3-5 (engine wiring + run + coherent output) are the remaining work — see the
end of this doc.

## What was built

A row-streaming **Gated Delta Net (GDN)** scan for the Qwen3.5-Next hybrid model, split
across **three xclbins** chained into a single `run_blob` dispatch (one `xrt::run` per
layer). The GDN state `S[128,128]` = 64 KB fp32 is too big for a 64 KB AIE2P tile as an
ObjectFifo packet, so the **state lives in a DDR BO** and is streamed ONE ROW (128 fp32
= 512 B) at a time through small depth-2 objectfifos.

| xclbin | kernel | job |
|--------|--------|-----|
| `fst_gdn_passA` | `gdn_passA_row` | 128 rows: `a=S0ᵀ@kn`, `b=S0ᵀ@qn` (one held `[a\|b]`=256 output) |
| `fst_gdn_delta` | `gdn_delta_full` | once: `kvm=gdec*a; delta=(v-kvm)*beta; c=kn·qn; y=gdec*b+delta*c` |
| `fst_gdn_passB` | `gdn_passB_block` | 8 rows/block × 16: `S2[r][j]=gdec*S0[r][j]+kn[r]*delta[j]` |

All fp32, pure delta rule (no transcendentals — `l2norm`/`exp`/`sigmoid` are computed
upstream and the kernel receives `qn, kn, gdec, beta` ready). `VEC=8` fp32 lanes,
`128/8 = 16` vectors per row. Sources: `kernels/fst_gdn_scan_kernel.cc`,
`kernels/gen_gdn_scan.py`. Probe: `tools/gdn_probe.cpp` + `scripts/gen_gdn_probe_data.py`.

## Result — bit-correct on hardware

```
[probe] NPU y vs numpy:  max|Δy|=5.821e-10
[probe] NPU S2 vs cpu_S2: max|ΔS|=1.490e-08
maxdy=5.821e-10 maxdS=1.490e-08 PASS
```

Test vector: non-zero `S0 = 0.01·randn` (seed 7, `g_logit=-1.3`, `beta=0.37`),
reference = `gdn_delta_rule` from `scripts/qwopus_ssm_ref.py` (already validated bit-correct
vs HF `transformers` recurrent: `max|Δy|=4.66e-9`). Both `y` and the full updated state
`S2` match the numpy reference to < 1e-6, with no element shift.

## How we got here — three bugs, three fixes

### Bug 1 — Two simultaneous held output fifos drain to zero
`passA` originally had **two** held output fifos `f_a`, `f_b` (acquire once, RMW in place,
release once). Both drained **all zeros** even with non-zero `S0` — masked while `S0=0`
(`y = gdec·b + delta·c` with `b = S0ᵀ@qn = 0`). The FFN GEMM — whose held `C`-matrix
pattern this mirrors and which **works** — has **one** held output. With only 2 output
DMA channels per tile, two simultaneously-held RMW outputs break the shim drain flush on
this IRON build.
**Fix:** merge `a,b` into one held `[a|b]=256` output fifo (FFN's single-held-output
pattern). `a_out` immediately became non-zero and correct.

### Bug 2 — 128-element multi-packet streaming OUTPUT drain shifts +2
`passB` (streaming output, 128 packets of 128) drained each packet **+2 elements** late
(robust across every TAP variant tried: 2D, 3D, 4D, silu-style). The 1024-element silu/mul
drain is proven bit-correct on this same IRON build.
**Fix:** restructure `passB` to output **1024-element packets** (8 rows/block, 16 blocks)
— the proven silu/mul drain geometry. This removed the +2 shift structure. (It was a red
herring that masked bug 3; both had to be fixed.)

### Bug 3 — `aie::load_v<V>` silently rounds to the nearest 8-float-aligned address  ← THE REAL ROOT CAUSE
After bugs 1-2, `passB` still produced garbage. The **const-7.0** test (kernel writes a
constant, input-independent) was **perfect** — proving the write/drain path was fine and
isolating the bug to **loads**. The `s2=delta` test then showed output = `[kn_i, gdec,
delta[0:]] = input[128:256]` — the load read from `input+128`, not `input+130`.

`aie::load_v<V>` (V=8 fp32) **requires 8-float (32-byte) alignment and silently reads from
the nearest aligned address — no fault, just wrong data.** The original packet
`[S0(128)|kn_i(1)|gdec(1)|delta(128)]=258` put `delta` at offset **130** (130%8=2) and
row stride **258** (258%8=2, misaligning every row r≥1's `S0`). `load_v(delta+130)` rounded
to 128 and read `[kn_i, gdec, delta[0:6]]`.

**Fix:** relayout the row to `[S0(128)|delta(128)|kn_i(1)|gdec(1)|pad(5)]=264` — `delta`
at offset 128 and row stride 264, **both multiples of 8**. (kn_i/gdec are scalar reads,
no alignment need.) `passA` escaped this because each row is its own aligned packet and
`kn_i`/`qn_i` are scalars.

After this one-line-layout fix: **PASS, max|Δy|=5.8e-10, max|ΔS|=1.5e-8.**

### The diagnostic that cracked it (reusable)
> Always test a **constant-output** kernel (write 7.0, read no input) to isolate load
> corruption from drain bugs. The const test being perfect proved the write/drain path was
> fine; then `s2=delta` (load one input field) showed the load rounding. The trivial
> `s2=S0` copy looked "bit-correct" for row 0 but `per-row shift=0 max=0.49` exposed the
> later-row corruption that row 0's aligned start had hidden.

## AIE2P rules now confirmed (two, both load-bearing for future kernels)

1. **Never feed `aie::load_v` from C++ static `.bss`** — reads garbage (existing rule, see
   `docs/HY3_FUSED_FFN_POSTMORTEM.md`).
2. **Every `aie::load_v`/`store_v` address must be a multiple of 8 floats** — both the
   field offset within a packet AND the row stride. Misaligned loads silently round and
   return wrong data (new rule, this work).

## Build / run the probe

```bash
python3 kernels/gen_gdn_scan.py            # -> kernels/fst_gdn_{passA,delta,passB}.xclbin + _insts.bin
python3 scripts/gen_gdn_probe_data.py gdn_probe_data.bin && cp gdn_probe_data.bin kernels/
FFLM=$(pwd)/build/_deps/fastflowlm-src/src/include
g++ -std=c++17 -O2 -I/usr/include -I$FFLM -Iinclude tools/gdn_probe.cpp -o tools/gdn_probe \
    -L/usr/lib/x86_64-linux-gnu -lxrt_coreutil -laiebu -luuid -lpthread
XILINX_XRT=/usr ./tools/gdn_probe kernels gdn_probe_data.bin   # -> PASS
```

## Steps 3-5 (remaining)

- **Step 3 — engine wiring.** Add `ARCH_QWEN35` to `enum Arch` (`include/fst_engine.h:143`),
  a per-layer weight struct mirroring `Hy3LayerWeights` (`:338`) loaded from the
  `TID_Q35_*` tensors (`scripts/fst_converter.py:192-214`), `process_ssm_layer()` that:
  (a) host-SIMD qkv/gate projections (reuse `host_gemm_*` from `fst_engine.cpp:123/159/185`),
  (b) conv1d + `A_log`/`alpha`/`beta` → per-head `g_logit`, `beta` + `l2norm` → `qn,kn`,
  (c) the bit-correct GDN scan via `run_blob` (state BO in Tier-1 RAM; build the 130/642/
  264 packets and dispatch `fst_gdn_passA`→`fst_gdn_delta`→`fst_gdn_passB`),
  (d) group norm + output projection. Then the GQA layers (L%4==3) reuse the HY3 SIMD
  host-attention path, and the MTP block (blk.64) reuses HY3 NextN. `generate_qwen35`
  greedy loop + `fst_main.cpp` dispatch hook + xclbin registration at init
  (`fst_engine.cpp:1604`).
- **Step 4 — build + run.** `cmake --build build -j$(nproc)`; `timeout 120 systemd-run
  --user --scope -p MemoryMax=35G ./build/ds4_npu_engine --model qwopus.fst --prompt
  "Hello, what are you?" --tokens 10 --temp 0.6 2>&1 | tee logs/run_gdn.log`.
- **Step 5 — final postmortem** (compile/tile-fit, hw bit-correct, engine wiring for 48 SSM
  layers, decode tok/s, exact output text + coherence).

The hard, previously-blocking milestone — a bit-correct NPU GDN scan with state in DDR —
is **done**. The remaining work is host-side wiring of the surrounding SSM block and the
generate loop, which is substantial but no longer blocked by an NPU correctness unknown.