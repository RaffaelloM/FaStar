# FaStar — High-Performance C++ Inference Engine for DeepSeek V4 Flash on AMD Ryzen AI NPU

**FaStar runs DeepSeek V4 Flash — a 284-billion-parameter Mixture-of-Experts model — on a
laptop.** It executes the full inference path on the AMD Ryzen AI 9 365's integrated XDNA2
NPU, spilling the ~150 GB of expert weights across **SSD → RAM → NPU scratch buffers** using
**expert virtual memory**: MoE experts are treated like virtual-memory pages — stored on an
NVMe SSD, cached in RAM, and uploaded to NPU scratch buffers on demand.

The engine is mathematically faithful to the HuggingFace reference (coherent English output;
greedy prefill argmax matches the HF ground truth) and fits the 150 GB model into a 64 GB RAM
budget via a paged expert cache.

> ⚠️ This is a research/engineering showcase, not a production server. Decode runs at
> **~0.05 tokens/sec** (see [Known limitations](#known-limitations)). The project's value is
> the architecture — expert virtual memory + on-NPU MLA/FFN via IRON-generated kernels — and a
> working end-to-end 284B inference path on consumer NPU hardware.
>
> 🆕 **Now also runs Tencent Hunyuan-3.0 (HY3)** — a 299B GQA + sigmoid-router + NextN-MTP
> Mixture-of-Experts (~17B active), the opposite attention family from DeepSeek V4 Flash's MLA.
> Decode is now **~0.14 tok/s** (OpenMP + AVX2/FMA-SIMD host attention). See the
> [HY3 section](#hunyuan-30-hy3) below and the [HY3 model card](HY3_MODEL_CARD.md).

---

## Features

- **100% NPU execution.** MLA attention, expert FFN (fused dequant + GEMM), the MoE router,
  RMSNorm/SiLU/RoPE/softmax, and the LM head are all IRON-generated MLIR-AIE kernels running on
  the XDNA2 NPU. The CPU only drives dispatch, routing host code and KV-cache bookkeeping.
- **DSpark speculative decoding.** A small draft model proposes tokens that the main model
  verifies in a block, amortizing the prefill cost (`--draft_model`).
- **SSD expert paging (expert virtual memory).** 1376 experts live on SSD in a page-aligned
  `.fst` container; an LRU RAM cache with predictive prefetch feeds a persistent host-only BO
  pool, so hot experts skip the SSD round-trip.
- **Built-in Web UI.** A lightweight HTTP chat server (`--serve`) streams tokens back to the
  browser as Server-Sent Events, with a persistent multi-turn KV cache.
- **Zero-vendor build.** No `third_party/` folder. Header-only dependencies
  ([cpp-httplib](https://github.com/yhirose/cpp-httplib), [nlohmann/json](https://github.com/nlohmann/json),
  and the [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) NPU instruction-sequence headers)
  are fetched automatically by CMake `FetchContent` on first configure.
- **Auto model download.** On first run, if the model weights are missing, FaStar fetches them
  from HuggingFace (resumable) so a fresh clone is runnable with no manual setup.

---

## Hunyuan-3.0 (HY3)

FaStar has been extended to run **Tencent Hunyuan-3.0** (`hy_v3`) — a 299B-parameter
Mixture-of-Experts (~17B active per token) using **GQA + sigmoid router + NextN MTP**, the
opposite attention family from DeepSeek V4 Flash's MLA. The same expert-virtual-memory
architecture (SSD → RAM → NPU) runs it end-to-end on the Ryzen AI 9 365 NPU, with **zero
mandatory new NPU kernels** — HY3 reuses the proven dequant/GEMM/`ew_unified`/router/`lm_head`
xclbins (only the expert block size differs, read from the header) and **drops `mla_unified`**
(GQA needs no latent compression), freeing ≥3 of the 9-hw_context cap.

### HY3 architecture (from the GGUF metadata)

| Hyperparameter | Value |
|---|---|
| Blocks | 81 (blk.0 dense L0 · blk.1–79 MoE · blk.80 NextN MTP) |
| Hidden | 4096 |
| Attention | GQA, 64 Q heads / 8 KV heads, head_dim 128, per-head Q/K RMSNorm |
| Experts | 192 routed, top-8, + 1 always-on shared expert |
| Expert / dense inter | 1536 / 13312 (L0) |
| Router | sigmoid + per-expert bias (bias added to selection only); weight ÷ Σ × 2.826 |
| RoPE | YaRN NeoX (base 11158840, factor 4.0, orig 262144 → 1 M context) |
| Vocab | 120,832 (weight-tied LM head) |
| `.fst` size | 173.82 GB (15,360 MXFP4 expert blocks, 10,027,008 B each, page-aligned) |

### Running HY3

```bash
export XILINX_XRT=/usr
# hy3.fst + tokenizer.json come from the HY3 HuggingFace repo (see HY3_MODEL_CARD.md)
./build/ds4_npu_engine --model hy3.fst --prompt "Hello" --tokens 32 --temp 0.0
#   -> "Hello! How can I help you today ..."
```

The HY3 path is selected automatically from the `.fst` header (`qh=64, kh=8, hd2=128`).
Speculative decoding uses the **NextN MTP head** (block 80, shares the trunk embedding +
LM head) rather than the separate DSpark draft model used for DeepSeek V4 Flash. The
on-device expert FFN runs a fused 4-dispatch/layer path (`FST_HY3_FUSED_FFN`:
dequant-8-experts + multicore gate/up/down GEMM + host SiLU/mul).

### HY3 performance (measured on Ryzen AI 9 365)

| | |
|---|---|
| Decode (coherent, M=16) | **~0.12 tok/s** |
| Decode (raw M=1) | **~0.14 tok/s** |
| Prefill (M=16 templated) | ~185 s |
| Decode layer floor | ~89 ms (SSD ~50 ms · NPU FFN ~37 ms · host attn ~7 ms) |

The dominant HY3 decode cost is **host GQA attention** (the 4 projection matvecs, ~55% of
the layer at baseline). FaStar now parallelizes them with **OpenMP** (row-parallel,
bit-identical) + **AVX2/FMA SIMD** (8-wide bf16→fp32 FMA inner loop, tree-reduce — not
bit-identical, verified empirically to preserve the greedy argmax). This lifted decode
from 0.07 → **0.14 tok/s raw (+100%) / 0.12 tok/s coherent (+71%)**; host attention is no
longer the floor and SSD expert loading is back to the largest share.

### HY3 model + model card

The converted `hy3.fst` (173.82 GB) + `tokenizer.json` are self-contained — unlike the
DeepSeek-V4-Flash-DSpark set, HY3 needs **no sidecar files** (its GQA + sigmoid router use
no MLA/HC or hash-routing sidecars; RMSNorm weights live in the `.fst` shared bank). See
**[HY3_MODEL_CARD.md](HY3_MODEL_CARD.md)** for the HuggingFace model card (Apache-2.0,
inherited from `tencent/Hy3` via the `satgeze/Hy3-1M-GGUF` Q3_K_M source). Conversion is
reproducible: `scripts/hy3_download.py` → `scripts/fst_converter.py hy3` → `scripts/verify_fst.py`.

Full build notes (the converter, the HY3 engine path, the fused-FFN dispatch collapse,
and the levers-1+3 / OpenMP / SIMD measurements) are in `HY3_PLAN.md`,
`HY3_PIVOT_POSTMORTEM.md`, and `HY3_FUSED_FFN_POSTMORTEM.md`.

---

## Hardware requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| APU | AMD Ryzen AI 9 365 (XDNA2 NPU) | The NPU is the compute target. |
| RAM | **64 GB** minimum | The 150 GB model is paged; ~54 GB peak RSS observed. |
| Storage | **NVMe SSD**, ~200 GB free | Experts live on SSD; read latency dominates miss cost. |
| GPU | Not required | The iGPU is unused; all inference is CPU (router/KV) + NPU. |

## Software requirements

Target platform: **Ubuntu 24.04** (any modern Linux with the packages below works).

| Dependency | Purpose | Install |
|------------|---------|---------|
| **XRT** (Xilinx Runtime) | NPU device + buffer API | `apt install xrt` (CMake config at `/usr/share/cmake/XRT`, libs in `/usr/lib`) |
| **AIEBU** | Assembles NPU instruction blobs → ELF | Ryzen AI SW stack (CMake config at `/usr/share/cmake/AIEBU`) |
| **AMDXDNA driver** | Kernel module for the NPU | `lsmod \| grep amdxdna` (Ryzen AI driver package) |
| **CMake ≥ 3.16** | Build | `apt install cmake` |
| **g++ (C++17)** | Compiler | `apt install build-essential` |
| **Python 3 + `tokenizers`** | HF BPE tokenizer bridge | `pip install tokenizers` (used by `scripts/fst_tokenize.py`) |
| **wget** (or curl) | Model auto-download | `apt install wget` |
| Git | FetchContent clones the header deps | `apt install git` |
| **IRON / MLIR-AIE** *(kernel rebuild only)* | Regenerate `.xclbin` kernels | Only needed to recompile kernels; prebuilt kernels ship in `kernels/`. |

cpp-httplib, nlohmann/json, and the FastFlowLM headers are downloaded by CMake — no manual
install of those is needed.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/<you>/FaStar.git
cd FaStar

# 2. Build (CMake fetches cpp-httplib + nlohmann/json + FastFlowLM headers on first configure)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
#    -> build/ds4_npu_engine

# 3. Run (XRT lives in /usr on Ubuntu, not /opt/xilinx)
export XILINX_XRT=/usr

#    First run with no model present auto-downloads the ~163 GB model set from HuggingFace
#    (resumable — re-run to continue a partial fetch):
./build/ds4_npu_engine --serve --port 8080
#    then open http://localhost:8080/ in a browser.
```

If you already have the `.fst` model files, place them in the current directory (or pass
`--model-dir <dir>`) and the download step is skipped.

> **Model availability.** FaStar downloads from
> [`RaffaelloMolinari/Deepseek-V4-Flash-DSpark-FST`](https://huggingface.co/RaffaelloMolinari/Deepseek-V4-Flash-DSpark-FST)
> on first run. Ensure the `.fst` files, their `.fst.hc` / `.fst.norm` / `.fst.tid2eid`
> sidecars, `tokenizer.json`, and (for speculative decoding) `dspark_draft.fst` are present in
> that HuggingFace repo. The `.fst.norm` and `.fst.hc` sidecars are required for coherent
> output (the bare `.fst` alone has 40×-too-small RMSNorm weights).

---

## Usage

```bash
export XILINX_XRT=/usr

# One-shot generation (greedy, deterministic):
./build/ds4_npu_engine --model deepseek_v4_dspark.fst \
                       --prompt "Explain quantum computing" --tokens 128 --temp 0.0

# With speculative decoding (loads the DSpark draft model):
./build/ds4_npu_engine --model deepseek_v4_dspark.fst --draft_model dspark_draft.fst \
                       --prompt "Hello world" --tokens 256

# Interactive multi-turn (persistent KV cache across turns):
./build/ds4_npu_engine --model deepseek_v4_dspark.fst --interactive --tokens 128

# Web UI / chat server:
./build/ds4_npu_engine --model deepseek_v4_dspark.fst --serve --port 8080
```

### CLI flags

| Flag | Env var | Purpose |
|------|---------|---------|
| `--model <path>` | — | `.fst` model file (default: `deepseek_v4_dspark.fst`). |
| `--model-dir <path>` | — | Directory holding the models + tokenizer (auto-download target; default: CWD). |
| `--draft_model <path>` | — | DSpark draft `.fst` for speculative decoding. |
| `--prompt <text>` | — | Prompt text (one-shot / interactive modes). |
| `--tokens <n>` | — | Max tokens to generate (default 128). |
| `--temp <f>` | — | Sampling temperature (default 0.7; `≤0` = greedy/argmax). |
| `--top_p <f>` | — | Top-p nucleus (default 0.9). |
| `--tokenizer <path>` | `FST_TOKENIZER` | `tokenizer.json` path. |
| `--tokenize_script <p>` | `FST_TOKENIZE_SCRIPT` | `fst_tokenize.py` path. |
| `--kernel_dir <path>` | `FST_KERNEL_DIR` | Directory holding `.xclbin` / `_insts.bin` (default `./kernels`). |
| `--interactive` | — | Multi-turn API with persistent KV cache. |
| `--serve` | — | Start the HTTP web UI / chat server. |
| `--port <n>` | — | Server port (default 8080). |
| `--web-dir <path>` | — | Directory holding `index.html` (default `./web`). |
| `--no-sd` | — | Disable speculative decoding (plain autoregressive decode). |
| `--skip-prefill` | — | Continue from current KV state (skip prefill). |

---

## Web UI (chat server)

Start the built-in HTTP server + chat UI with `--serve`:

```bash
export XILINX_XRT=/usr
./build/ds4_npu_engine --model deepseek_v4_dspark.fst --serve --port 8080
```

Then open <http://localhost:8080/>.

- **Send a message** → the server tokenizes it with the DeepSeek chat template, runs
  `prefill_user` (append-aware prefill) and streams reply tokens back as **Server-Sent Events**
  (`text/event-stream`); tokens appear in real time.
- **Multi-turn**: the KV cache and V4 compressor state persist across messages, so follow-up
  questions reuse the conversation context without re-prefilling history.
- **New chat** → `POST /reset` clears the KV cache and starts a fresh session.

| Method | Path | Body / purpose |
|--------|------|----------------|
| `GET`  | `/`        | Serves `web/index.html`. |
| `POST` | `/generate`| JSON `{prompt, tokens?, temp?, top_p?, reset?}` → SSE stream of `{id, text, done}`. |
| `POST` | `/reset`   | Clear KV cache + compressor (start a new chat). |

Generation requests are serialized (the NPU is single-instance), so concurrent `/generate`
calls queue. The first token of a turn takes ~2 minutes (prefill of a 43-layer 284B model on
this NPU); subsequent tokens stream at ~0.05 tok/s.

---

## Architecture overview

FaStar splits execution across three tiers:

```
   SSD (.fst)  ──page→  RAM (ExpertPager LRU)  ──upload→  NPU scratch (persistent host_only BOs)
```

### Expert virtual memory (`ExpertPager`)
- Expert block size ≈ 16 MB (13.4 MB payload + alignment), stored as dense 17-byte MXFP4
  blocks (1 e8m0 scale + 16 FP4 nibbles).
- An LRU RAM cache (6 GB staging) with a background predictive-prefetch thread:
  `predict_and_prefetch()` reuses layer *L*'s experts at *L+1*;
  `predict_and_prefetch_from_draft()` runs the draft router to predict exact experts.
- The **authoritative** cache is a persistent host-only BO pool (`FSTEngine::get_expert_bo`)
  holding hot experts device-readable, checked before the pager on every dispatch.

### NPU execution (`AiebuKernelCache`)
- Each xclbin gets a **permanent** `hw_context`. The AMDXDNA driver caps simultaneous
  hw_contexts at **9**; unified xclbins keep the engine under this.
- Kernels register once at startup; per-dispatch instruction blobs run on the existing
  contexts. An in-process `NpuSequenceBuilder` / `run_blob` path (ported from FastFlowLM's
  `npu_sequence`) builds dynamic multi-op blobs at runtime — the foundation for future kernel
  fusion (many DMA micro-ops in one host→NPU submission).

### MLA (Multi-head Latent Attention)
- Q latent 1024, KV latent 512. Q/K/V compression projections run on the NPU.
- The KV cache stores compressed latents + positional embeddings, pre-sized to `max_seq` and
  written at `seq_pos * KV_LORA`.
- A V4 KV-compressor streams hidden through per-layer state and emits 512-dim compressed KV
  rows at ratio boundaries.

### DSpark speculative decoding
- A small draft model proposes tokens; the main model verifies them in a block.
- `--draft_model dspark_draft.fst` enables it; `--no-sd` forces plain autoregressive decode.

### Interactive / multi-turn API
`FSTEngine` exposes `reset_session()`, `prefill_user(ids, temp, top_p)` and
`decode_step(prev_tid, temp, top_p)`. `prefill_user` **appends** at the current sequence
position (resetting the compressor only on the first turn), so a multi-turn chat reuses the KV
cache across turns. The CLI (`--interactive`) and the web server (`--serve`) both build on it.

---

## Project structure

```
FaStar/
├── src/                  C++ engine sources
│   ├── fst_engine.cpp      Inference orchestrator: layers, NPU dispatch, MLA/FFN/SD
│   ├── fst_main.cpp        CLI + interactive loop + HTTP web server + HF auto-download
│   └── expert_pager.cpp    Expert virtual memory: SSD→RAM LRU cache + prefetch
├── include/              Headers (fst_engine.h, expert_pager.h, fst_aiebu_cache.hpp)
├── kernels/              NPU kernels: IRON compile scripts + kernel .cc sources
│                         + compiled .xclbin / _insts.bin (prebuilt, committed)
├── scripts/              Python: model converter, tokenizer bridge, verify/bench
├── tools/                Standalone C++ probes (insts decoder, packing probes, etc.)
├── web/                  Chat UI (index.html, served by --serve)
├── CMakeLists.txt        Build (finds XRT + AIEBU; FetchContent for the 3 header deps)
├── xrt.ini               XRT runtime config (verbosity / debug flags)
└── README.md
```

No `third_party/` folder is committed. Header-only dependencies are fetched at configure time
into `build/_deps/` (gitignored).

---

## Model format & conversion

Convert a HuggingFace DeepSeek checkpoint to the `.fst` container:

```bash
python3 scripts/fst_converter.py --model deepseek-ai/DeepSeek-V4-Flash-DSpark \
                                 --output deepseek_v4_dspark.fst
```

The `.fst` format stores a page-aligned config header, shared tensors (attention, router,
norms) in Q8_0 / BF16, and expert blocks in dense DS4 MXFP4. Verify integrity:

```bash
python3 scripts/verify_fst.py deepseek_v4_dspark.fst
python3 scripts/check_fst.py  deepseek_v4_dspark.fst
```

---

## NPU kernel compilation

Prebuilt kernels ship in `kernels/` (`*.xclbin` + `*_insts.bin`). You only need to recompile if
you change a kernel. Compilation uses AMD **IRON** (MLIR-AIE):

```bash
export PATH="$HOME/.local/bin:$PATH"
export PEANO_INSTALL_DIR="$HOME/.local/lib/python3.14/site-packages/llvm-aie"

# Example: rebuild the LM head kernel
python3 kernels/compile_lm_head.py
#    -> kernels/fst_lm_head.xclbin + kernels/fst_lm_head_insts.bin
```

The "unified" pipeline is the current set the engine loads:

| Script | Kernel | Purpose |
|--------|--------|---------|
| `compile_ew_unified.py` | `fst_ew_unified.xclbin` | RMSNorm / SiLU / mul / softmax / RoPE / router |
| `compile_ffn_unified.py` | `fst_ffn_unified.xclbin` | Expert FFN (fused dequant + GEMM) |
| `compile_mla_unified.py` | `fst_mla_unified.xclbin` | Unified MLA attention |
| `compile_dequant_q4k.py` | `fst_dequant_q4k.xclbin` | MXFP4→BF16 dequantization |
| `compile_lm_head.py` | `fst_lm_head.xclbin` | LM head projection |
| `compile_router.py` | `fst_router.xclbin` | MoE router (sqrtsoftmax + top-k) |

---

## Known limitations

- **Throughput: ~0.05 tokens/sec.** Decode is NPU-compute-bound per dispatch. Dispatch *count*
  reduction (op-replication packing) was proven correct on silicon but does not move tok/s — a
  packed N-copy blob does N× the NPU compute. Reaching >1 tok/s requires IRON-level **fusion**
  (fewer GEMM round-trips, on-device K-accumulation, tiled reuse), which is future work.
- **9 hw_context cap.** The AMDXDNA driver limits simultaneous hw_contexts to 9; the engine
  uses unified xclbins to fit. Adding a new kernel xclbin may require retiring another.
- **150 GB model on 64 GB RAM.** Works via expert paging, but the first prefill is
  SSD-miss-dominated (~2 min) and expert cache thrash drops throughput if the working set
  exceeds RAM.
- **Single instance.** A file lock (`/tmp/fastar_npu.lock`) prevents two processes from
  fighting over NPU contexts.
- **Hard exit.** The process calls `_exit()` on completion to avoid a known AMDXDMA hang when
  tearing down many BOs/contexts in destructors.

---

## Troubleshooting

**NPU DMA deadlock (syncobj timeout at Layer 0):** mixing kernels from different xclbins on one
hw_context. Each xclbin gets its own hw_context (handled by `AiebuKernelCache`).

**XRT device init failure:** ensure `XILINX_XRT=/usr`, `lsmod | grep amdxdna` shows the driver,
and no other FaStar instance holds `/tmp/fastar_npu.lock`.

**Expert cache thrash:** monitor the hit rate in the run log; if <90%, raise the RAM cache or
improve the prefetch strategy.

**IRON compile failures:** `import aie.iron` needs a specific LLVM-AIE install; re-run the
PEANO installer and check `PEANO_INSTALL_DIR`. (Only needed to recompile kernels.)

**Model download fails / 404:** the `.fst` files must be present in the HuggingFace repo
`RaffaelloMolinari/Deepseek-V4-Flash-DSpark-FST`. Downloads are resumable — re-run to continue a
partial fetch. To use a locally-converted model instead, place the `.fst` (+ sidecars) in the
working directory or pass `--model-dir`.

---

## Acknowledgements

FaStar builds directly on the ideas and tooling of several open projects:

- **[antirez/dwarfstar](https://github.com/antirez/dwarfstar)** (`ds4`) — the DeepSeek-V4
  MXFP4 quantization format and the single-file reference architecture that FaStar's engine
  structure and faithful numerics are measured against.
- **[amd/IRON](https://github.com/amd/iron)** and **[Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)**
  — the MLIR-AIE compiler toolchain used to generate every NPU kernel (`compile_*.py`).
- **[FastFlowLM](https://github.com/FastFlowLM/FastFlowLM)** — the AIEBU dispatch pattern and
  `npu_sequence` runtime that FaStar's in-process `NpuSequenceBuilder`/`run_blob` path is
  ported from; its `npu_utils` headers are a build-time dependency.
- **[DeepSeek-AI](https://github.com/deepseek-ai)** — the DSpark speculative-decoding design and
  the DeepSeek V4 Flash model.
- **[yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib)** and
  **[nlohmann/json](https://github.com/nlohmann/json)** — the header-only HTTP server and JSON
  library powering the web UI.

## License

FaStar is provided as-is for research and educational use. Bundled header dependencies retain
their respective licenses (see each upstream project).