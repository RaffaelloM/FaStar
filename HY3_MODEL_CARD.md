---
license: apache-2.0
base_model:
  - tencent/Hy3
  - satgeze/Hy3-1M-GGUF
tags:
  - mxfp4
  - npu
  - amd-xdna
  - ryzen-ai
  - moe
  - hunyuan
  - faStar
  - expert-virtual-memory
pipeline_tag: text-generation
language:
  - en
  - zh
inference: false
---

# Hunyuan-3.0 (`hy_v3`) — FaStar MXFP4 NPU container

This repository repackages **Tencent Hunyuan-3.0** (`hy_v3`, a 299B-parameter
Mixture-of-Experts, ~17B active per token) into FaStar's page-aligned `.fst`
container so it can run end-to-end on the **AMD Ryzen AI 9 365 XDNA2 NPU** via
**expert virtual memory**: the ~174 GB of expert weights live on an NVMe SSD,
are cached in RAM, and are uploaded to NPU scratch buffers on demand — treated
like virtual-memory pages.

> ⚠️ **This is not a standalone model.** It requires the [FaStar inference
> engine](https://github.com/RaffaelloM/FaStar) (built from source) to run. The
> `.fst` format is a custom container, not GGUF/safetensors — the engine
> dequantizes experts on-NPU. There is no `transformers` / `llama.cpp` load path.

## Files

| File | Size | Contents |
|------|------|----------|
| `hy3.fst` | 173.82 GB | FaStar container: 128-byte `FSTH` header → BF16/F32 shared-bank tensors → 15,360 MXFP4 expert blocks |
| `tokenizer.json` | 9.5 MB | HF BPE tokenizer (120,000 vocab + 818 added tokens) from `tencent/Hy3` |

**No sidecar files** (unlike the DeepSeek-V4-Flash-DSpark-FST repo). HY3's GQA
attention + sigmoid router need no MLA/Hybrid-Connection or hash-routing
sidecars; all RMSNorm weights are stored in the `.fst` shared bank. `hy3.fst` +
`tokenizer.json` is the complete runtime set.

## Architecture (parsed from the live GGUF metadata)

| Hyperparameter | Value |
|---|---|
| Architecture | `hy_v3` (GQA + sigmoid router + NextN MTP) |
| Blocks | **81** (blk.0 dense L0 · blk.1–79 MoE · blk.80 NextN MTP) |
| Hidden | 4096 |
| Q / KV heads | 64 / 8 (GQA 8:1, head_dim 128) |
| Experts / top-k | 192 routed · top-8 · + 1 always-on shared expert |
| Expert inter dim | 1536 (dense L0 inter 13312) |
| Router | sigmoid + per-expert bias (bias added to selection only); weight ÷ Σ × 2.826 |
| Activation / norm | SiLU · RMSNorm (ε = 1e-5) |
| RoPE | YaRN NeoX rotate-half (base 11158840, factor 4.0, orig 262144 → 1 M context) |
| Vocab | 120,832 (weight-tied LM head) |
| Total / active params | ~299 B / ~17 B |

**Expert geometry (byte-exact):** gate/up `[1536, 4096]`, down `[4096, 1536]`.
Each expert block = gate+up+down = **10,027,008 B**, exactly page-aligned
(`10,027,008 = 2448 × 4096`). Each 17-byte sub-block = 1 e8m0 scale + 16 FP4
nibbles — the MXFP4 dense layout the on-NPU dequant+GEMM kernel consumes.

## Quantization

- **Source:** `hy3-1M-MTP-Q3_K_M.gguf` from
  [`satgeze/Hy3-1M-GGUF`](https://huggingface.co/satgeze/Hy3-1M-GGUF)
  (Apache-2.0, derived from [`tencent/Hy3`](https://huggingface.co/tencent/Hy3)).
- **Target:** MXFP4 experts (4-bit + 8-bit block scale) + BF16/F32 shared
  tensors (attention projections, router, norms, dense L0, shared expert, MTP).
- **Caveat — double quantization:** Q3_K_M → BF16 → MXFP4. This is faithful
  enough for coherent generation (verified) but is **not** a bit-exact-weights
  reproduction of the original fp16 checkpoint. For a faithfulness study, start
  from BF16 safetensors instead.

## Hardware & software

| | |
|---|---|
| APU | AMD Ryzen AI 9 365 (XDNA2 NPU2, Zen 5 cores) |
| RAM | 64 GB minimum (experts are paged) |
| Storage | NVMe SSD, ~200 GB free (experts live on SSD; read latency dominates miss cost) |
| GPU | not used |
| Runtime | Ubuntu 24.04 · XRT (Xilinx Runtime) · AIEBU · `amdxdna` driver |
| Engine | [FaStar](https://github.com/RaffaelloM/FaStar), built from source |

## How to run

```bash
# 1. Build the FaStar engine (CMake fetches header deps on first configure)
git clone https://github.com/RaffaelloM/FaStar.git
cd FaStar
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)          # -> build/ds4_npu_engine

# 2. Fetch this model set (hy3.fst + tokenizer.json) into a model dir
#    (or pass --model-dir <dir> where you placed them)

# 3. Run (XRT lives in /usr on Ubuntu, not /opt/xilinx)
export XILINX_XRT=/usr
./build/ds4_npu_engine --model hy3.fst \
                       --prompt "Hello" --tokens 32 --temp 0.0
#    -> "Hello! How can I help you today ..."
```

The HY3 path is enabled automatically when the engine detects the `hy_v3`
header signature (`qh=64, kh=8, hd2=128`). Speculative decoding via the NextN
MTP head (block 80, shared trunk embedding + LM head) is wired behind the
`--draft_model` / no-SD flags.

## Performance (measured on Ryzen AI 9 365)

| | |
|---|---|
| Decode (coherent, M=16 templated) | **~0.12 tok/s** |
| Decode (raw M=1) | **~0.14 tok/s** |
| Prefill (M=16 templated "Hello") | ~185 s |
| Prefill (M=1 raw) | ~30 s |
| Sample greedy output | `"Hello"` → `Hello! How can I help you today` |
| Decode layer floor | ~89 ms (SSD expert load ~50 ms · NPU FFN ~37 ms · host attention ~7 ms) |

> ⚠️ This is a **research / engineering showcase, not a production server.**
> The decode rate is low because the model is bandwidth-bound: 8 experts × 80
> layers × 10 MB ≈ 6.4 GB/token read from SSD at ~1 GB/s. The value is the
> architecture — expert virtual memory + on-NPU dequant/GEMM/attention via
> IRON-generated MLIR-AIE kernels — and a working end-to-end 299B MoE inference
> path on a consumer NPU. The host attention path is OpenMP + AVX2/FMA
> vectorized; the on-device expert FFN is a fused 4-dispatch/layer path
> (dequant-8experts + multicore gate/up/down GEMM + host SiLU/mul).

## Reproducing the `.fst`

```bash
python3 scripts/hy3_download.py                       # ~145 GB Q3_K_M GGUF (resumable)
python3 scripts/fst_converter.py hy3 <gguf> hy3.fst    # GGUF -> MXFP4 .fst
python3 scripts/verify_fst.py hy3.fst                  # integrity + round-trip check
# tokenizer.json is copied from tencent/Hy3 (engine auto-downloads it on first run)
```

## License

**Apache-2.0**, inherited from the source GGUF repository
[`satgeze/Hy3-1M-GGUF`](https://huggingface.co/satgeze/Hy3-1M-GGUF) and the
upstream model [`tencent/Hy3`](https://huggingface.co/tencent/Hy3). This `.fst`
repackaging is a derivative; the same license terms apply. No additional
restrictions are imposed beyond those in Apache-2.0 and the upstream model's
terms.

## Citation / credits

- Upstream model: **Tencent Hunyuan-3.0** — [`tencent/Hy3`](https://huggingface.co/tencent/Hy3)
- GGUF quantization: [`satgeze/Hy3-1M-GGUF`](https://huggingface.co/satgeze/Hy3-1M-GGUF)
- Inference engine: [FaStar](https://github.com/RaffaelloM/FaStar)