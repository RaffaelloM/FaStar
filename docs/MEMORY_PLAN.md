# MEMORY_PLAN.md — Tiered KV Cache for Qwopus3.6-27B (qwen35) on FaStar

**Goal:** support ≈1M-token decode context on a 64 GB RAM machine (AMD Ryzen AI 9 365)
for the dense Qwen3.5-Next model, where the 14.5 GB MXFP4 weights are fully resident
and the KV cache must be streamed/paged because it does not fit in RAM at full length.

> **Architecture reality (from the GGUF header — see post-mortem §a):** Qwopus3.6 is
> `general.architecture = qwen35`, a **hybrid Mamba2-style SSM + GQA** dense model, not
> a plain GQA transformer. Of 65 blocks, **48 are SSM** (state-space, O(1) state) and
> **17 are full GQA attention** (16 trunk layers at `L % 4 == 3`, plus the NextN/MTP
> layer `blk.64`). **Only those 17 layers have a token-growing KV cache.** This is the
> single most important fact for this plan: the KV cache is ~17/65 of a pure-attention
> model's, which is what makes 1M context feasible at all on 64 GB.

---

## 1. KV memory math (drives the tier split)

Per attention layer, per token, BF16, GQA 4 KV-heads × head_dim 256:

```
K = 4 * 256 * 2 B = 2048 B   V = 2048 B   →  K+V = 4096 B / token / attn-layer
× 17 attn layers                              →  69 632 B / token  ≈ 68 KB / token
```

| context | KV size (17 attn layers) | fits in RAM (≈45 GB free after weights+OS)? |
|--------:|-------------------------:|:-------------------------------------------|
| 32 k    |   2.18 GB                | yes — **Tier 1 (hot)**                     |
| 256 k   |  17.4 GB                 | yes (native `qwen35.context_length`=262144)|
| 512 k   |  34.9 GB                 | tight — needs Tier 2 paging under pressure |
| 1 M     |  68.0 GB                 | **no** — Tier 2 (SSD-mmap) mandatory       |

SSM layers add a **fixed** state `[group=16, state=128, head_dim=256]` ≈ 1 MB/layer × 48
≈ 48 MB total, independent of sequence length. Negligible; always resident (Tier 1).

So: **Tier 1 holds 32 k tokens of KV for the 17 attn layers + all SSM state (~2.2 GB).**
Beyond 32 k, older attn KV pages out to SSD. 1 M tokens requires SSD-backed KV + the
Memory Palace for ancient context (see §4).

---

## 2. The 3-tier KV cache

### Tier 1 — Hot (resident in NPU `host_only` BOs)
- **What:** the most recent `HOT = 32 k` tokens' K/V for the 17 attention layers, plus
  the constant SSM state for all 48 SSM layers, plus the running SSM recurrent state
  (one `[16,128,256]` slab per SSM layer, updated each decode step).
- **Where:** per-layer `xrt::bo(..., flags::host_only, ...)` — exactly the pattern
  `FSTEngine::init_kv_cache_bo()` already uses for `kv_cache_`/`hy3_kv_cache_`
  (`fst_engine.cpp:1398`, `fst_engine.h:108/119/553-554`). Add a `Q35KVCache` struct
  analogous to `Hy3KVCache` but only instantiated for the 17 attn layers.
- **Why host_only:** the attention qk/sv kernels already require host_only B-side BOs
  (`bo_mla_bhost_`, `fst_engine.h:692`); KV stays host-side and is paged to device tiles
  per attention pass. No device-memory pressure.
- **Eviction:** when the hot window advances past `HOT`, the oldest `W` tokens
  (default `W = 4 k`, aligned to the SSM→attn cadence) are **flushed to Tier 2** and the
  BO slot is overwritten ring-buffer style.

### Tier 2 — Cold (SSD-mmap, OS-paged)
- **What:** K/V for all attn-layer tokens older than the hot window, in a single
  pre-allocated, page-aligned file `kv_cache_ssd.bin`.
- **Layout:** `[n_attn_layers=17][max_tokens_CAP][K+V = 4096 B]`, file-backed, 4 KB-aligned
  slabs so each token's K/V is one page-friendly row. For 1 M tokens: 68 GB file on NVMe
  (sparse-allocated with `fallocate(FALLOC_FL_PUNCH_HOLE)` on eviction so only live pages
  consume SSD).
- **Mechanism:** `mmap(MAP_SHARED)` the file; `madvise(MADV_RANDOM)` by default,
  `MADV_WILLNEED` to prefetch the ~`W` tokens about to re-enter attention (e.g. when a
  long-range dependency re-references old tokens, or during prefill of a cached prefix).
  This is the approach of llama.cpp PR [#21792](https://github.com/ggml-org/llama.cpp/pull/21792)
  ("optional mmap KV cache") and the `--kv-mmap-path` fork
  ([Perinban/llama.cpp@539a8aa](https://github.com/Perinban/llama.cpp/commit/539a8aa10166f4ac7930f649fdf12ee8202ac4a6)),
  which wrap the mapping with `ggml_backend_cpu_buffer_from_ptr()`. Issue [#20697](https://github.com/ggml-org/llama.cpp/issues/20697)
  (`--cache-disk`) is the same use case (coding agents exhausting RAM on UMA systems).
- **FaStar reuse:** `ExpertPager` (`src/expert_pager.cpp`) already runs a background
  `pread` worker with `loading_cv_` and predictive prefetch. The same worker pattern
  drives Tier-2 prefetch: when the decode loop is about to attend beyond the hot window,
  queue `madvise(MADV_WILLNEED)` (or a `pread`-into-host-buffer fallback) for the
  required token range so the OS faults the pages in ahead of the attention pass.
  `posix_fadvise(POSIX_FADV_DONTNEED)` on evicted ranges returns pages to the OS.
- **CPU-only constraint:** mmap KV is host-side only (the attention compute already runs
  on host for HY3 — see memory note *HY3 real floor = HOST attention*); device BOs are
  not mmap'd. Consistent with the llama.cpp PRs' CPU-only limitation.

### Tier 3 — Memory Palace (summarized ancient context)
- **What:** when a token range is evicted from Tier 2 (SSD pressure, or context > the
  `CAP` we are willing to keep on SSD), it is not discarded: a **recursive summary** of
  that span is written into a local palace store, and the literal K/V is dropped.
  The summary is re-injected as system/working-context text on demand.
- **Where:** a local C++ store — initially a JSON file
  `qwopus.palace.jsonl` (one record per evicted span: `{span_id, token_range, summary,
  embedding[...]}`), upgraded to an in-process HNSW vector index (or sqlite + a small
  HNSW) for semantic retrieval. This is MemGPT/Letta **archival storage**
  ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560), [Letta memory docs](https://docs.letta.com/letta-code/memory/)):
  main context = system + working context + FIFO queue; external context = recall +
  archival (pgvector HNSW). Letta's "dreaming" (background subagents write lessons into
  memory) maps to our offline summarization pass.
- **Summarizer:** the model itself (self-summarization) or a small auxiliary — write the
  evicted span through a greedy summarize prompt, store the result. Recursive: a palace
  entry can itself summarize older palace entries (MemGPT's recursive summary in the
  FIFO queue's first slot).
- **Retrieval:** on a user turn that references ancient context, embed the query, do a
  top-k palace lookup, inject the matching summaries into the working context (system
  block) before the next prefill. The LLM acts as its own librarian (MemGPT
  function-chaining / `core_memory_replace` analog) — here surfaced as an explicit
  `palace_recall(query, k)` engine API.

### Eviction / retrieval flow
```
decode step emits token t (seq_len grows)
  ├─ SSM layers: update O(1) recurrent state (always Tier 1, ~48 MB)
  └─ attn layers: append K[t], V[t] to Tier 1 ring
       if (t - hot_start) > HOT:
           flush oldest W tokens  →  Tier 2 (mmap writeback / madvise DONTNEED on src)
           if Tier 2 size for this session > CAP_SSD:
               summarize oldest Tier 2 span  →  Tier 3 (palace)
               punch hole in kv_cache_ssd.bin for that span

on long-range attention needing old tokens:
  palace_recall(query)  →  inject summaries into working context (re-prefill)
  OR  madvise(MADV_WILLNEED) the Tier 2 range  →  re-load into a Tier 1 side buffer
```

---

## 3. Integration into `fst_engine.cpp`

1. **New arch path `ARCH_QWEN35`** alongside `ARCH_HY3` (detected in the loader at
   `fst_engine.cpp:1496` from `general.architecture=="qwen35"` / the new
   `TID_Q35_CFG` blob). Config unpacks `full_attention_interval`, `nextn_predict_layers`,
   SSM params, and the per-layer `TID_Q35_LAYER_TYPES` array.
2. **`Q35KVCache3T`** class (new header `include/q35_kv_cache.h`):
   - `std::array<xrt::bo, 17> k_hot, v_hot;` (host_only, sized `HOT*2048*2` B)
   - `int fd_ssd; void* ssd_mmap;` for `kv_cache_ssd.bin`
   - `Palace palace_;` (JSONL + optional HNSW)
   - `append_token(layer, K, V)`, `flush_window()`, `recall(query, k)`,
     `prefetch_range(lo, hi)` (calls the shared background worker).
3. **Decode loop:** the existing per-layer loop branches on `layer_types[L]`:
   SSM → recurrent update (new Mamba2 host/SIMD kernel — **not yet implemented**, see
   post-mortem §b); attn → GQA attention reusing the **HY3 host-attention path**
   (`host_gemm_bnk/xnk/f32f32`, `fst_engine.cpp:123/159/~185`, already SIMD-optimized)
   reading K/V from the 3-tier cache.
4. **Prefill:** chunked (16 tokens) as today; writes directly into Tier 1 until `HOT`
   exceeded, then spills to Tier 2 in the same pass.
5. **Session resume:** antirez/ds4 ships "Disk KV Cache as a first-class disk citizen
   with SHA1-keyed cache files, session resume, prefix reuse, contexts up to 1M tokens"
   ([antirez/ds4](https://github.com/antirez/ds4), [Anemll/ds4-ssd](https://github.com/Anemll/ds4-ssd)).
   We mirror this: `kv_cache_ssd.bin` is keyed by a hash of (model id + token-prefix) so
   a resumed session maps the file and is live in milliseconds, no re-prefill.

---

## 4. Honest scope & caveats

- **Native context is 262 144 (256 k), not 1 M.** `qwen35.context_length = 262144`.
  Reaching 1 M requires YaRN-style RoPE extrapolation (`rope.freq_base = 1e7`,
  `rope.dimension_sections = [11,11,10,0]` already favor long range) **and** the SSD +
  palace tiers absorbing 68 GB of KV. The 1 M target is achievable in principle but is
  beyond the model's trained context — quality at 1 M is unverified.
- **The engine cannot run this model yet.** 48/65 layers are SSM; FaStar has no Mamba2
   scan kernel (only MLA / GQA attention + FFN dequant/GEMM). The GQA attention layers
   and the dense FFN *can* reuse the HY3 host-attention + fused-FFN paths; the SSM
   layers and the NextN head need new host (OpenMP/SIMD, à la the HY3 attention win) or
   NPU kernels. The 3-tier KV design is forward-looking and decoupled from that work.
- **Tier 3 (palace) summarization costs a forward pass** per evicted span; amortized
  over `W`-token spans and done off the decode hot path (background / between turns).
- **CPU-only mmap:** consistent with the llama.cpp PRs — device BOs are not mmap'd; the
  attention that consumes cold KV runs on host, which is already the case for HY3.

---

## Sources
- llama.cpp KV-mmap: PR [#21792](https://github.com/ggml-org/llama.cpp/pull/21792),
  PR [#18747](https://github.com/ggml-org/llama.cpp/pull/18747),
  issue [#20697](https://github.com/ggml-org/llama.cpp/issues/20697),
  [Perinban/llama.cpp `--kv-mmap-path` fork](https://github.com/Perinban/llama.cpp/commit/539a8aa10166f4ac7930f649fdf12ee8202ac4a6),
  [src/llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/1191758c/src/llama-kv-cache.cpp).
- MemGPT / Letta: [arXiv:2310.08560](https://arxiv.org/abs/2310.08560),
  [Letta memory docs](https://docs.letta.com/letta-code/memory/).
- antirez ds4 (DwarfStar): [antirez/ds4](https://github.com/antirez/ds4),
  [Anemll/ds4-ssd](https://github.com/Anemll/ds4-ssd).