#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <unordered_map>
#include <cstdlib>
#include <new>
#include <functional>
#include "expert_pager.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/experimental/xrt_ext.h"
using bf16_t = uint16_t;

/* ── NpuSequenceBuilder ───────────────────────────────────────────────────
 * In-process NPU instruction-sequence builder, ported from FastFlowLM's
 * npu_sequence (Source/FastFlowLM-main/src/include/npu_utils/npu_instr_utils.hpp).
 * Instead of loading a STATIC _insts.bin baked at IRON compile time, this builds
 * the control sequence at runtime from npu_cmd ops (DMA block moves, DDR address
 * patches, register writes, waits, mask writes), then dumps it to a uint32 blob
 * that AiebuKernelCache::run_blob turns into one XRT dispatch (via aiebu ->
 * xrt::module -> xrt::ext::kernel).  This is the FastFlowLM fusion pattern: many
 * DMA micro-ops bundled into a single host->NPU submission, with tiling /
 * buffer addresses parameterized at runtime.
 *
 * <climits> must precede the npu headers (npu_cmd.hpp uses UCHAR_MAX). */
#include <climits>
#include "npu_utils/npu_instr_utils.hpp"

class NpuSequenceBuilder {
public:
    NpuSequenceBuilder() : seq_(device_npu2, /*enable_preemption=*/false) {}

    /* DMA copy DDR<->tile (ND, up to 4 dims).  Mirrors npu_sequence::npu_dma_memcpy_nd.
     * arg_idx = which kernel BO argument this BD reads/writes (0-based among the BO
     * args passed to run_blob).  S2MM (tile->DDR) auto-issues a token. */
    void dma_memcpy(int elem_size, int arg_idx, dma_direction dir,
                    npu_tiles tile, npu_bd_id bd, npu_it_channel ch,
                    std::vector<uint32_t> offset, std::vector<uint32_t> size,
                    std::vector<uint32_t> stride,
                    int packet_id = -1, bool issue_token = false) {
        seq_.npu_dma_memcpy_nd(elem_size, arg_idx, dir, tile, bd, ch,
                               offset, size, stride, packet_id, 0, issue_token);
    }
    /* Wait for a DMA channel to drain. */
    void dma_wait(npu_tiles tile, dma_direction dir, npu_it_channel ch) {
        seq_.npu_dma_wait(tile, dir, ch);
    }
    /* Runtime register write (e.g. core control registers). */
    void rtp_write(npu_tiles tile, uint32_t addr, uint32_t value) {
        seq_.rtp_write(tile, addr, value);
    }
    void mask_write(npu_tiles tile, uint32_t addr, uint32_t value, uint32_t mask) {
        seq_.npu_maskwrite(tile, addr, value, mask);
    }
    void clear() { seq_.clear_cmds(); }
    bool valid() const { return const_cast<npu_sequence&>(seq_).sequence_valid(); }

    /* Finalize the cmd list into the uint32 instruction blob consumed by
     * AiebuKernelCache::run_blob.  Invalidates nothing; safe to call once after
     * all ops are appended. */
    std::vector<uint32_t> build() {
        auto d = seq_.dump();              // cmds2seq if not yet valid
        return std::vector<uint32_t>(d.first, d.first + d.second);
    }

private:
    npu_sequence seq_;
};

template<typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    T* allocate(size_t n) {
        size_t bytes = n * sizeof(T);
        constexpr size_t kAlign = 64;
        size_t padded = (bytes + kAlign - 1) & ~(size_t)(kAlign - 1);
        void* p = std::aligned_alloc(kAlign, padded);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) noexcept { free(p); }
    template<typename U> bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template<typename U> bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};

constexpr int INTER_DIM=2048,MLA_Q_LORA=1024,MLA_KV_LORA=512;
constexpr int MLA_N_HEADS=64,MLA_HEAD_DIM=512,MLA_Q_FLAT=MLA_N_HEADS*MLA_HEAD_DIM;
constexpr int MLA_KV_DECOMP=1024;
constexpr int MLA_ROPE_DIM=64;

/* ── Hybrid Connection (HC): DeepSeek V4 4-stream residual ────────────────
 * The residual is NOT a single vector but N_HC=4 streams of hidden_dim.
 * Each sublayer (attn, ffn) is wrapped by hc_pre (read streams -> sublayer
 * input + post/comb gates) and hc_post (write new streams).  See ds4.c
 * hc_pre_from_state_one / hc_post_one / output_hc_head_one. */
constexpr int N_HC = 4;
constexpr int HC_DIM = N_HC * 4096;          /* 16384 — concat of 4 streams */
constexpr int HC_MIX_DIM = 2 * N_HC + N_HC * N_HC;  /* 24 — pre+post+comb */
constexpr int HC_SINKHORN_ITER = 20;
constexpr float HC_EPS = 1e-6f;

struct KVCacheEntry {
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kv_latent;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> k_pe;
    xrt::bo kv_latent_bo;
    xrt::bo k_pe_bo;
};

/* V4 KV-compressor per-layer state (ds4.c attn_state_kv / attn_state_score /
 * attn_comp_kv).  state_kv/state_sc are [2*ratio, comp_width] for ratio=4
 * (two-lane: rows [0,ratio)=primary prev-window, [ratio,2*ratio)=secondary
 * current-window) and [ratio, comp_width] for ratio=128.  cache holds the
 * emitted 512-dim compressed KV rows [n_comp, 512].  Reset per generation. */
struct CompressorState {
    std::vector<float, AlignedAllocator<float>> state_kv;   /* [2*ratio, comp_width] */
    std::vector<float, AlignedAllocator<float>> state_sc;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> cache;     /* [n_comp, 512] */
    int n_comp = 0;
};

struct ModelConfig {
    int n_layers = 0;
    int n_experts = 0;
    int top_k = 0;
    int hidden_dim = 0;
    int vocab_size = 0;
    int max_seq = 128;
    int rope_dim = MLA_ROPE_DIM;
    size_t expert_block_bytes = 0;
    /* DeepSeek-V4 FFN/router params (from config.json; hardcoded defaults
     * match the HF checkpoint so no .fst reconversion is needed). */
    float swiglu_limit = 10.0f;   /* Expert SwiGLU clamp: gate<=lim, up∈±lim (model.py 605-607) */
    int n_hash_layers = 3;        /* L0..L2 use tid2eid hash routing (model.py 561) */
    float route_scale = 1.5f;     /* routed_scaling_factor: weights*=route_scale after
                                   * normalize-to-sum-1 (model.py 588, config.json
                                   * routed_scaling_factor=1.5).  FaStar previously
                                   * omitted this → MoE under-driven 0.667×.  Env
                                   * FST_ROUTE_SCALE overrides at startup. */
    /* V4 KV-compression: per-layer ratio (0/4/128), from TID 40 (F32 [n_layers]).
     * 0 = plain MLA (L0-1, L41-42); 4 = attention compressor w/ two-lane pool
     * (even L2..L40); 128 = odd L3..L39 (deferred — emits no rows for S<128). */
    std::vector<int> compress_ratios;
};

/* DSpark draft model config (from .fst header reserved fields) */
struct DraftConfig {
    int n_layers = 0;        /* 3 MTP stages */
    int n_experts = 0;       /* 256 */
    int top_k = 0;           /* 6 */
    int hidden_dim = 0;      /* 4096 */
    int vocab_size = 0;      /* 129280 */
    int block_size = 5;      /* dspark_block_size: gamma+1 = 5 */
    int markov_rank = 256;   /* dspark_markov_rank */
    int noise_token_id = 128799;
    int max_seq = 128;
    int window_size = 128;   /* dspark sliding-window KV size (win) */
};

struct LayerWeightsC {
    std::vector<bf16_t, AlignedAllocator<bf16_t>> attn_norm,moe_norm,q_norm,kv_norm,wq_a,wq_b,wkv,wo_a,wo_b;
    std::vector<float> router,router_bias;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> shared_gate,shared_up,shared_down;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> w_k_decompress,w_v_decompress;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> w_k_pe;
    std::vector<float> attn_sinks;   /* TID 22: per-head sink logit (n_heads F32) */
    /* V4 KV compressor (ratio=4 attention compressor).  cmp_ape is F32
     * [ratio, comp_width]; cmp_wkv/cmp_wgate are BF16 [comp_width, 4096] in
     * native [N,K] (bcol — NO transpose); cmp_norm is BF16 [512].  Empty for
     * ratio=0 layers.  cmp_ratio=4|128, cmp_comp_width=1024|512.  Only ratio=4
     * emits rows for S<128 (ratio=128 boundary at pos 127 is never reached). */
    std::vector<float> cmp_ape;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> cmp_wkv, cmp_wgate, cmp_norm;
    int cmp_ratio = 0;
    int cmp_comp_width = 0;
    /* HC (Hybrid Connection) per-layer: fn=[HC_DIM,HC_MIX_DIM] BF16 matvec
     * control matrix; scale=[3] F32 (pre/post/comb); base=[24] F32. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hc_attn_fn, hc_ffn_fn;
    std::vector<float> hc_attn_scale, hc_attn_base, hc_ffn_scale, hc_ffn_base;
    xrt::bo bo_wq_a,bo_wq_b,bo_wkv,bo_wo_a,bo_wo_b;
    xrt::bo bo_shared_gate,bo_shared_up,bo_shared_down;
    xrt::bo bo_w_k_decompress,bo_w_v_decompress,bo_w_k_pe;
};

struct ModelWeightsC {
    bf16_t* embedding_table = nullptr;
    size_t  embedding_bytes = 0;
    bf16_t* lm_head = nullptr;
    size_t  lm_head_bytes = 0;
    bf16_t* final_norm = nullptr;
    size_t  final_norm_bytes = 0;
    /* HC global output collapse: fn=[HC_DIM, N_HC] BF16, scale=[1] F32, base=[4] F32.
     * Reduces the 4-stream residual to a plain vector before final_norm + lm_head. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hc_head_fn;
    std::vector<float> hc_head_scale, hc_head_base;
    ~ModelWeightsC();
};

/* Draft model shared weights per stage (mirrors LayerWeightsC but lighter) */
struct DraftLayerWeights {
    std::vector<bf16_t, AlignedAllocator<bf16_t>> attn_norm, ffn_norm;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> wq_a, wq_b, wkv, wo_a, wo_b;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> q_norm, kv_norm;
    std::vector<float> router, router_bias;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> shared_gate, shared_up, shared_down;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> main_proj, main_norm;
    /* HC (Hybrid Connection) per-stage: fn=[HC_DIM,HC_MIX_DIM] BF16 (stored
     * [HC_MIX_DIM,HC_DIM] in checkpoint, transposed at load), scale=[3] F32,
     * base=[HC_MIX_DIM] F32.  Draft uses the SAME HC structure as the main
     * model (hc_mult=4).  Empty => pre-HC fallback (plain single-stream). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hc_attn_fn, hc_ffn_fn;
    std::vector<float> hc_attn_scale, hc_attn_base, hc_ffn_scale, hc_ffn_base;
    std::vector<float> attn_sinks;   /* TID 22: per-head sink logit (n_heads F32) */
    /* Last stage only: */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> draft_norm;   /* final norm */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> markov_w1;    /* [vocab, rank] */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> markov_w2;    /* [vocab, rank] */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> confidence_proj; /* [1, dim+rank] */
    /* Last-stage HC output collapse: hc_head_fn=[HC_DIM, N_HC] BF16 (stored
     * [N_HC,HC_DIM], transposed at load), scale=[1] F32, base=[N_HC] F32. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hc_head_fn;
    std::vector<float> hc_head_scale, hc_head_base;
    xrt::bo bo_wq_a, bo_wq_b, bo_wkv, bo_wo_a, bo_wo_b;
    xrt::bo bo_shared_gate, bo_shared_up, bo_shared_down;
    xrt::bo bo_main_proj;
};

/* DSpark TIDs (must match fst_converter.py) */
constexpr int DRAFT_TID_EMBED = 0;
constexpr int DRAFT_TID_OUTPUT_NORM = 1;
constexpr int DRAFT_TID_LM_HEAD = 2;
constexpr int DRAFT_TID_INPUT_NORM = 3;
constexpr int DRAFT_TID_POST_ATTN_NORM = 4;
constexpr int DRAFT_TID_Q_PROJ = 5;
constexpr int DRAFT_TID_K_PROJ = 6;
constexpr int DRAFT_TID_O_PROJ = 8;
constexpr int DRAFT_TID_ROUTER = 9;
constexpr int DRAFT_TID_ROUTER_BIAS = 10;
constexpr int DRAFT_TID_SHARED_GATE = 11;
constexpr int DRAFT_TID_SHARED_UP = 12;
constexpr int DRAFT_TID_SHARED_DOWN = 13;
constexpr int DRAFT_TID_Q_NORM = 14;
constexpr int DRAFT_TID_KV_NORM = 15;
constexpr int DRAFT_TID_MARKOV_W1 = 16;
constexpr int DRAFT_TID_MARKOV_W2 = 17;
constexpr int DRAFT_TID_CONFIDENCE_PROJ = 18;
constexpr int DRAFT_TID_MAIN_PROJ = 19;
constexpr int DRAFT_TID_MAIN_NORM = 20;
constexpr int DRAFT_TID_DRAFT_NORM = 21;
constexpr int DRAFT_TID_ATTN_SINK = 22;
constexpr int DRAFT_GLOBAL_LAYER = 0xFFFF;

class AiebuKernelCache;

class FSTEngine {
public:
    ModelConfig config_;
    DraftConfig draft_cfg_;
    std::vector<LayerWeightsC> shared_;
    ModelWeightsC model_;
    FSTEngine(const std::string& fst_path,size_t cache_mb=6000);
    ~FSTEngine();
    void generate(const std::vector<int>&,int,float,float=0.9f,void(*)(int,const char*,void*)=nullptr,void*u=nullptr);
    /* DSpark speculative decoding: load draft model, run SD loop */
    void load_draft_model(const std::string& draft_fst_path);
    void generate_dspark(const std::vector<int>&,int,float,float=0.9f,
                          void(*)(int,const char*,void*)=nullptr,void*u=nullptr,
                          bool skip_prefill=false);
    bool has_draft() const { return draft_loaded_; }
    std::vector<bf16_t, AlignedAllocator<bf16_t>> forward_embeddings(const std::vector<int>& token_ids);
    void apply_final_norm_and_lm_head(bf16_t* hidden_state, int M, float* logits);
    int sample_token(float* logits, int vocab_size, float temperature, float top_p);
    int eos_token_id() const { return 1; }

    /* ── Interactive multi-turn API ──────────────────────────────────────
     * KV cache + V4 compressor state persist across calls, so a multi-turn
     * chat reuses the session context without re-prefilling history.
     *   reset_session()  — clear all KV/compressor state, start a fresh chat.
     *   prefill_user()   — feed a user message, return the first reply token.
     *                      First call after reset_session() resets the
     *                      compressor; subsequent calls APPEND at the current
     *                      sequence position (no re-prefill of prior turns).
     *   decode_step()    — generate one token given the previous token id;
     *                      caller stops on eos_token_id() or a stop string. */
    void reset_session();
    int  prefill_user(const std::vector<int>& token_ids, float temperature, float top_p);
    int  decode_step(int prev_tid, float temperature, float top_p);
    int  seq_pos() const { return seq_pos_; }
private:
    void load_shared_weights(const std::string&);
    /* Load ONLY the Hybrid Connection weights (TIDs 23-31) from a tiny
     * <model>.fst.hc sidecar produced by `fst_converter.py hc`.  Mirrors the
     * inline HC reads in load_shared_weights but against a separate file, so
     * the 150 GB main .fst need not be regenerated just to add HC. */
    void load_hc_weights(const std::string& hc_path);
    /* Load ONLY the RMSNorm weights + attention sinks (TIDs 1,3,4,14,15,22)
     * from a tiny <model>.fst.norm sidecar produced by `fst_converter.py norm`.
     * Overrides the bad/stale norms baked into the main .fst by TID, which also
     * sidesteps the dim-key "4096x1x1" collision that swapped attn_norm/moe_norm
     * in load_shared_weights.  Norms are BF16 [hidden_dim]; sinks are F32 [n_heads]. */
    void load_norm_override(const std::string& norm_path);
    void init_device();
    void process_layer(int,bf16_t*,int,const int* input_ids=nullptr);
    void process_mla(int,bf16_t*,int);
    /* V4 KV compressor: stream one token's attn-normed hidden through the
     * per-layer compressor state; on a ratio boundary emit a 512-dim
     * compressed KV row into cmp_state_[lid].cache.  NPU projects (qc kernel),
     * host-float control ops (APE/pool/RMS-norm/RoPE-tail/E4M3) — matches ds4.c
     * compressor_decode_one + compressor_pool_decode_state.  Only ratio=4 emits
     * for S<128. */
    void compress_token(int lid, const bf16_t* hidden_normed, int pos);
    void reset_compressor_state();   /* zero per-layer state + cache (per generation) */
    void process_expert_ffn(int,const bf16_t*,bf16_t*,int,const int*,int,const float*);
    void process_shared_expert(int,const bf16_t*,bf16_t*,int);
    void npu_gemm(const char* kname, bf16_t* C, const bf16_t* A, const bf16_t* B, int M, int N, int K);
    void npu_gemm(const char* kname, bf16_t* C, const bf16_t* A, const xrt::bo& bo_B, size_t b_bytes, int M, int N, int K);
    /* Vectorized MLA GEMM (canonical kernels.mm+zero, 100% aie::mmul).  One
     * dispatch for the 4 MLA xclbins (qc/wqb/ob/qksv).  bcol kernels read B as
     * [N,K] (native weight, NO transpose); brow reads B as [K,N].  B is a host
     * buffer copied into a reusable host_only BO, zero-padded to the kernel's
     * fixed [N_c,K_c]/[K_c,N_c].  M padded + M-tiled in M_c chunks.  Synchronous.
     * NO CPU GEMM, NO scalar kernel.  See npu_gemm_mla_vec in fst_engine.cpp. */
    void npu_gemm_mla_vec(const char* kname, bf16_t* C,
                            const bf16_t* A, const bf16_t* B_host,
                            int M, int N, int K,
                            const xrt::bo* B_dev = nullptr, size_t B_dev_off = 0);
    /* When B_dev is non-null AND the kernel's baked K_c == K (single K-chunk,
     * i.e. qck/wqb), the weight is read directly from the device-resident
     * persistent BO as a contiguous sub-buffer — NO per-dispatch host strided
     * copy and NO bo_mla_bhost_.sync.  Eliminates the dominant weight re-sync.
     * K-split kernels (ob, K_c<K) cannot use this (K-chunks are strided in the
     * [N,K] persistent BO) and fall back to the B_host path. */
    /* HC 8-row control matvec (M=8, K=HC_DIM, N=24 or 4) on the vectorized qc
     * kernel — reuses the qc hw_context (NO 9th context).  K-tiled 4096-deep,
     * partials accumulated in float; B is [K,N] (hc_fn layout), transpose-
     * copied per K-tile into the bcol [N_c,K_c] host BO.  ZERO CPU GEMM. */
    void npu_gemm_hc(bf16_t* C, const bf16_t* A, const bf16_t* B,
                       int M, int N, int K);
    /* Random-matrix GEMM correctness probe (env FST_RANDMAT_TEST). Feeds random
     * bf16 B [N,K] row-major through the real NPU GEMM and compares to float32
     * CPU ref. cos<0.999 == layout bug, not precision. Exits 0. */
    void randmat_test();
    void preload_all_layers();
    void init_kv_cache_bo();
    void apply_rope(bf16_t* q,bf16_t* k,int M,int S);
    void apply_rope_pe(bf16_t* pe,int npos,int pos_start);
    /* Build the per-position RoPE cos/sin LUT for `M` tokens starting at
     * seq_pos_, using the V4 Flash per-layer RoPE params (ds4.c
     * layer_rope_freq_base / rope_tail_layer_inplace).  Layers 0,1 (compress
     * ratio 0) use plain extrapolation, freq_base=10000.  Layers >=2 (ratio 4
     * even / 128 odd) use YaRN-interpolated RoPE: compress_rope_freq_base=160000,
     * scale_factor=16 (freq_scale=1/16), ext_factor=1, n_ctx_orig=65536,
     * beta_fast=32, beta_slow=1.  The YaRN magnitude scale cancels to 1.0, so
     * only the theta ramp-mix differs from extrapolation.  lut is [M, rope_dim]
     * interleaved (cos_d, sin_d, cos_{d+1}, sin_{d+1}, ...).  inverse=true
     * negates sin (rotates the SV attention output back, ds4.c sin_sign=-1). */
    void build_rope_lut(bf16_t* lut, int M, int lid, bool inverse=false);
    /* RoPE on the rope_dim tail of EVERY head's pe split (ds4.c model.py:505
     * rotates all n_heads tails, not just head 0).  `heads` is row-major
     * [M*n_heads, head_dim] (the q_full / out_pad layout: row r=m*n_heads+h),
     * so the tail of head h at position m is at offset r*head_dim+nope_dim.
     * Gathers each of the M*n_heads tails, builds a per-position [M, rope_dim]
     * LUT via build_rope_lut and broadcasts it across heads, runs npu_rope over
     * M*n_heads rows, scatters back.  inverse=true selects the inverse rotation
     * (sin_sign=-1) used on the attention output (ds4.c). */
    void rope_qpe_all_heads(bf16_t* heads, int M, int lid, bool inverse);
    /* YaRN RoPE on the 64-dim tail of a single 512-dim compressed KV row at
     * absolute position `pos` for layer `lid` (ds4.c rope_tail_layer_inplace
     * on out_comp).  Used by compress_token. */
    void rope_tail_yarn(float* x512, int lid, int pos);
    void prefill_tiled(bf16_t* hidden, int M, const int* input_ids=nullptr,
                       int pos_start=0, bool reset_cmp=true);
    /* ── Hybrid Connection (4-stream residual) ───────────────────────────
     * residual_hc layout: [M, N_HC, hd] (M tokens × 4 streams × hidden).
     * hc_pre: read streams -> sublayer input cur[M,hd] + post[M,N_HC] + comb[M,16].
     * hc_post: write new streams from block_out + gates + old streams.
     * output_hc_head: collapse final streams -> plain[M,hd] for final_norm+lm_head.
     * hc_fn_bo is the device BO holding the fn matrix (NPU matvec B-side). */
    void hc_pre(int M, const bf16_t* residual_hc, const bf16_t* hc_fn,
                const float* hc_scale, const float* hc_base,
                bf16_t* cur, float* post, float* comb);
    void hc_post(int M, const bf16_t* block_out, const bf16_t* residual_hc,
                 const float* post, const float* comb, bf16_t* out_hc);
    void output_hc_head(int M, const bf16_t* residual_hc, bf16_t* out_plain);

    void npu_elementwise(const char* kname, const bf16_t* in, bf16_t* out, int n);
    void npu_elementwise_bin(const char* kname, const bf16_t* in_a, const bf16_t* in_b, bf16_t* out, int n);
    /* BO-to-BO async variants: NO host round-trip.  Inputs/outputs stay on
     * device.  The run is pushed onto pending_runs_; caller flushes at the
     * cross-context dependency boundary. */
    void npu_ew_async(const char* kname, xrt::bo& in_bo, xrt::bo& out_bo);
    void npu_ew_bin_async(const char* kname, xrt::bo& in_a, xrt::bo& in_b, xrt::bo& out_bo);
    // Batched multi-core FFN core (FST_MC_FFN).  Processes experts in batches of
    // E_MC=6.  prepare_slot(slot, out_sub) fills expert slot_eids[slot]'s gate|up|
    // down (3*proj_elems bf16, 3x-stride) into the batched-BO sub-buffer out_sub
    // (routed: NPU dequant; shared: host memcpy of preloaded BF16).  weight_of
    // (slot, m) returns the router weight for expert slot_eids[slot] on token m
    // (0 = not routed / unused slot).  Reads back C_down [E_MC*Mx, hd] and
    // accumulates into acc[m*hd+d].  Caller uploads h (replicated to Mx rows)
    // into bo_scratch_in_ before calling.
    void run_batched_ffn(int M, int Mx, int hd, const std::vector<int>& slot_eids,
                         const std::function<void(int slot, xrt::bo& out_sub)>& prepare_slot,
                         const std::function<float(int slot, int m)>& weight_of,
                         float* acc, int dbg_lid, bool dbg_shared);
    void npu_rope(const char* kname, const bf16_t* in, const bf16_t* lut, bf16_t* out, int rows, int cols);
    void npu_softmax_row(bf16_t* scores, int S_padded);
    void npu_rmsnorm_weighted(bf16_t* out, const bf16_t* in, const bf16_t* weight, int d);
    void npu_router(int* eids, float* wts, const bf16_t* hidden, const float* router_weights, const float* router_bias, int M, int n_experts, int top_k, int hd, int lid, const int* input_ids);
    int npu_sample_token(const bf16_t* hidden, int M, float temperature, float top_p);

    /* BO-to-BO async GEMM: dispatch a non-tiled (N<=2048) GEMM writing to
     * a device output BO.  A is host data (uploaded), B is a device BO.
     * NO sync_from, NO memcpy of C.  Caller flushes at the dependency
     * boundary.  For N>2048, falls back to the tiled readback path.       */
    void npu_gemm_async(const char* kname, xrt::bo& bo_C,
                        const bf16_t* A, const xrt::bo& bo_B,
                        int M, int N, int K);

    std::unique_ptr<ExpertPager> pager_;
    xrt::device npu_device_;
    std::unique_ptr<AiebuKernelCache> kernel_cache_;
    std::vector<KVCacheEntry> kv_cache_;
    std::vector<CompressorState> cmp_state_;   /* per-layer V4 compressor state */
    int current_loaded_layer_ = -1;
    int grp_ = 0;
    std::vector<bf16_t> host_embedding_table_;
    std::vector<bf16_t> host_lm_head_;
    xrt::bo bo_lm_head_;
    int lm_head_n_pad_ = 131072;  /* vocab padded to a multiple of the 2048 N-tile */
    int seq_pos_=0;
    bool is_prefill_=true;
    float rope_freq_base_=10000.0f;
    /* Hash routing (model.py 561): L0..L<n_hash_layers use a per-token-id expert
     * table (tid2eid [n_hash_layers, vocab, n_activated]) instead of score-based
     * topk.  Loaded from the .fst.tid2eid sidecar (extracted from HF safetensors;
     * avoids a full reconversion).  Empty => hash layers fall back to score-based. */
    std::vector<int> tid2eid_;   /* [n_hash_layers * vocab * top_k], int32 */
    bool tid2eid_loaded_ = false;
    bool validate_npu_kernels();

    // Pre-allocated scratch BOs (zero allocation in hot path)
    // Use xrt::bo (has default ctor) and assign xrt::ext::bo in constructor
    xrt::bo bo_scratch_in_;
    xrt::bo bo_scratch_out_;
    xrt::bo bo_scratch_big_;   // For large B data (dequant, etc.)
    xrt::bo bo_scratch_packed_;
    xrt::bo bo_dequant_out_;  // NPU Q4_K dequant output (24 MB)
    xrt::bo bo_d1_, bo_d2_, bo_d3_, bo_d4_;
    // Batched multi-core FFN (FST_MC_FFN): 6-expert batched B in 3x-stride
    // gate|up|down layout, read by fst_expert_gemm_vec_mc (gate/up) and
    // fst_expert_gemm_down_mc (down).  E_MC=6 slots x 3 x proj_elems = 302 MB.
    xrt::bo bo_batch_w_;

    // ── 4-set FFN scratch BO pool (async pipeline) ───────────────────────
    // A pool of 4 independent scratch sets so that expert E+1's dequant
    // (which writes set[(E+1)&3].bo_w) can be dispatched while expert E's
    // down GEMM (reading set[E&3].bo_down) is still in-flight on a different
    // hw_context — without any BO being reused while a run reads it.  The
    // routed-expert loop rotates pool_idx_ across the 4 sets and defers each
    // expert's single host readback (down_out) to the NEXT expert's first
    // cross-context barrier, so (down_E || dequant_{E+1}) overlap.
    //
    // Same-context FIFO hazard (the bug that made the M=16 MLA async non-
    // deterministic): gate+up share the gemm_vec hw_context, so two experts'
    // gate runs on that context execute in submission order.  The pool makes
    // each expert's gate/up BOs distinct (bo_ha/bo_hb), and flush_pending_runs()
    // at every cross-context boundary + before set reuse guarantees no BO is
    // overwritten while an in-flight run reads it.  pending_runs_ never holds
    // more than two experts' runs (prev down + cur dequant) before a flush.
    //
    // The single-expert paths (process_shared_expert, process_draft_*) use
    // scratch_pool_[0] — no cross-expert overlap, but the same per-set BOs.
    struct ScratchSet {
        xrt::bo bo_w;       // dequant output: B[N,K] row-major (64 MB), read by gemm
        xrt::bo bo_ha;      // gate GEMM output  [Mx, INTER_DIM]
        xrt::bo bo_hb;      // up   GEMM output  [Mx, INTER_DIM]
        xrt::bo bo_silu;    // silu(gate)        [Mx, INTER_DIM]
        xrt::bo bo_mul;     // silu*up = down A  [Mx, INTER_DIM]
        xrt::bo bo_down;    // down GEMM output  [Mx, hd]  (the one host readback)
        int     owner_e = -1;  // routed expert id whose down is pending in this set
    };
    std::array<ScratchSet, 4> scratch_pool_;
    int pool_idx_ = 0;     // routed-expert set selector (rotates 0..3)

    // ── Persistent host_only BO cache for routed-expert packed weights ──
    // The dequant kernel reads the 13.37 MB MXFP4 packed weights every
    // dispatch; without a cache the engine memcpy's them host->device (sync_to,
    // ~2 ms) every expert every dispatch.  Caching a host_only BO per (layer,
    // expert) lets dequant read directly with NO per-dispatch memcpy/sync_to
    // (the host_only BO is device-readable; written once at insert time).
    // Bounded LRU (cap bytes) so it can't OOM: evicts the least-recently-used
    // BO when full (a BO miss re-loads from SSD via the pager).  Main model
    // only (draft uses draft_pager_ with its own 0..2 layer range — kept on
    // the memcpy path to avoid a (lid,e) key collision with the main model).
    // 20 GB holds all 1376 experts (18 GB) so every dispatch after warmup is a
    // HIT (no per-dispatch sync_to).  Because the BO cache holds all experts
    // device-readable, the pager's vector cache is now only a transient SSD-
    // load staging area (kept small via cache_mb) — it does NOT need to also
    // cache experts, avoiding a 2x memory duplication (vector + BO).
    static constexpr size_t EXPERT_BO_CAP = 20ULL * 1024 * 1024 * 1024; // 20 GB
    std::unordered_map<CacheKey, xrt::bo, CacheKeyHash> expert_bo_cache_;
    std::list<CacheKey> expert_bo_lru_;        // front = MRU
    size_t expert_bo_bytes_ = 0;
    // Returns a persistent host_only BO holding the (lid,e) expert's packed
    // weights.  BO-cache HIT returns immediately WITHOUT calling the pager
    // (no SSD read, no vector alloc — the BO is the authoritative copy).  MISS
    // loads the expert from SSD via the pager, copies it once into a new BO,
    // syncs, and LRU-evicts to stay under EXPERT_BO_CAP.
    xrt::bo& get_expert_bo(int lid, int e);

    // ── Draft model expert BO cache (mirror of the main cache above) ──────
    // Draft experts live in their own draft_pager_ (layers 0..2) and would
    // collide with the main cache on the (lid,e) key (main also uses lid 0..2),
    // so they get a SEPARATE cache.  Same persistent host_only BO + LRU pattern:
    // every draft FFN dispatch after warmup is a HIT (no per-dispatch memcpy,
    // no per-dispatch sync_to) — the same ~2ms-upload elimination that made the
    // main FFN path persistent.  Draft has only 3 layers × n_experts, so a modest
    // cap holds them all.
    static constexpr size_t DRAFT_EXPERT_BO_CAP = 4ULL * 1024 * 1024 * 1024; // 4 GB
    std::unordered_map<CacheKey, xrt::bo, CacheKeyHash> draft_expert_bo_cache_;
    std::list<CacheKey> draft_expert_bo_lru_;   // front = MRU
    size_t draft_expert_bo_bytes_ = 0;
    xrt::bo& get_draft_expert_bo(int lid, int e);

    std::vector<xrt::run> pending_runs_;
    void flush_pending_runs();   // dependency barrier

    // ── MLA BO-to-BO chain intermediates ──────────────────────────────
    // Dedicated BOs so the MLA path never reads GEMM results back to the
    // host (except the final ob for the residual add).  Sized for M_PAD=8
    // and S_MAX=128 (max_seq).
    xrt::bo bo_mla_h_;        // [M_PAD, hd] input hidden
    xrt::bo bo_mla_qc_;       // [M_PAD, MLA_Q_LORA]
    xrt::bo bo_mla_kv_;       // [M_PAD, MLA_KV_LORA]
    xrt::bo bo_mla_qfull_;    // [M_PAD, MLA_Q_FLAT]
    xrt::bo bo_mla_kpe_;      // [M_PAD, MLA_N_HEADS*rope_dim]
    xrt::bo bo_mla_scores_;   // [M_PAD, max_seq] QK scores
    xrt::bo bo_mla_ao_;       // [M_PAD, MLA_KV_DECOMP] SV output
    xrt::bo bo_mla_oa_;       // [M_PAD, hd]
    xrt::bo bo_mla_ob_;       // [M_PAD, hd]
    xrt::bo bo_rope_lut_;     // [M_PAD, rope_dim] cos/sin LUT
    // Per-M-tile (M=8) reusable BOs for qk/sv, which run on the kvc kernel
    // (8x4096x512).  The AMDXDNA driver rejects sub-BOs as kernel args, so we
    // slice M_LAT into 8-row tiles by copying each tile into these full BOs
    // (no per-tile alloc — sized once, reused every layer/tile).
    xrt::bo bo_mla_atile_;    // A tile [8, K_PAD=4096]
    xrt::bo bo_mla_ctile_;    // C tile [8, N_PAD=512]
    xrt::bo bo_mla_atile2_;   // ping-pong A tile (overlaps upload with prev run)
    xrt::bo bo_mla_ctile2_;   // ping-pong C tile (overlaps readback with next run)
    xrt::bo bo_mla_bhost_;    // host_only B tile [K_PAD, N_PAD] for qk/sv (kvc
                              // kernel BDs require host_only B; device BO hangs)
    xrt::bo bo_mla_bhost2_;   // ping-pong host_only B tile (M=16 async K-split)

    // ── Dispatch packing (NpuSequenceBuilder / run_blob) ───────────────
    // Packed wqb: the engine currently dispatches wqb 16× (one per N-tile of
    // the 32768-wide wq_b projection).  wqb_packed_blob_ is ONE instruction
    // blob with 16 replicated copies of the wqb op-words, each copy's bd1
    // (weight, arg_idx=1) and bd2 (output, arg_idx=2) DDR_PATCH arg_off patched
    // to that N-tile's offset, so ONE run_blob dispatch computes all 16 tiles
    // (proven byte-identical to 16 single dispatches by tools/wqb_pack_probe).
    // bo_wqb_pack_out_ holds the 16 contiguous output tiles [16 * (M_c*N_c*2)].
    std::vector<uint32_t> wqb_packed_blob_;
    xrt::bo bo_wqb_pack_out_;   // 16 * c_tile_b = 1 MiB

    // Packed OA: the O-projection runs 8 grouped GEMMs (one per wo_a group,
    // each [M=16,N=1024,K=4096] on the qck kernel).  oa_packed_blob_ is ONE
    // instruction blob with 8 replicated copies of the qck op-words, each
    // copy's bd0 (A=heads_g, arg_idx=0), bd1 (B=wo_a, arg_idx=1) and bd2
    // (C=low_g, arg_idx=2) DDR_PATCH arg_off advanced to that group's slice,
    // so ONE run_blob dispatch computes all 8 groups (8 -> 1).  Unlike wqb the
    // A matrix differs per group, so bd0 is patched too.  bo_oa_pack_a_ holds
    // the 8 contiguous A tiles, bo_oa_pack_out_ the 8 contiguous C tiles.
    std::vector<uint32_t> oa_packed_blob_;
    xrt::bo bo_oa_pack_a_;     // 8 * a_tile_b (16*4096*2) = 1 MiB
    xrt::bo bo_oa_pack_out_;   // 8 * c_tile_b (16*1024*2) = 256 KiB

    // ── DSpark draft model ────────────────────────────────────────────
    bool draft_loaded_ = false;
    std::vector<DraftLayerWeights> draft_shared_;
    std::vector<bf16_t> draft_embedding_table_;  /* shared with main */
    std::vector<bf16_t> draft_lm_head_;          /* shared with main */
    std::vector<bf16_t> draft_lm_head_padded_;  /* [D, N_PAD] transposed+zero-padded, lazily built */
    std::unique_ptr<ExpertPager> draft_pager_;
    std::vector<KVCacheEntry> draft_kv_cache_;
    int draft_current_loaded_layer_ = -1;

    /* Draft model internal methods */
    void load_draft_shared_weights(const std::string& path);
    void ensure_draft_layer_loaded(int lid);
    void process_draft_layer(int lid, bf16_t* h, int M, const bf16_t* main_x);
    void process_draft_mla(int lid, bf16_t* h, int M, const bf16_t* main_x);
    void process_draft_expert_ffn(int lid, const bf16_t* h, bf16_t* out, int M);
    void process_draft_shared_expert(int lid, const bf16_t* h, bf16_t* out, int M);

    /* Draft forward: produces gamma candidate tokens + confidence scores.
     * main_hidden: the main model's last hidden state (dim*3 for 3 target layers).
     * last_token_id: the last accepted token.
     * Returns: draft_tokens[gamma], confidence[gamma] */
    struct DraftResult {
        std::vector<int> tokens;       /* gamma candidate token IDs */
        std::vector<float> confidence; /* per-token confidence scores */
    };
    DraftResult forward_draft(const bf16_t* main_hidden, int last_token_id,
                               float temperature);

    /* SD statistics for post-mortem */
    long sd_total_draft_tokens_ = 0;
    long sd_accepted_tokens_ = 0;
    long sd_verify_passes_ = 0;
    int draft_seq_pos_ = 0;  /* draft model's own sequence position */

    /* ── HF-faithful DSpark wiring ──────────────────────────────────────
     * main_hidden = concat of HC-mean(h) over the 3 target layers (40,41,42),
     * i.e. mean over the N_HC streams -> [hd] per layer, concat -> [3*hd=12288].
     * Captured during the MAIN model's forward (prefill + each verify pass).
     * target_hc_[i] holds [n_tokens, hd] captured at layer (40+i); the caller
     * selects the accepted token's row and concats the 3 into main_hidden.
     * draft_kv_window_[l] is the per-stage sliding window of main_kv (one
     * main_kv per token, slot = global_pos % window_size); main_kv comes from
     * main_x = main_norm(main_proj(main_hidden)) shared across all stages. */
    std::array<std::vector<bf16_t>, 3> target_hc_;   /* per-target-layer HC-mean capture */
    std::vector<bf16_t> draft_main_x_;              /* [hd] shared projected main hidden */
    int draft_global_pos_ = 0;                       /* #tokens fed to the draft KV window */
    int target_anchor_idx_ = 0;                      /* row in target_hc_ for the last accepted token */
    void capture_target_hc(int lid, const bf16_t* h, int M);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> compute_draft_main_x(int count, const bf16_t* main_hidden);
    void fill_draft_window(int count, const bf16_t* main_x, int base_pos);
};
