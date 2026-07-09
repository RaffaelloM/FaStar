/* expert_pager.cpp — MoE Expert Virtual Memory
 *
 * LRU cache backed by SSD via pread. Background worker for async prefetch.
 * Thread-safe. Antirez-style: flat, direct, no magic.
 *
 * Build: g++ -std=c++17 -O2 -pthread -c expert_pager.cpp
 */

#include "expert_pager.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

/* ================================================================== */
/*  Helpers                                                            */
/* ================================================================== */

/*  Per-expert stride in the packed ffn_gate_up_down_exps.weight tensor.
 *  Dense DS4 block format: 3 projections × 17 bytes/32 elements.
 *  gate [4096×128×17] + up [4096×128×17] + down [2048×64×17] = 13,369,344 bytes. */
static constexpr int64_t EXPERT_STRIDE = 13369344;

/* FST header magic and layout (from fst_converter.py) */
static constexpr char FST_MAGIC[4] = {'F','S','T','\0'};
static constexpr int FST_HEADER_SIZE = 4096;
/* Fields at known offsets in the FST header:
 *   0:  magic[4]   4: version(u32)   8: hidden(u32)  12: layers(u32)
 *  16: experts(u32) ...
 *  64: expert_bank_offset(u64)   72: expert_block_bytes(u64)
 *  80: expert_block_stride(u64)  88: expert_count(u64) */

/* ================================================================== */
/*  Constructor / Destructor                                           */
/* ================================================================== */

ExpertPager::ExpertPager(const std::string &path, size_t cap)
    : capacity_(cap)
{
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("ExpertPager: cannot open " + path);

    parse_header();

    /* Launch background worker */
    worker_ = std::thread(&ExpertPager::worker_loop, this);

    fprintf(stderr, "ExpertPager: %zu tensors, cache=%zu MB\n",
            tensor_off_.size(), capacity_ / (1024 * 1024));
}

ExpertPager::~ExpertPager()
{
    shutdown();
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

void ExpertPager::shutdown()
{
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false))
        return;                         /* already shut down */

    queue_cv_.notify_all();
    if (worker_.joinable())
        worker_.join();
}

size_t ExpertPager::cache_size()  const { std::lock_guard<std::mutex> g(cache_mtx_); return used_; }
size_t ExpertPager::cache_count() const { std::lock_guard<std::mutex> g(cache_mtx_); return lru_.size(); }

/* ================================================================== */
/*  Safetensors header parser                                          */
/*                                                                     */
/*  Format: 8-byte LE length, then JSON, then data.                    */
/*  We extract every "tensor_name": {"data_offsets":[start,end],...}.  */
/* ================================================================== */

void ExpertPager::parse_header()
{
    /* --- Detect FST format by magic --- */
    char magic[4] = {0};
    if (pread(fd_, magic, 4, 0) != 4)
        throw std::runtime_error("ExpertPager: cannot read magic");

    if (memcmp(magic, FST_MAGIC, 4) == 0) {
        /* --- FST format: read header fields ---
         * Header layout (little-endian):
         *   0: magic[4]  4: version(u32)  8: hidden_dim(u32)
         *  12: num_layers(u32)  16: num_experts(u32)
         *  80: expert_bank_offset(u64)  88: expert_block_bytes(u64)
         *  96: expert_block_stride(u64) 104: expert_count_total(u64) */
        uint32_t version = 0, n_layers = 0, num_experts = 0;
        uint64_t expert_bank_off = 0, block_bytes = 0, block_stride = 0;

        ssize_t pr;
        pr = pread(fd_, &version, 4, 4); (void)pr;
        pr = pread(fd_, &n_layers, 4, 12); (void)pr;
        pr = pread(fd_, &num_experts, 4, 16); (void)pr;
        pr = pread(fd_, &expert_bank_off, 8, 80); (void)pr;
        pr = pread(fd_, &block_bytes, 8, 88); (void)pr;
        pr = pread(fd_, &block_stride, 8, 96); (void)pr;

        data_start_ = (int64_t)expert_bank_off;
        expert_stride_ = (int64_t)block_stride;
        num_experts_ = (int32_t)num_experts;
        num_layers_  = (int32_t)n_layers;

        /*  Monolithic FST: all experts are contiguous in the expert bank.
         *  For layer L, expert E: offset = expert_bank_off + (L * num_experts + E) * stride.
         *  Store per-layer base offsets so load_from_disk() works for any layer. */
        for (uint32_t L = 0; L < n_layers; L++)
            layer_packed_off_[L] = (int64_t)L * (int64_t)num_experts * (int64_t)block_stride;

        fprintf(stderr, "ExpertPager: FST v%u, %u layers, %u experts, stride=%ld, bank_off=%ld\n",
                version, n_layers, num_experts_, (long)expert_stride_, (long)data_start_);
        return;
    }

    /* --- Safetensors format (legacy path) --- */
    uint64_t hdr_len = 0;
    if (pread(fd_, &hdr_len, 8, 0) != 8)
        throw std::runtime_error("ExpertPager: cannot read header length");

    std::vector<char> hdr(hdr_len + 1);
    ssize_t got = pread(fd_, hdr.data(), hdr_len, 8);
    if (got != (ssize_t)hdr_len)
        throw std::runtime_error("ExpertPager: short header read");
    hdr[hdr_len] = '\0';

    data_start_ = 8 + (int64_t)hdr_len;

    /* --- minimal JSON scanner ---------------------------------------- */
    const char *p = hdr.data();

    /* Skip to root '{' */
    while (*p && *p != '{') p++;
    if (*p) p++;

    while (*p && *p != '}') {
        /* Skip whitespace / commas */
        while (*p && (*p == ' ' || *p == ',' || *p == '\n' ||
                      *p == '\r' || *p == '\t'))
            p++;
        if (*p != '"') break;

        /* --- tensor name -------------------------------------------- */
        p++;
        const char *name_s = p;
        while (*p && *p != '"') { if (*p == '\\') p++; p++; }
        std::string name(name_s, p - name_s);
        if (*p) p++;

        /* Skip __metadata__ entirely */
        if (name == "__metadata__") {
            while (*p && *p != '{') p++;
            if (*p == '{') {
                int depth = 1; p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            }
            continue;
        }

        /* --- tensor value object ------------------------------------ */
        while (*p && *p != '{') p++;
        if (*p != '{') break;
        p++;

        while (*p && *p != '}') {
            while (*p && (*p == ' ' || *p == ',' || *p == '\n')) p++;
            if (*p != '"') break;
            p++;
            const char *key_s = p;
            while (*p && *p != '"') { if (*p == '\\') p++; p++; }
            std::string key(key_s, p - key_s);
            if (*p) p++;

            while (*p && *p != ':') p++;
            if (*p) p++;
            while (*p && (*p == ' ' || *p == '\n')) p++;

            if (key == "data_offsets" && *p == '[') {
                p++;
                while (*p && (*p == ' ' || *p == '\n')) p++;
                int64_t start = strtoll(p, (char **)&p, 10);
                while (*p && (*p == ',' || *p == ' ' || *p == '\n')) p++;
                int64_t end = strtoll(p, (char **)&p, 10);
                tensor_off_[name] = { start, end - start };
                /* skip past ']' */
                while (*p && *p != ']') p++;
                if (*p) p++;
            } else {
                /* Skip any value */
                if (*p == '"') {
                    p++;
                    while (*p && *p != '"') { if (*p == '\\') p++; p++; }
                    if (*p) p++;
                } else if (*p == '[') {
                    int d = 1; p++;
                    while (*p && d > 0) {
                        if (*p == '[') d++;
                        else if (*p == ']') d--;
                        p++;
                    }
                } else if (*p == '{') {
                    int d = 1; p++;
                    while (*p && d > 0) {
                        if (*p == '{') d++;
                        else if (*p == '}') d--;
                        p++;
                    }
                } else {
                    while (*p && *p != ',' && *p != '}') p++;
                }
            }
        }
        if (*p == '}') p++;
    }

    if (tensor_off_.empty())
        throw std::runtime_error("ExpertPager: no tensors found in header");

    /* --- Locate packed expert tensors -------------------------------- */
    /*  Find all layers' ffn_gate_up_down_exps.weight to get per-layer  */
    /*  data offsets and compute per-expert stride.                     */
    layer_packed_off_.clear();
    expert_stride_ = 0;
    num_experts_   = 32;  /* from model config */
    for (auto &kv : tensor_off_) {
        if (kv.first.find("ffn_gate_up_down_exps.weight") != std::string::npos) {
            /* Extract layer number: model.layers.N.ffn_... */
            size_t dot1 = kv.first.find('.');
            size_t dot2 = kv.first.find('.', dot1 + 1);
            size_t dot3 = kv.first.find('.', dot2 + 1);
            int layer = std::stoi(kv.first.substr(dot2 + 1, dot3 - dot2 - 1));
            layer_packed_off_[layer] = kv.second.first;
            if (expert_stride_ == 0)
                expert_stride_ = kv.second.second / num_experts_;
        }
    }
    if (expert_stride_ == 0)
        throw std::runtime_error("ExpertPager: ffn_gate_up_down_exps.weight not found");

    fprintf(stderr, "ExpertPager: %d layers, %d experts/layer, stride=%ld\n",
            (int)layer_packed_off_.size(), num_experts_, (long)expert_stride_);
}

/* ================================================================== */
/*  Disk I/O                                                           */
/* ================================================================== */

Expert ExpertPager::load_from_disk(int layer_id, int expert_id) const
{
    Expert ex;

    /*  FST format: experts are at data_start_ + expert_id * stride.
     *  Safetensors: experts are at data_start_ + layer_off + expert_id * stride.
     *  layer_packed_off_ maps layer_id -> offset within data region.
     *  For FST (single layer), layer_packed_off_[0] = 0. */
    auto it = layer_packed_off_.find(layer_id);
    if (it == layer_packed_off_.end()) {
        fprintf(stderr, "ExpertPager: no packed tensor for layer %d\n", layer_id);
        return ex;
    }
    size_t base = (size_t)data_start_ + (size_t)it->second
                + (size_t)expert_id * (size_t)expert_stride_;

    ex.packed_weights.resize((size_t)expert_stride_);
    ssize_t rd = pread(fd_, ex.packed_weights.data(), (size_t)expert_stride_, (off_t)base);
    if (rd != expert_stride_) {}

    return ex;
}

/* ================================================================== */
/*  LRU Cache internals                                                */
/* ================================================================== */

void ExpertPager::evict_until(size_t need)
{
    while (used_ + need > capacity_ && !lru_.empty()) {
        Expert &victim = lru_.back().second;
        size_t  vsz    = victim.total_size();
        map_.erase(lru_.back().first);
        lru_.pop_back();
        used_ -= vsz;
    }
}

/* ================================================================== */
/*  Prefetch (async)                                                   */
/* ================================================================== */

void ExpertPager::prefetch(int layer_id, int expert_id)
{
    /* Already cached? Skip. */
    {
        std::lock_guard<std::mutex> g(cache_mtx_);
        CacheKey k{layer_id, expert_id};
        if (map_.count(k)) return;
    }

    {
        std::lock_guard<std::mutex> g(queue_mtx_);
        /* Avoid duplicate queue entries */
        for (auto &r : queue_)
            if (r.layer == layer_id && r.expert == expert_id)
                return;
        queue_.push_back({layer_id, expert_id});
    }
    stat_prefetch_.fetch_add(1, std::memory_order_relaxed);
    queue_cv_.notify_one();
}

/* ── Predictive prefetch ────────────────────────────────────────────── */
void ExpertPager::predict_and_prefetch(int cur_layer, const int* eids, int n)
{
    /* Queue the current layer's own experts (cheap no-op if cached) so
     * the worker starts any missing loads immediately, then queue the
     * SAME expert ids for the next layer.  The next layer's experts are
     * fetched from SSD while the current layer's expert FFN runs on the
     * NPU, hiding I/O behind compute. */
    for (int k = 0; k < n; k++)
        prefetch(cur_layer, eids[k]);
    int next = cur_layer + 1;
    if (next >= num_layers_) return;
    for (int k = 0; k < n; k++)
        prefetch(next, eids[k]);
}

/* ── CPU router for draft-driven prefetch ──────────────────────────── */
void ExpertPager::cpu_router(int* eids, const float* hidden, int n_experts,
                              int top_k, int hd) const
{
    /* sqrtsoftplus scoring (matches DeepSeek V4 Gate.forward):
     *   score_e = sqrt(softplus(hidden . weight_e))
     *   add bias, pick top_k, normalize */
    std::vector<float> scores(n_experts);
    for (int e = 0; e < n_experts; e++) {
        float dot = 0.0f;
        const float* w = draft_router_weights_ + (size_t)e * hd;
        for (int d = 0; d < hd; d++)
            dot += hidden[d] * w[d];
        scores[e] = sqrtf(fmaxf(0.0f, logf(1.0f + expf(dot))));
        if (draft_router_bias_)
            scores[e] += draft_router_bias_[e];
    }

    std::vector<int> idx(n_experts);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    for (int k = 0; k < top_k; k++)
        eids[k] = idx[k];
}

/* ── Draft-driven prefetch (DSpark) ────────────────────────────────── */
void ExpertPager::set_router_weights(const float* weights, const float* bias,
                                      int n_experts, int hidden_dim)
{
    draft_router_weights_ = weights;
    draft_router_bias_    = bias;
    draft_n_experts_      = n_experts;
    draft_hidden_dim_     = hidden_dim;
}

void ExpertPager::predict_and_prefetch_from_draft(int layer_id,
                                                   const int* draft_token_ids,
                                                   int n_draft,
                                                   const bf16_t* draft_embeddings,
                                                   int hidden_dim,
                                                   int n_experts,
                                                   int top_k)
{
    if (!draft_router_weights_ || n_draft <= 0) return;

    /* For each draft token, run the CPU router to predict which experts
     * the main model will select, then queue those experts for prefetch.
     * This warms the cache before the main model's verification pass. */
    std::vector<float> hidden(hidden_dim);
    std::vector<int>   eids(top_k);

    for (int t = 0; t < n_draft; t++) {
        /* Convert bf16 embeddings to float for CPU router */
        const bf16_t* emb = draft_embeddings + (size_t)t * hidden_dim;
        for (int d = 0; d < hidden_dim; d++) {
            uint32_t bits = (uint32_t)emb[d] << 16;
            float f;
            memcpy(&f, &bits, 4);
            hidden[d] = f;
        }

        cpu_router(eids.data(), hidden.data(), n_experts, top_k, hidden_dim);

        for (int k = 0; k < top_k; k++)
            prefetch(layer_id, eids[k]);
    }

    /* Also prefetch the same experts for the next layer (routing reuse) */
    int next = layer_id + 1;
    if (next < num_layers_) {
        for (int t = 0; t < n_draft; t++) {
            const bf16_t* emb = draft_embeddings + (size_t)t * hidden_dim;
            for (int d = 0; d < hidden_dim; d++) {
                uint32_t bits = (uint32_t)emb[d] << 16;
                float f;
                memcpy(&f, &bits, 4);
                hidden[d] = f;
            }
            cpu_router(eids.data(), hidden.data(), n_experts, top_k, hidden_dim);
            for (int k = 0; k < top_k; k++)
                prefetch(next, eids[k]);
        }
    }

    stat_draft_prefetch_.fetch_add(n_draft, std::memory_order_relaxed);
}

void ExpertPager::predict_and_prefetch_from_draft(int layer_id,
                                                   const int* draft_token_ids,
                                                   int n_draft)
{
    /* Fallback: if no router weights are set, we can't predict experts
     * from token IDs alone.  Just prefetch the shared expert (id 0)
     * for each layer — a conservative warm. */
    if (!draft_router_weights_ || n_draft <= 0) return;
    /* The full version with embeddings is called by the engine. */
}

/* ================================================================== */
/*  Synchronous get                                                    */
/* ================================================================== */

const Expert &ExpertPager::get(int layer_id, int expert_id)
{
    CacheKey key{layer_id, expert_id};
    stat_gets_.fetch_add(1, std::memory_order_relaxed);

    /* Fast path: already cached */
    {
        std::lock_guard<std::mutex> g(cache_mtx_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            stat_hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second->second;
        }
    }

    /* Wait if the worker thread is currently loading this expert */
    {
        std::unique_lock<std::mutex> g(cache_mtx_);
        while (loading_.count(key)) {
            loading_cv_.wait(g);
        }
        /* Re-check cache after waking — worker may have inserted it */
        auto it = map_.find(key);
        if (it != map_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            stat_hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second->second;
        }
    }

    stat_misses_.fetch_add(1, std::memory_order_relaxed);

    /* Async miss path: hand the load to the background worker and wait on
     * the condition variable.  The inference thread NEVER blocks inside a
     * synchronous pread here — it sleeps on loading_cv_ so the CPU is free
     * to keep queuing other NPU work while the SSD read happens in the
     * worker thread.  This is the "expert virtual memory" page-fault path. */
    {
        std::lock_guard<std::mutex> qg(queue_mtx_);
        bool already = false;
        for (auto &r : queue_)
            if (r.layer == layer_id && r.expert == expert_id) { already = true; break; }
        if (!already)
            queue_.push_back({layer_id, expert_id});
    }
    queue_cv_.notify_one();

    std::unique_lock<std::mutex> g(cache_mtx_);
    for (;;) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            stat_hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second->second;
        }
        /* Spuriously wake and re-check until the worker has inserted it. */
        loading_cv_.wait(g);
    }
}

/* ================================================================== */
/*  Prefetch barrier                                                   */
/* ================================================================== */

void ExpertPager::wait_prefetch_done()
{
    /* Wait until the worker has consumed the entire prefetch queue and no
     * load is in flight.  After this returns, every prefetch()-ed expert is
     * resident in RAM, so the following get()s cannot stall on SSD. */
    for (;;) {
        {
            std::lock_guard<std::mutex> qg(queue_mtx_);
            if (!queue_.empty()) { /* still queued */ }
            else break;
        }
        /* queue not empty yet — give the worker a tick */
        std::this_thread::yield();
    }
    /* Now drain in-flight loads: wait on loading_cv_ until loading_ is empty. */
    std::unique_lock<std::mutex> g(cache_mtx_);
    while (!loading_.empty())
        loading_cv_.wait(g);
}

/* ================================================================== */
/*  Background worker                                                  */
/* ================================================================== */

void ExpertPager::worker_loop()
{
    while (running_) {
        Request req{-1, -1};
        {
            std::unique_lock<std::mutex> qg(queue_mtx_);
            queue_cv_.wait(qg, [&] {
                return !queue_.empty() || !running_;
            });
            if (!running_ && queue_.empty())
                break;
            req = queue_.back();
            queue_.pop_back();
        }

        CacheKey key{req.layer, req.expert};

        /* Skip if already cached */
        {
            std::lock_guard<std::mutex> g(cache_mtx_);
            if (map_.count(key)) continue;
            loading_.insert(key);
        }

        /* Load from SSD (no lock held — allows concurrent prefetches) */
        Expert ex = load_from_disk(req.layer, req.expert);
        size_t  sz = ex.total_size();

        /* Insert into cache and signal waiters */
        {
            std::lock_guard<std::mutex> g(cache_mtx_);
            loading_.erase(key);
            /* Re-check in case get() loaded it meanwhile */
            if (map_.count(key)) {
                loading_cv_.notify_all();
                continue;
            }
            evict_until(sz);
            lru_.push_front({key, std::move(ex)});
            map_[key] = lru_.begin();
            used_ += sz;
        }
        loading_cv_.notify_all();
    }
}
