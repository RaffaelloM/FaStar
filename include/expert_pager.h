/* expert_pager.h — MoE Expert Virtual Memory
 *
 * Pages expert FFN weights between SSD and RAM using an LRU cache.
 * Async prefetch via background thread + pread.
 * Synchronous get blocks if not cached.
 *
 * Build: g++ -std=c++17 -O2 -pthread -c expert_pager.cpp
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <cstdint>
#include <utility>

using bf16_t = uint16_t;

/* ── Expert representation ────────────────────────────────────────── */
/*  Single contiguous buffer with dense DS4 block format.             */
/*  Layout: gate [out=4096, in=2048, 17B/block], up [same],          */
/*          down [out=2048, in=4096, 17B/block].                      */
/*  Each 17-byte block: 1 e8m0 scale + 16 packed FP4 nibbles.       */

struct Expert {
    std::vector<uint8_t> packed_weights;  /* ~12.75 MB dense FP4 blocks */

    size_t total_size() const {
        return packed_weights.size();
    }
};

/* ── LRU cache key ────────────────────────────────────────────────── */

struct CacheKey {
    int layer_id;
    int expert_id;

    bool operator==(const CacheKey &o) const {
        return layer_id == o.layer_id && expert_id == o.expert_id;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey &k) const {
        size_t h1 = std::hash<int>()(k.layer_id);
        size_t h2 = std::hash<int>()(k.expert_id);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

/* ── ExpertPager ──────────────────────────────────────────────────── */

class ExpertPager {
public:
    /*  safetensors_path: path to the model file on SSD.
     *  ram_cache_bytes:  maximum bytes to keep in RAM.                */
    ExpertPager(const std::string &safetensors_path, size_t ram_cache_bytes);
    ~ExpertPager();

    /*  Stop the background worker thread. Safe to call multiple times. */
    void shutdown();

    /*  Async: queue an expert to be loaded from SSD in the background.
     *  Returns immediately. No-op if already cached.                   */
    void prefetch(int layer_id, int expert_id);

    /*  Predictive prefetch (Markov-like): upon knowing the top-k experts
     *  selected by layer L's router, queue those same expert ids for the
     *  NEXT layer (L+1).  DeepSeek MoE routing reuses expert ids across
     *  adjacent layers for a given token, so this warms the cache while
     *  layer L's expert FFN runs on the NPU — the SSD read of L+1's
     *  experts overlaps with L's NPU compute.  cur_layer's own experts
     *  are also queued (cheap if already loading).                    */
    void predict_and_prefetch(int cur_layer, const int* eids, int n);

    /*  Ping-pong double-buffer prefetch (decode overlap): the engine calls
     *  wait_layer_loaded(L) as a BARRIER at the start of layer L — it blocks
     *  until every expert (L, eids[i]) is resident, queueing any that are
     *  neither cached nor loading first.  The main inference thread never
     *  pread's: it sleeps on loading_cv_ while the background SSD worker reads.
     *  Once the barrier returns, the FFN's get()s are guaranteed instant cache
     *  hits (no SSD stall on the compute path).  Call prefetch_layer_async(L+1)
     *  right after the barrier (Markov: eids_{L+1} ~= eids_L) so the worker
     *  reads L+1's experts from SSD while the NPU computes L — overlapping I/O
     *  with compute.  This is the "expert virtual memory" double-buffer: L is
     *  the read-side buffer, L+1 is the write-side buffer the worker fills.   */
    void prefetch_layer_async(int layer_id, const int* eids, int n);
    void wait_layer_loaded(int layer_id, const int* eids, int n);

    /*  Draft-driven prefetch (DSpark): before the main model verification
     *  pass, the draft model has already generated gamma candidate tokens.
     *  This method runs the layer's router (CPU) on the draft token
     *  embeddings, extracts top_k expert IDs for each draft token, and
     *  queues ALL of them for prefetch from SSD into RAM.  This is the
     *  killer optimization: the pager background thread preloads the
     *  EXACT experts the main model will need, pushing cache hit rate
     *  to >95%.  The router weights and embedding table must be set
     *  via set_router_weights() and set_embedding_table() first.       */
    void predict_and_prefetch_from_draft(int layer_id,
                                          const int* draft_token_ids,
                                          int n_draft,
                                          const bf16_t* draft_embeddings,
                                          int hidden_dim,
                                          int n_experts,
                                          int top_k);
    void predict_and_prefetch_from_draft(int layer_id,
                                          const int* draft_token_ids,
                                          int n_draft);

    /*  Provide router weights for CPU-based draft routing.
     *  router_weights: [n_experts, hidden_dim] float32.
     *  router_bias:    [n_experts] float32 (optional, may be nullptr).
     *  The pager does NOT copy the data — caller must keep it alive.     */
    void set_router_weights(const float* weights, const float* bias,
                             int n_experts, int hidden_dim);

    /*  Sync: return a reference to the expert's data.
     *  If cached, returns immediately (updates LRU position).
     *  If not cached, the inference thread NEVER does a synchronous
     *  pread — it queues the load on the background worker and sleeps on
     *  a condition variable, so the CPU is free while SSD I/O happens. */
    const Expert &get(int layer_id, int expert_id);

    /*  Block until the prefetch queue is fully drained and every queued
     *  expert is resident in RAM.  Call this as a barrier BEFORE a phase
     *  whose experts were just prefetch()-ed, so the subsequent get()s are
     *  guaranteed instant cache hits (no SSD stall during NPU execution). */
    void wait_prefetch_done();

    /*  How many bytes are currently in the cache.                      */
    size_t cache_size() const;

    /*  How many experts are currently in the cache.                      */
    size_t cache_count() const;

    /*  Prediction / cache statistics for the post-mortem.               */
    long gets()      const { return stat_gets_.load(); }
    long hits()      const { return stat_hits_.load(); }
    long misses()    const { return stat_misses_.load(); }
    long prefetched()const { return stat_prefetch_.load(); }
    long draft_prefetch() const { return stat_draft_prefetch_.load(); }
    /* SSD-stall accounting: time spent in get()'s slow path (waiting on an
     * in-flight load or a reactive miss).  hit_rate is misleading because a
     * miss that later succeeds still counts as a hit; wait_us / wait_count
     * are the real "did the SSD stall?" signal. */
    long wait_us()    const { return stat_wait_us_.load(); }
    long wait_count() const { return stat_wait_count_.load(); }
    long max_wait_us()const { return stat_max_wait_us_.load(); }
    /* Ping-pong barrier accounting: time the main thread spent BLOCKED inside
     * wait_layer_loaded() — i.e. SSD read exposed on the compute path that the
     * prefetch did NOT manage to hide.  If this stays near zero, ping-pong is
     * fully overlapping I/O with NPU compute; if it grows, the worker can't
     * keep up (compute < SSD) or Markov misses forced reactive loads.        */
    long barrier_us()    const { return stat_barrier_us_.load(); }
    long barrier_count() const { return stat_barrier_count_.load(); }
    long max_barrier_us()const { return stat_max_barrier_us_.load(); }
    int  prefetch_ahead() const { return prefetch_ahead_; }
    double hit_rate() const {
        long g = stat_gets_.load();
        return g ? (double)stat_hits_.load() / (double)g : 0.0;
    }

private:
    /* ── Disk ────────────────────────────────────────────────────── */
    int fd_;
    int64_t data_start_;   /* byte offset of data region in safetensors */

    /*  Map: tensor name -> (offset within data region, byte size).
     *  Populated once from the safetensors header.                    */
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> tensor_off_;

    /* ── Packed expert tensor (dense DS4 blocks) ──────────────────── */
    /*  Each expert = gate + up + down, 17 bytes per 32-element block. */
    int64_t expert_stride_ = 0;   /* bytes per expert (13,369,344)    */
    int32_t num_experts_   = 0;   /* experts per layer (32)           */
    int32_t num_layers_    = 0;   /* total layers (from header)        */
    /*  Per-layer data offset (within data region). Non-contiguous.    */
    std::unordered_map<int, int64_t> layer_packed_off_;

    /* ── LRU Cache ───────────────────────────────────────────────── */
    size_t capacity_;
    size_t used_ = 0;

    using LruIter = std::list<std::pair<CacheKey, Expert>>::iterator;

    std::list<std::pair<CacheKey, Expert>>        lru_;       /* front = MRU */
    std::unordered_map<CacheKey, LruIter, CacheKeyHash> map_; /* key -> iter */
    std::unordered_set<CacheKey, CacheKeyHash>   loading_;   /* in-flight prefetches */
    mutable std::mutex cache_mtx_;
    std::condition_variable loading_cv_;

    /* ── Prefetch Queue ──────────────────────────────────────────── */
    struct Request { int layer; int expert; };

    std::vector<Request>  queue_;     /* LIFO: pop from back */
    std::mutex            queue_mtx_;
    std::condition_variable queue_cv_;
    std::thread           worker_;
    std::atomic<bool>     running_{true};

    /* ── Prefetch lead ────────────────────────────────────────────── */
    /*  How many layers ahead to queue (Markov: same expert ids).  Set from
     *  FST_PREFETCH_AHEAD env (default 1).  The SSD worker reads ~80 ms/layer
     *  while decode compute is ~125 ms/layer, so a 2-3 layer head start lets
     *  reads finish before get() needs them.  0 = no predictive prefetch
     *  (reactive miss-path only) = the old baseline.                       */
    int                   prefetch_ahead_{1};

    /* ── Statistics (lock-free) ──────────────────────────────────── */
    std::atomic<long> stat_gets_{0};
    std::atomic<long> stat_hits_{0};
    std::atomic<long> stat_misses_{0};
    std::atomic<long> stat_prefetch_{0};
    std::atomic<long> stat_draft_prefetch_{0};
    std::atomic<long> stat_wait_us_{0};      /* total us spent in get() slow path */
    std::atomic<long> stat_wait_count_{0};   /* # of gets that took the slow path */
    std::atomic<long> stat_max_wait_us_{0};  /* worst single get() wait (us) */
    std::atomic<long> stat_barrier_us_{0};     /* us blocked in wait_layer_loaded */
    std::atomic<long> stat_barrier_count_{0};  /* # of wait_layer_loaded calls */
    std::atomic<long> stat_max_barrier_us_{0}; /* worst single barrier wait (us) */

    /* ── Draft-driven prefetch: router weights for CPU routing ──── */
    /*  Non-owning pointers — caller (FSTEngine) keeps data alive.    */
    const float* draft_router_weights_ = nullptr;
    const float* draft_router_bias_    = nullptr;
    int          draft_n_experts_      = 0;
    int          draft_hidden_dim_     = 0;

    /* ── Internal ────────────────────────────────────────────────── */
    void parse_header();
    void worker_loop();
    void evict_until(size_t need);
    Expert load_from_disk(int layer_id, int expert_id) const;
    /*  CPU router: runs sqrtsoftplus top-k on hidden state to predict
     *  which experts a token will need.  Used by draft-driven prefetch. */
    void cpu_router(int* eids, const float* hidden, int n_experts,
                    int top_k, int hd) const;
};
