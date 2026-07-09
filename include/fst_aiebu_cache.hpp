#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <list>
#include <fstream>

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include "aiebu/aiebu.h"

/* Join a base directory onto a relative path.  Absolute paths and the
 * sentinel base_dir_="." are passed through unchanged, so the default
 * (FST_KERNEL_DIR unset) reproduces the original CWD-relative behaviour. */
static inline std::string aiebu_join_path(const std::string& dir, const std::string& path) {
    if (path.empty()) return path;
    if (path[0] == '/') return path;              // already absolute
    if (dir.empty() || dir == ".") return path;   // CWD-relative (legacy default)
    return dir + "/" + path;
}

/* AiebuKernelCache v3 — Permanent contexts, NO LRU eviction.
 *
 * KEY PRINCIPLE: All hw_contexts are permanent. No eviction ever.
 * This requires that total xclbin count fits within the 8 context limit.
 *
 * For DSpark with unified xclbins:
 * - fst_expert_gemm_vec.xclbin: 1 context (gemm, gemm2)
 * - fst_expert_gemm_down.xclbin: 1 context (gemm_down)
 * - fst_dequant_v4.xclbin: 1 context (dequant)
 * - fst_mla_unified.xclbin: 1 context (qc, kvc, qk, sv, oa, ob, wq_b, k_pe)
 * - fst_ew_unified.xclbin: 1 context (rmsnorm, silu, mul, softmax, rope, router, router_post)
 * Total: 5 permanent contexts (fits within the driver limit on this system).
 */
class AiebuKernelCache {
public:
    AiebuKernelCache(xrt::device& dev, const std::string& base_xclbin)
        : dev_(dev), grp_data_(0), grp_inst_(0), stats_{0,0,0}
    {
        base_xb_ = xrt::xclbin(base_xclbin);
        dev_.register_xclbin(base_xb_);
        base_path_ = base_xclbin;

        // Find the MLIR kernel name in base xclbin
        for (auto& k : base_xb_.get_kernels())
            if (k.get_name().rfind("MLIR_AIE", 0) == 0) {
                base_kernel_name_ = k.get_name();
                break;
            }
    }

    /* Register a kernel with explicit MLIR kernel name.
     * For unified xclbins, multiple kernels share the same xclbin_path.
     * The kernel_name is the actual MLIR function name (e.g., "gate_op", "qk_op").
     */
    void register_kernel_ex(const std::string& name,
                            const std::string& inst_path,
                            const std::string& xclbin_path,
                            const std::string& kernel_name = "") {
        const std::string inst_p = aiebu_join_path(base_dir_, inst_path);
        std::ifstream f(inst_p, std::ios::binary | std::ios::ate);
        if (!f) {
            fprintf(stderr, "[aiebu] ERROR: Cannot open insts file: %s\n", inst_p.c_str());
            return;
        }
        size_t sz = (size_t)f.tellg();
        f.seekg(0);
        std::vector<char> data(sz);
        f.read(data.data(), (std::streamsize)sz);

        char* elf_buf = nullptr;
        uint32_t elf_sz = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            data.data(), (uint32_t)data.size(),
            NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
        if (elf_sz == 0 || !elf_buf) {
            fprintf(stderr, "[aiebu] ERROR: aiebu_assembler_get_elf failed for %s\n", name.c_str());
            return;
        }

        xrt::elf elf(elf_buf, elf_sz);
        auto mod = std::make_unique<xrt::module>(elf);
        free(elf_buf);

        // Register the xclbin (shared by multiple kernels for unified xclbins).
        // base_path_ (from the ctor) is already resolved; bare xclbin_path gets
        // prefixed with base_dir_ (FST_KERNEL_DIR) so register_kernel callers
        // can stay CWD-relative.
        const std::string xb_path = xclbin_path.empty()
            ? base_path_ : aiebu_join_path(base_dir_, xclbin_path);
        xrt::xclbin xb = xclbin_path.empty() ? base_xb_ : xrt::xclbin(xb_path);

        // Only register each xclbin once
        if (xclbin_path.empty() || registered_xclbins_.find(xb_path) == registered_xclbins_.end()) {
            try {
                dev_.register_xclbin(xb);
                registered_xclbins_.insert(xb_path);
                fprintf(stderr, "[aiebu] Registered xclbin: %s\n", xb_path.c_str());
            } catch (const std::exception& e) {
                fprintf(stderr, "[aiebu] WARN: register_xclbin failed for %s: %s\n",
                        xb_path.c_str(), e.what());
            }
        }
        xrt::uuid uid = xb.get_uuid();

        // Create hw_context immediately (permanent, no eviction)
        const std::string& key = xb_path;
        if (ctxs_.find(key) == ctxs_.end()) {
            /* AMDXDNA driver hard limit is 16 concurrent hw_contexts.
             * Exceeding it causes DRM_IOCTL_AMDXDNA_CREATE_HWCTX to return
             * -EINVAL, or worse, panics the kernel amdxdna module.
             * Fail with a clear C++ exception before we corrupt the OS. */
            constexpr size_t MAX_HW_CONTEXTS = 9;
            if (ctxs_.size() >= MAX_HW_CONTEXTS) {
                fprintf(stderr, "[aiebu] FATAL: hw_context limit reached (%zu/%zu). "
                        "Cannot create context for %s. Reduce xclbin count or "
                        "use unified xclbins.\n",
                        ctxs_.size(), MAX_HW_CONTEXTS, xb_path.c_str());
                throw std::runtime_error("hw_context limit exceeded");
            }
            try {
                auto ctx = std::make_unique<xrt::hw_context>(dev_, uid);
                ctxs_[key] = std::move(ctx);
                /* Record the MLIR kernel name for this xclbin so run_blob can
                 * build a dynamic xrt::ext::kernel on this context later. */
                if (kernel_name.empty()) {
                    for (auto& k : xb.get_kernels())
                        if (k.get_name().rfind("MLIR_AIE", 0) == 0) {
                            ctx_kname_[key] = k.get_name();
                            break;
                        }
                } else {
                    ctx_kname_[key] = kernel_name;
                }
                stats_.creates++;
                fprintf(stderr, "[aiebu] Created PERMANENT hw_context for %s (total: %zu)\n",
                        xb_path.c_str(), ctxs_.size());
            } catch (const std::exception& e) {
                fprintf(stderr, "[aiebu] ERROR: Failed to create hw_context for %s: %s\n",
                        xb_path.c_str(), e.what());
                return;
            }
        }

        Entry e;
        e.xclbin_path = xb_path;
        e.uuid = uid;
        e.mod = std::move(mod);
        e.kernel_name = kernel_name.empty() ? base_kernel_name_ : kernel_name;
        /* Keep the raw insts blob so the same kernel can be re-run via the
         * dynamic run_blob path (FST_SEQ_BUILDER self-test / future fused
         * sequences that reuse this xclbin's design). */
        if ((sz % 4) == 0) {
            e.insts_blob.resize(sz / 4);
            std::memcpy(e.insts_blob.data(), data.data(), sz);
        }

        // Create kernel immediately using the permanent context
        try {
            e.krnl = std::make_unique<xrt::ext::kernel>(*ctxs_[key], *e.mod, e.kernel_name);
            e.krnl_valid = true;
            fprintf(stderr, "[aiebu] Created kernel '%s' -> '%s'\n",
                    name.c_str(), e.kernel_name.c_str());
        } catch (const std::exception& e) {
            fprintf(stderr, "[aiebu] ERROR: Failed to create kernel %s: %s\n",
                    name.c_str(), e.what());
            return;
        }

        entries_[name] = std::move(e);

        /* Sample group ids from the first kernel for host_only weight BOs. */
        if (entries_.size() == 1) {
            auto& kr = *entries_[name].krnl;
            try { grp_data_ = (int)kr.group_id(3); } catch (...) { grp_data_ = 0; }
            try { grp_inst_ = (int)kr.group_id(1); } catch (...) { grp_inst_ = 0; }
        }
    }

    /* Register a kernel from a unified xclbin.
     *
     * On this XRT/driver stack the unified xclbin exposes exactly one host
     * symbol: "MLIR_AIE".  Kernel selection is performed by the instruction
     * payload (the per-kernel _insts.bin), not by the xrt::ext::kernel name.
     * We therefore always bind to "MLIR_AIE" and store the original instruction
     * blob so callers can pass it as an argument to the run object. */
    void register_kernel(const std::string& name, const std::string& inst_path,
                         const std::string& xclbin_path = "") {
        register_kernel_ex(name, inst_path, xclbin_path, "");
    }

    xrt::ext::kernel& get(const std::string& name) {
        auto& e = entries_.at(name);
        if (!e.krnl_valid || !e.krnl) {
            fprintf(stderr, "[aiebu] ERROR: Kernel %s is not valid\n", name.c_str());
            throw std::runtime_error("Kernel not valid");
        }
        return *e.krnl;
    }

    bool has(const std::string& name) const {
        auto it = entries_.find(name);
        return it != entries_.end() && it->second.krnl_valid;
    }

    /* Directory holding the _insts.bin / .xclbin files.  Set from FST_KERNEL_DIR
     * (or a CLI arg) before any register_kernel call; bare relative paths passed
     * to register_kernel are resolved against it.  Default "." = CWD-relative,
     * identical to the original behaviour (no regression). */
    void set_base_dir(const std::string& dir) { base_dir_ = dir.empty() ? "." : dir; }
    const std::string& base_dir() const { return base_dir_; }

    enum Phase {
        MAIN_PHASE,
        DRAFT_PHASE,
        LM_HEAD_PHASE
    };

    /* No-op: phase switching disabled with permanent contexts */
    void set_phase(Phase phase) {
        (void)phase;
        // No-op - all contexts are permanent
    }

    void flush_all_except_base() {
        // No-op - all contexts are permanent
    }

    void hard_reset() {
        // No-op - all contexts are permanent
    }

    int data_group_id() const { return grp_data_; }
    int inst_group_id() const { return grp_inst_; }
    size_t total_count() const { return entries_.size(); }
    size_t active_contexts() const { return ctxs_.size(); }

    struct Stats { int creates; int evictions; int hits; };
    Stats stats() const { return stats_; }

    /* ── Dynamic in-process instruction sequences ─────────────────────────
     * Counterpart to register_kernel_ex (which bakes a STATIC _insts.bin into
     * the module at registration time).  run_blob takes an instruction blob
     * built AT RUNTIME by NpuSequenceBuilder (the FastFlowLM npu_sequence
     * pattern: npu_dma_memcpy_nd / npu_dma_wait / rtp_write / npu_maskwrite),
     * builds the ELF from it via aiebu on the xclbin's EXISTING hw_context,
     * wraps it in an xrt::module, creates an xrt::ext::kernel, and returns a
     * STARTED xrt::run — one XRT dispatch that bundles every DMA/register/wait
     * micro-op in the blob.  Mirrors FastFlowLM npu_app::_setup_kernel +
     * create_run (npu_utils.hpp:106/262).
     *
     * The module+kernel are cached by a hash of the blob, so an unchanged
     * sequence (e.g. the same tiling every layer) is NOT rebuilt per dispatch;
     * only a genuinely changed sequence triggers an aiebu re-assembly.  This is
     * what lets one fused xclbin serve varying tiling/addresses across layers
     * without a static _insts.bin per shape. */
    xrt::run run_blob(const std::string& xclbin_key,
                      const std::vector<uint32_t>& blob,
                      xrt::bo* const* args, size_t nargs) {
        auto ctxit = ctxs_.find(xclbin_key);
        if (ctxit == ctxs_.end())
            throw std::runtime_error("run_blob: no hw_context for xclbin " + xclbin_key);
        auto knit = ctx_kname_.find(xclbin_key);
        const std::string& kname = (knit != ctx_kname_.end()) ? knit->second : base_kernel_name_;
        size_t h = blob_hash(blob);
        DynEntry& de = dyn_cache_[xclbin_key];
        if (!de.krnl || de.blob_hash != h) {
            char* elf_buf = nullptr;
            uint32_t esz = aiebu_assembler_get_elf(
                aiebu_assembler_buffer_type_blob_instr_transaction,
                (char*)blob.data(), (uint32_t)(blob.size() * sizeof(uint32_t)),
                NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
            if (esz == 0 || !elf_buf)
                throw std::runtime_error("run_blob: aiebu_assembler_get_elf failed for " + xclbin_key);
            xrt::elf elf(elf_buf, esz);
            de.mod = std::make_unique<xrt::module>(elf);
            de.krnl = std::make_unique<xrt::ext::kernel>(*ctxit->second, *de.mod, kname);
            de.blob_hash = h;
            free(elf_buf);
        }
        xrt::run run(*de.krnl);
        run.set_arg(0, 3);
        run.set_arg(1, 0);
        run.set_arg(2, 0);
        for (size_t i = 0; i < nargs; i++)
            run.set_arg(3 + i, *args[i]);
        run.start();
        return run;
    }

    /* Look up the hw_context + MLIR kernel name registered for an xclbin path
     * (so NpuSequenceBuilder callers can target a known xclbin). */
    bool has_xclbin(const std::string& xclbin_key) const {
        return ctxs_.find(xclbin_key) != ctxs_.end();
    }
    /* xclbin path a registered kernel name is bound to (so callers can target it
     * with a custom blob via run_blob). Empty string if the name isn't known. */
    const std::string& entry_xclbin_path(const std::string& name) const {
        auto it = entries_.find(name);
        return (it != entries_.end()) ? it->second.xclbin_path : base_path_;
    }
    const std::string& xclbin_kernel_name(const std::string& xclbin_key) const {
        auto it = ctx_kname_.find(xclbin_key);
        return (it != ctx_kname_.end()) ? it->second : base_kernel_name_;
    }

    /* Re-run an already-registered kernel through the DYNAMIC run_blob path,
     * using its stored insts blob.  Used by the FST_SEQ_BUILDER self-test to
     * prove the in-process blob -> aiebu -> xrt::module -> kernel -> run path
     * produces identical results to the static file-based module.  Same args
     * layout as the normal krnl(3,0,0,...) dispatch. */
    xrt::run run_registered_blob(const std::string& name,
                                 xrt::bo* const* args, size_t nargs) {
        auto& e = entries_.at(name);
        return run_blob(e.xclbin_path, e.insts_blob, args, nargs);
    }

private:
    // NO MAX_ACTIVE limit - we create as many contexts as needed
    // and let the driver reject if we exceed 8

    struct Entry {
        std::string xclbin_path;
        xrt::uuid uuid;
        std::unique_ptr<xrt::module> mod;
        std::unique_ptr<xrt::ext::kernel> krnl;
        std::string kernel_name;  // Actual MLIR kernel name
        bool krnl_valid = false;
        std::vector<uint32_t> insts_blob;  // raw insts (for run_registered_blob / dynamic re-build)
    };

    /* Dynamic-sequence cache: per-xclbin module+kernel rebuilt only when the
     * instruction blob changes (keyed by a hash of the blob). */
    struct DynEntry {
        std::unique_ptr<xrt::module> mod;
        std::unique_ptr<xrt::ext::kernel> krnl;
        size_t blob_hash = 0;
    };

    static size_t blob_hash(const std::vector<uint32_t>& blob) {
        size_t h = 1469598103934665603ULL;                 // FNV-1a 64-bit offset
        for (uint32_t w : blob) {
            h ^= (size_t)w;
            h *= 1099511628211ULL;
        }
        return h;
    }

    xrt::device& dev_;
    xrt::xclbin base_xb_;
    std::string base_path_;
    std::string base_dir_ = ".";   /* FST_KERNEL_DIR: where _insts/.xclbin live */
    std::string base_kernel_name_;
    std::unordered_set<std::string> registered_xclbins_;
    int grp_data_;
    int grp_inst_;
    Stats stats_;
    std::unordered_map<std::string, Entry> entries_;
    std::unordered_map<std::string, std::unique_ptr<xrt::hw_context>> ctxs_;
    std::unordered_map<std::string, std::string> ctx_kname_;   // xclbin_path -> MLIR kernel name
    std::unordered_map<std::string, DynEntry> dyn_cache_;      // xclbin_path -> dynamic module
};
