// tools/gdn_8kslab_full_probe.cpp — FULL-RECURRENCE 8 KB-stack probe.
// Loads fst_gdn_8kslab_full.xclbin (real GDN 3-pass recurrence, K=8 × 48 v-heads,
// 8 KB bf16 stack S) + fst_gdn_8kslab_noop.xclbin (same 8 KB stack + DMA, no
// recurrence), runs both, and reports full_ms / noop_ms / recurrence_ms and the
// per-vec S-stack-access latency — the decisive number that fills the empty cell
// in docs/QWOPUS_GDN_8KB_SLAB_ANALYSIS.md.
//
//   XILINX_XRT=/usr ./tools/gdn_8kslab_full_probe kernels
//
// Decision: per-vec ~tens-of-ns (fast, ≤8 KB stack pipelines under the real
// recurrence) => slab structure VIABLE; ~1µs+ (degrades to the ≥16 KB rate) =>
// impasse stands, GDN M=K on-tile-state dead.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include "aiebu/aiebu.h"

constexpr int NVH = 48, TOTAL = NVH;
constexpr int HV = 128, COLS = 32, KST = 8, V = 8, NCV = COLS / V;
// S-stack vec accesses (2tile-comparable convention: passA-read + passB-write):
// per token = HV*NCV (passA read) + HV*NCV (passB write) = 128*4 + 128*4 = 1024;
// × K=8 × NVH = 393216.  (+init 512/v-head, ~0.4% — folded into recurrence.)
constexpr double SVEC = (double)HV * NCV * 2 * KST * NVH;   // 393216

static std::vector<char> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    std::vector<char> d((size_t)f.tellg()); f.seekg(0); f.read(d.data(), d.size()); return d;
}

// Run one xclbin, return steady-state ms (best of NRUNS, after a warmup).  in is
// unused by the kernel (deterministic) but streamed to exercise the shim MM2S.
static double run_one(xrt::device& dev, const std::string& dx, const std::string& ix,
                      std::vector<float>& out) {
    xrt::xclbin xclb(dx); dev.register_xclbin(xclb); xrt::uuid uid = xclb.get_uuid();
    xrt::hw_context ctx(dev, uid);
    auto insts = read_file(ix); char* eb = nullptr;
    uint32_t es = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction, insts.data(),
        (uint32_t)insts.size(), NULL, 0, (void**)&eb, NULL, 0, "", "", NULL, 0);
    xrt::elf elf(eb, es); xrt::module mod(elf); free(eb);
    std::string kn; for (auto& k : xclb.get_kernels()) kn = k.get_name();
    xrt::ext::kernel krnl(ctx, mod, kn);
    xrt::ext::bo boIn(dev, (size_t)TOTAL * 4), boOut(dev, (size_t)TOTAL * 4);
    { auto* p = boIn.map<float*>(); for (int i = 0; i < TOTAL; i++) p[i] = 0.0f; }
    boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    constexpr int NRUNS = 5;
    double best = 1e9;
    for (int r = 0; r < NRUNS + 1; ++r) {        // +1 warmup
        { auto* p = boOut.map<float*>(); memset(p, 0, (size_t)TOTAL * 4); }
        boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t0 = std::chrono::steady_clock::now();
        auto run = krnl(3, 0, 0, static_cast<xrt::bo&>(boIn), static_cast<xrt::bo&>(boOut));
        if (run.wait(60000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "TIMEOUT %s\n", dx.c_str()); return -1.0; }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r > 0 && ms < best) best = ms;        // skip warmup
    }
    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* po = boOut.map<float*>();
    out.assign(po, po + TOTAL);
    return best;
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : ".";
    std::string DXf = std::string(dir) + "/fst_gdn_8kslab_full.xclbin";
    std::string IXf = std::string(dir) + "/fst_gdn_8kslab_full_insts.bin";
    std::string DXn = std::string(dir) + "/fst_gdn_8kslab_noop.xclbin";
    std::string IXn = std::string(dir) + "/fst_gdn_8kslab_noop_insts.bin";

    xrt::device dev(0);
    std::vector<float> outf, outn;
    double full_ms = run_one(dev, DXf, IXf, outf);
    double noop_ms = run_one(dev, DXn, IXn, outn);
    if (full_ms < 0 || noop_ms < 0) return 3;

    double rec_ms = full_ms - noop_ms;
    double per_vec_ns = (rec_ms * 1e6) / SVEC;   // ms→µs*1000 then /vec = ns/vec

    // sanity: full checksum should be finite + vary per v-head (recurrence ran)
    float mn = 1e30f, mx = -1e30f; int finite = 0;
    for (int v = 0; v < NVH; v++) {
        float y = outf[v];
        if (std::isfinite(y)) finite++;
        if (y < mn) mn = y; if (y > mx) mx = y;
    }

    fprintf(stderr, "=== gdn_8kslab_full probe (1 tile, 8KB stack, 48 v-head x K=8) ===\n");
    fprintf(stderr, "full_ms   = %.2f ms  (best of 5, +1 warmup)\n", full_ms);
    fprintf(stderr, "noop_ms   = %.2f ms  (stack-alloc + DMA floor)\n", noop_ms);
    fprintf(stderr, "recurrence= full - noop = %.2f ms\n", rec_ms);
    fprintf(stderr, "S-vec accesses (passA-read + passB-write) = %.0f\n", SVEC);
    fprintf(stderr, "per-vec S-stack latency = %.2f ns/vec\n", per_vec_ns);
    fprintf(stderr, "full checksum: finite %d/48, range [%.3f, %.3f]%s\n",
            finite, mn, mx, (finite == NVH && mn != mx) ? " (recurrence ran, varies per v-head — OK)" : " (SUSPECT)");
    fprintf(stderr, "--- decision ---\n");
    if (per_vec_ns < 250.0)
        fprintf(stderr, "FAST: %.2f ns/vec (< 250) => <=8KB stack pipelines under the real recurrence.\n"
                        "      Slab structure VIABLE -> proceed to the 4-tile bf16-S engine kernel.\n", per_vec_ns);
    else
        fprintf(stderr, "SLOW: %.2f ns/vec (>= 250) => <=8KB stack degrades toward the >=16KB ~1.3us/vec rate.\n"
                        "      Impasse stands -> GDN M=K on-tile-state DEAD; live lever = bf16-S on shipped row-stream.\n", per_vec_ns);
    return 0;
}