// tools/qwopus_ffn_noop.cpp — Stage-1.1 micro-probe for the NPU FFN lever.
//
// Validates + times the 1-tile M=1 MXFP4 matvec (fst_qwopus_ffn_noop.xclbin):
//   out[n] = sum_k W[n,k]·h[k],  W packed MXFP4 (17-byte blocks padded to 18),
//   n = 0..N_TILE-1 (2176 = 17408/8 rows of the gate weight), K=5120.
//
// Synthesizes N_TILE rows of random MXFP4 + a random h, packs the weight as
// bytes-reinterpreted-as-float (the IRON uint8-as-2nd-input no-DMA workaround),
// dispatches krnl(3,0,0,boH,boW,boOut) x3, times the steady-state, compares to
// the host fp32 matvec reference (the same math as mxfp4_matvec_f32), and
// reports:
//   - max|Δout| (rel) vs host ref  -> correctness (gate < ~1e-4; a bug is O(1))
//   - per-tile weight-read throughput (MB/s) -> the make-or-break bandwidth
//     number.  >= ~1.5 GB/s/tile => 8-tile aggregate >= ~12 GB/s => FFN-NPU
//     viable (beats host ~4.7 GB/s).  ~200 MB/s/tile => dead-end.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr int HV_K      = 5120;
constexpr int GROUPS    = HV_K / 32;        // 160
constexpr int PAD_BYTES = 18;
constexpr int ROW_BYTES = GROUPS * PAD_BYTES;   // 2880
constexpr int RPB       = 4;
constexpr int N_TILE    = 2176;             // 17408 / 8
constexpr int NPKT      = N_TILE / RPB;     // 544
constexpr int WPKT      = RPB * ROW_BYTES;  // 11520 B
constexpr int H_BYTES   = HV_K * 4;         // 20480

static const float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

struct XclbinKern { xrt::hw_context ctx; xrt::ext::kernel krnl; };
static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[probe] MISSING %s\n", path.c_str()); exit(5); }
    std::vector<char> d((size_t)f.tellg()); f.seekg(0); f.read(d.data(), d.size()); return d;
}
static XclbinKern load_xclbin(xrt::device& dev, const std::string& xp, const std::vector<char>& insts) {
    xrt::xclbin xclb(xp); dev.register_xclbin(xclb); xrt::uuid uid = xclb.get_uuid();
    auto ctxp = std::make_unique<xrt::hw_context>(dev, uid);
    char* eb = nullptr;
    uint32_t es = aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(), (uint32_t)insts.size(), NULL, 0, (void**)&eb, NULL, 0, "", "", NULL, 0);
    if (!es || !eb) { fprintf(stderr, "[probe] aiebu_get_elf FAILED\n"); exit(2); }
    xrt::elf elf(eb, es); xrt::module mod(elf); free(eb);
    std::string kn; for (auto& k : xclb.get_kernels()) kn = k.get_name();
    fprintf(stderr, "[probe] %s kernel=%s (elf %u B)\n", xp.c_str(), kn.c_str(), es);
    xrt::ext::kernel krnl(*ctxp, mod, kn);
    return {std::move(*ctxp), std::move(krnl)};
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : "kernels";
    std::string DX = std::string(dir) + "/fst_qwopus_ffn_noop.xclbin";
    std::string IX = std::string(dir) + "/fst_qwopus_ffn_noop_insts.bin";

    // ---- synthesize N_TILE rows of random MXFP4 + random h ----
    std::vector<uint8_t> wraw((size_t)N_TILE * ROW_BYTES, 0);
    std::vector<float>   h(HV_K);
    unsigned rs = 20260713u;
    auto rnd = [&]() { rs = rs * 1664525u + 1013904223u; return rs; };
    for (int n = 0; n < N_TILE; n++) {
        uint8_t* row = wraw.data() + (size_t)n * ROW_BYTES;
        for (int g = 0; g < GROUPS; g++) {
            uint8_t* blk = row + (size_t)g * PAD_BYTES;
            blk[0] = (uint8_t)(120 + (rnd() & 15));        // e8m0 scale in [120,135]
            for (int i = 0; i < 16; i++) blk[1 + i] = (uint8_t)(rnd() & 0xFF);
        }
    }
    for (int k = 0; k < HV_K; k++) {
        float f = ((int)((rnd() >> 8) % 2000) - 1000) * 0.001f;
        h[k] = f;
    }

    // ---- host fp32 matvec reference (matches mxfp4_matvec_f32) ----
    std::vector<float> ref(N_TILE, 0.0f);
    for (int n = 0; n < N_TILE; n++) {
        const uint8_t* row = wraw.data() + (size_t)n * ROW_BYTES;
        float acc = 0.0f;
        for (int g = 0; g < GROUPS; g++) {
            const uint8_t* blk = row + (size_t)g * PAD_BYTES;
            uint8_t sc = blk[0];
            if (sc == 0) continue;
            float scale = std::ldexp(1.0f, (int)sc - 127);
            const uint8_t* nb = blk + 1;
            for (int i = 0; i < 32; i++) {
                uint8_t byte = nb[i >> 1];
                uint8_t nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                acc += FP4_LUT[nib] * scale * h[g * 32 + i];
            }
        }
        ref[n] = acc;
    }

    // ---- pack inp_w: bytes reinterpreted as float (544 × 11520 B = 6.26 MB) ----
    const size_t W_TOTAL = (size_t)NPKT * WPKT;            // 6266880 B
    std::vector<float>   inp_w(W_TOTAL / 4);
    std::vector<float>   inp_h(HV_K);
    memcpy(inp_h.data(), h.data(), H_BYTES);
    for (int p = 0; p < NPKT; p++)
        memcpy((uint8_t*)inp_w.data() + (size_t)p * WPKT,
               wraw.data() + (size_t)p * WPKT, WPKT);

    xrt::device dev(0);
    XclbinKern kX = load_xclbin(dev, DX, read_file(IX));

    xrt::ext::bo boH(dev, H_BYTES);
    xrt::ext::bo boW(dev, W_TOTAL);
    xrt::ext::bo boO(dev, (size_t)N_TILE * 4);
    { auto* p = boH.map<float*>(); memcpy(p, inp_h.data(), H_BYTES); boH.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boW.map<float*>(); memcpy(p, inp_w.data(), W_TOTAL); boW.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boO.map<float*>(); memset(p, 0, (size_t)N_TILE * 4); boO.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    fprintf(stderr, "[probe] dispatch 1-tile M=1 MXFP4 matvec (%d rows, %d pkts, %.2f MB weight) x3...\n",
            N_TILE, NPKT, (double)W_TOTAL / 1e6);
    double ms = 0.0;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        auto run = kX.krnl(3, 0, 0, static_cast<xrt::bo&>(boH),
                                    static_cast<xrt::bo&>(boW),
                                    static_cast<xrt::bo&>(boO));
        if (run.wait(60000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] DISPATCH FAILED/TIMEOUT\n"); return 3; }
        double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[probe] iter %d latency = %.2f ms\n", it, d);
        if (it == 2) ms = d;
    }
    boO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float* out = boO.map<float*>();

    // ---- compare ----
    double maxd = 0; int wn = -1;
    int arg_ref = 0, arg_npu = 0; float vref = -1e30f, vnpu = -1e30f;
    for (int n = 0; n < N_TILE; n++) {
        double r = std::fabs((double)out[n] - ref[n]) / std::max(std::fabs((double)ref[n]), 1.0);
        if (r > maxd) { maxd = r; wn = n; }
        if (ref[n] > vref) { vref = ref[n]; arg_ref = n; }
        if (out[n] > vnpu) { vnpu = out[n]; arg_npu = n; }
    }
    double gb_s = ((double)W_TOTAL / 1e9) / (ms / 1e3);     // per-tile weight-read GB/s
    fprintf(stderr, "[probe] max|Δout|=%.3e (row %d)  argmax ref=%d npu=%d\n", maxd, wn, arg_ref, arg_npu);
    fprintf(stderr, "[probe] per-tile weight-read throughput = %.2f GB/s  (x8 aggregate ~= %.2f GB/s)\n",
            gb_s, gb_s * 8);
    bool ok = (maxd < 1e-4) && (arg_ref == arg_npu);
    printf("maxd=%.3e argmax_ok=%d per_tile=%.2fGB/s agg8=%.2fGB/s steady=%.2fms %s\n",
           maxd, (int)(arg_ref == arg_npu), gb_s, gb_s * 8, ms, ok ? "PASS" : "FAIL");
    if (ok) { fprintf(stderr, "[probe] PASS: matvec bit-correct (rel <1e-4), %.2f GB/s/tile -> 8-tile ~%.2f GB/s (host FFN ~4.7 GB/s)\n",
                      gb_s, gb_s * 8); return 0; }
    fprintf(stderr, "[probe] FAIL (maxd=%.3e, argmax ref=%d npu=%d)\n", maxd, arg_ref, arg_npu);
    return 4;
}