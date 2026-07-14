// tools/gdn_chunkwise_probe.cpp — host probe for the 1-tile chunkwise M=K GDN
// kernel (fst_gdn_chunkwise.xclbin, Stage 1.3 standalone).  Reads toy input
// written by scripts/gdn_chunkwise_probe_ref.py (kernels/inp_s0.bin,
// kernels/inp_par.bin), dispatches the single 1-worker op (2 MM2S + 2 S2MM,
// K=8 tokens × 48 v-heads), writes kernels/out_y.bin + kernels/out_sf.bin, and
// reports dispatch latency (chunkwise premise: ~tens of ms, NOT 60 s).
//
// Compare against the numpy reference with:
//   python3 scripts/gdn_chunkwise_probe_ref.py cmp
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

constexpr int HV = 128, K = 8, NVH = 48, PKT_PAR = 392;
constexpr size_t N_S0  = (size_t)NVH * HV * HV;        // 786432 fp32
constexpr size_t N_PAR = (size_t)NVH * K * PKT_PAR;    // 150528 fp32
constexpr size_t N_Y   = (size_t)NVH * K * HV;         // 49152 fp32
constexpr size_t N_SF  = (size_t)NVH * HV * HV;        // 786432 fp32

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
    std::string DX = std::string(dir) + "/fst_gdn_chunkwise.xclbin";
    std::string IX = std::string(dir) + "/fst_gdn_chunkwise_insts.bin";
    std::string FS0 = std::string(dir) + "/inp_s0.bin";
    std::string FPA = std::string(dir) + "/inp_par.bin";
    std::string FOY = std::string(dir) + "/out_y.bin";
    std::string FOS = std::string(dir) + "/out_sf.bin";

    auto vs0 = read_file(FS0), vpa = read_file(FPA);
    if (vs0.size() != N_S0 * 4 || vpa.size() != N_PAR * 4) {
        fprintf(stderr, "[probe] input size mismatch: s0=%zu par=%zu (expected %zu %zu)\n",
                vs0.size(), vpa.size(), N_S0 * 4, N_PAR * 4); return 4; }

    xrt::device dev(0);
    XclbinKern kX = load_xclbin(dev, DX, read_file(IX));

    xrt::ext::bo boS0(dev, N_S0 * 4), boPar(dev, N_PAR * 4);
    xrt::ext::bo boY(dev, N_Y * 4), boSf(dev, N_SF * 4);
    { auto* p = boS0.map<float*>();  memcpy(p, vs0.data(), vs0.size()); boS0.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boPar.map<float*>(); memcpy(p, vpa.data(), vpa.size()); boPar.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boY.map<float*>();   memset(p, 0, N_Y * 4);  boY.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boSf.map<float*>();  memset(p, 0, N_SF * 4);  boSf.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    fprintf(stderr, "[probe] dispatch 1-tile chunkwise M=K GDN (K=8, 48 v-heads, 1 dispatch) x3...\n");
    double ms = 0.0;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        auto run = kX.krnl(3, 0, 0,
                           static_cast<xrt::bo&>(boS0), static_cast<xrt::bo&>(boPar),
                           static_cast<xrt::bo&>(boY),  static_cast<xrt::bo&>(boSf));
        if (run.wait(60000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] DISPATCH FAILED/TIMEOUT\n"); return 3; }
        double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[probe] iter %d latency = %.2f ms\n", it, d);
        if (it == 2) ms = d;   // report steady-state (3rd)
    }

    boY.sync(XCL_BO_SYNC_BO_FROM_DEVICE); boSf.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    { auto* p = boY.map<float*>();  std::ofstream f(FOY, std::ios::binary); f.write((char*)p, N_Y * 4); }
    { auto* p = boSf.map<float*>(); std::ofstream f(FOS, std::ios::binary); f.write((char*)p, N_SF * 4); }
    fprintf(stderr, "[probe] wrote %s + %s  (latency %.2f ms, %.3f ms/token/layer)\n",
            FOY.c_str(), FOS.c_str(), ms, ms / K);
    return 0;
}