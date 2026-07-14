// tools/gdn_chunkwise_8tile_probe.cpp — host probe for the 8-TILE column-split
// chunkwise M=K GDN kernel (fst_gdn_chunkwise_8tile.xclbin, Stage 1.2-v2).
//
// 2 BOs only (unified f_in / f_out streams, 328-float packets):
//   boIn  = kernels/inp_8t.bin   (packed by scripts/gdn_chunkwise_8tile_ref.py)
//   boOut = kernels/out_8t.bin   (raw 328-packet drain, unpacked by the ref)
//
// Layout (must match gen_gdn_chunkwise_8tile.py runtime): each BO holds
// [chain A half | chain B half]; each chain half = 48 v-heads × (128 + 8)
// packets × 328 floats.  Chain B offset = 48*136*328 floats.
//
// Dispatches krnl(3, 0, 0, boIn, boOut) x3, reports steady-state latency, writes
// out_8t.bin.  Compare with: python3 scripts/gdn_chunkwise_8tile_ref.py cmp
#include <chrono>
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

constexpr int HV = 128, K = 8, NVH = 48, PKT = 328, PER_VH = HV + K;  // 136
constexpr size_t N_CHAIN = (size_t)NVH * PER_VH;           // 6528 packets/chain
constexpr size_t N_PKT   = 2 * N_CHAIN;                     // 13056 packets total
constexpr size_t N_FLT   = N_PKT * PKT;                     // 4,286,208 fp32 (~17MB)

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
    std::string DX  = std::string(dir) + "/fst_gdn_chunkwise_8tile.xclbin";
    std::string IX  = std::string(dir) + "/fst_gdn_chunkwise_8tile_insts.bin";
    std::string FIN = std::string(dir) + "/inp_8t.bin";
    std::string FOUT= std::string(dir) + "/out_8t.bin";

    auto vin = read_file(FIN);
    if (vin.size() != N_FLT * 4) {
        fprintf(stderr, "[probe] inp_8t size %zu != expected %zu\n", vin.size(), N_FLT * 4); return 4; }

    xrt::device dev(0);
    XclbinKern kX = load_xclbin(dev, DX, read_file(IX));

    xrt::ext::bo boIn(dev, N_FLT * 4), boOut(dev, N_FLT * 4);
    { auto* p = boIn.map<float*>();  memcpy(p, vin.data(), vin.size()); boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boOut.map<float*>(); memset(p, 0, N_FLT * 4); boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    fprintf(stderr, "[probe] dispatch 8-tile chunkwise M=K GDN (K=8, 48 v-heads, 2 chains x 4 tiles) x3...\n");
    double ms = 0.0;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        auto run = kX.krnl(3, 0, 0, static_cast<xrt::bo&>(boIn), static_cast<xrt::bo&>(boOut));
        if (run.wait(60000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] DISPATCH FAILED/TIMEOUT\n"); return 3; }
        double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[probe] iter %d latency = %.2f ms\n", it, d);
        if (it == 2) ms = d;
    }

    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    { auto* p = boOut.map<float*>(); std::ofstream f(FOUT, std::ios::binary); f.write((char*)p, N_FLT * 4); }
    fprintf(stderr, "[probe] wrote %s  (steady latency %.2f ms, %.3f ms/token/layer over K=%d)\n",
            FOUT.c_str(), ms, ms / K, K);
    return 0;
}