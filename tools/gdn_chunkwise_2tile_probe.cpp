// tools/gdn_chunkwise_2tile_probe.cpp — host probe for the 2-TILE column-split
// single-call stack-S chunkwise M=K GDN kernel (fst_gdn_chunkwise_2tile.xclbin).
//
// 2 BOs (bf16 unified per-v-head packets):
//   boIn  = kernels/inp_2t.bin   (packed by scripts/gdn_chunkwise_2tile_ref.py)
//   boOut = kernels/out_2t.bin   (unpacked by the ref)
//
// Layout (matches gen_gdn_chunkwise_2tile.py): each BO = [tile0 half | tile1
// half]; tile half = 48 v-heads × IN_PKT(10768) bf16 [in] / OUT_PKT(8704) [out].
// Tile1 offset = 48*10768 (in) / 48*8704 (out).
//
// Dispatches krnl(3,0,0, boIn, boOut) x3, reports steady-state latency, writes
// out_2t.bin.  Compare with: python3 scripts/gdn_chunkwise_2tile_ref.py cmp
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

constexpr int HV=128, K=8, NVH=48, COLS=64, IN_PAR=322;
constexpr size_t IN_PKT  = HV*COLS + K*IN_PAR;     // 10768
constexpr size_t OUT_PKT = K*COLS + HV*COLS;        // 8704
constexpr size_t HALF   = NVH * IN_PKT;             // 48 × 10768 bf16 (tile0 in)
constexpr size_t HALF_O = NVH * OUT_PKT;            // 48 × 8704  bf16 (tile0 out)
constexpr size_t N_IN   = 2 * HALF;                 // both tiles
constexpr size_t N_OUT  = 2 * HALF_O;

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
    std::string DX  = std::string(dir) + "/fst_gdn_chunkwise_2tile.xclbin";
    std::string IX  = std::string(dir) + "/fst_gdn_chunkwise_2tile_insts.bin";
    std::string FIN = std::string(dir) + "/inp_2t.bin";
    std::string FOUT= std::string(dir) + "/out_2t.bin";

    auto vin = read_file(FIN);
    if (vin.size() != N_IN * 2) {   // bf16 = 2 bytes
        fprintf(stderr, "[probe] inp_2t size %zu != expected %zu\n", vin.size(), N_IN * 2); return 4; }

    xrt::device dev(0);
    XclbinKern kX = load_xclbin(dev, DX, read_file(IX));

    xrt::ext::bo boIn(dev, N_IN * 2), boOut(dev, N_OUT * 2);
    { auto* p = boIn.map<char*>();  memcpy(p, vin.data(), vin.size()); boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boOut.map<char*>(); memset(p, 0, N_OUT * 2); boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    fprintf(stderr, "[probe] dispatch 2-tile stack-S chunkwise M=K GDN (K=8, 48 v-heads, 2 indep tiles) x3...\n");
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
    { auto* p = boOut.map<char*>(); std::ofstream f(FOUT, std::ios::binary); f.write(p, N_OUT * 2); }
    fprintf(stderr, "[probe] wrote %s  (steady latency %.2f ms, %.3f ms/token/layer over K=%d)\n",
            FOUT.c_str(), ms, ms / K, K);
    return 0;
}