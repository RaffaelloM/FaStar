// tools/gdn_4tile_probe.cpp — host probe for the 4-tile 8 KB-stack micro-test
// (fst_gdn_chunkwise_4tile_micro.xclbin).  Packs 48 v-heads of input
// ([gdec=0.5, ro0=0, ro1=32, ro2=64, ro3=96, pad]), dispatches the single
// 4-worker op (1 MM2S + 1 S2MM, 3-hop core↔core chain), reads the total BO,
// verifies total == 4128.0, and reports dispatch latency (chunkwise speed
// premise: ~30 ms, NOT 60 s).
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

constexpr int PKT = 8;
constexpr int NV  = 48;

struct XclbinKern { xrt::hw_context ctx; xrt::ext::kernel krnl; };

static XclbinKern load_xclbin(xrt::device& dev, const std::string& xclb_path,
                              const std::vector<char>& insts) {
    xrt::xclbin xclb(xclb_path);
    dev.register_xclbin(xclb);
    xrt::uuid uid = xclb.get_uuid();
    auto ctxp = std::make_unique<xrt::hw_context>(dev, uid);
    char* elf_buf = nullptr;
    uint32_t elf_sz = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(), (uint32_t)insts.size(),
        NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
    if (elf_sz == 0 || !elf_buf) { fprintf(stderr, "[probe] aiebu_get_elf FAILED\n"); exit(2); }
    xrt::elf elf(elf_buf, elf_sz); xrt::module mod(elf); free(elf_buf);
    std::string kname; for (auto& k : xclb.get_kernels()) kname = k.get_name();
    fprintf(stderr, "[probe] %s kernel=%s (elf %u B)\n", xclb_path.c_str(), kname.c_str(), elf_sz);
    xrt::ext::kernel krnl(*ctxp, mod, kname);
    return {std::move(*ctxp), std::move(krnl)};
}
static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[probe] MISSING %s\n", path.c_str()); exit(5); }
    std::vector<char> d((size_t)f.tellg()); f.seekg(0); f.read(d.data(), d.size()); return d;
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : ".";
    std::string DX = std::string(dir) + "/fst_gdn_chunkwise_4tile_micro.xclbin";
    std::string IX = std::string(dir) + "/fst_gdn_chunkwise_4tile_micro_insts.bin";

    // in0[vh*8] = [gdec=0.5, ro0=0, ro1=32, ro2=64, ro3=96, pad, pad, pad]
    std::vector<float> in0((size_t)NV * PKT, 0.0f);
    for (int vh = 0; vh < NV; vh++) {
        in0[vh * PKT + 0] = 0.5f;
        in0[vh * PKT + 1] = 0.0f;
        in0[vh * PKT + 2] = 32.0f;
        in0[vh * PKT + 3] = 64.0f;
        in0[vh * PKT + 4] = 96.0f;
    }

    xrt::device dev(0);
    XclbinKern kX = load_xclbin(dev, DX, read_file(IX));

    xrt::ext::bo boIn(dev, (size_t)NV * PKT * 4);
    xrt::ext::bo boOut(dev, (size_t)NV * PKT * 4);
    { auto* p = boIn.map<float*>();  memcpy(p, in0.data(), in0.size() * 4); boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boOut.map<float*>(); memset(p, 0, (size_t)NV * PKT * 4);    boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    fprintf(stderr, "[probe] dispatch 4-tile 8KB-stack micro-test (48 v-heads)...\n");
    auto t0 = std::chrono::steady_clock::now();
    auto run = kX.krnl(3, 0, 0, static_cast<xrt::bo&>(boIn), static_cast<xrt::bo&>(boOut));
    if (run.wait(60000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] DISPATCH FAILED/TIMEOUT\n"); return 3; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float* out = boOut.map<float*>();

    int nbad = 0; float worst = 0.0f; int worst_vh = -1;
    for (int vh = 0; vh < NV; vh++) {
        float total = out[vh * PKT + 0];
        float d = std::fabs(total - 4128.0f);
        if (d > 1e-3f) { nbad++; if (d > worst) { worst = d; worst_vh = vh; } }
        if (vh < 4 || d > 1e-3f)
            fprintf(stderr, "[probe] vh %2d total=%g (Δ=%g)\n", vh, total, d);
    }
    fprintf(stderr, "[probe] dispatch latency = %.2f ms\n", ms);
    if (nbad == 0) {
        fprintf(stderr, "[probe] PASS: all 48 v-heads total == 4128.0  (8KB stack RMW + 3-hop core<->core chain OK)\n");
        return 0;
    }
    fprintf(stderr, "[probe] FAIL: %d/48 v-heads wrong (worst Δ=%g at vh %d)\n", nbad, worst, worst_vh);
    return 1;
}