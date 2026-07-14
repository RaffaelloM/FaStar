// tools/gdn_cross_probe.cpp — minimal host probe for the core↔core ObjectFifo
// micro-test (fst_gdn_cross_micro.xclbin).  Packs 48 v-heads of input
// ([gdec=0.5, row_offset, pad]), dispatches the single 2-worker op, reads the
// total BO, and verifies total == 4128.0 for every v-head.
//
// Tests ONLY the worker→worker ObjectFifo (W0.prod -> W1.cons) + the per-tile
// 1 MM2S + <=1 S2MM shim pattern.  Pass = all 48 totals == 4128.0 (fp32-exact).
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
    // report kernel arg count to confirm dispatch convention
    for (auto& k : xclb.get_kernels()) {
        auto args = k.get_args();
        fprintf(stderr, "[probe]   kernel '%s' args=%zu\n", k.get_name().c_str(), args.size());
        for (size_t i = 0; i < args.size(); i++)
            fprintf(stderr, "[probe]     [%zu] %s group_id=%u\n", i, args[i].get_name().c_str(), args[i].get_index());
    }
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
    std::string DX = std::string(dir) + "/fst_gdn_cross_micro.xclbin";
    std::string IX = std::string(dir) + "/fst_gdn_cross_micro_insts.bin";

    // in_w0[vh*8] = [0.5, 0, 0,0,0,0,0,0] ; in_w1[vh*8] = [0.5, 64, 0,...]
    std::vector<float> in_w0((size_t)NV * PKT, 0.0f);
    std::vector<float> in_w1((size_t)NV * PKT, 0.0f);
    for (int vh = 0; vh < NV; vh++) {
        in_w0[vh * PKT + 0] = 0.5f; in_w0[vh * PKT + 1] = 0.0f;
        in_w1[vh * PKT + 0] = 0.5f; in_w1[vh * PKT + 1] = 64.0f;
    }

    xrt::device dev(0);
    XclbinKern kX = load_xclbin(dev, DX, read_file(IX));

    xrt::ext::bo boIn0(dev, (size_t)NV * PKT * 4);
    xrt::ext::bo boIn1(dev, (size_t)NV * PKT * 4);
    xrt::ext::bo boOut(dev, (size_t)NV * PKT * 4);
    { auto* p = boIn0.map<float*>(); memcpy(p, in_w0.data(), in_w0.size() * 4); }
    { auto* p = boIn1.map<float*>(); memcpy(p, in_w1.data(), in_w1.size() * 4); }
    { auto* p = boOut.map<float*>(); memset(p, 0, (size_t)NV * PKT * 4); }
    boIn0.sync(XCL_BO_SYNC_BO_TO_DEVICE); boIn1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    fprintf(stderr, "[probe] dispatch core<->core micro-test (48 v-heads)...\n");
    auto run = kX.krnl(3, 0, 0, static_cast<xrt::bo&>(boIn0),
                                static_cast<xrt::bo&>(boIn1),
                                static_cast<xrt::bo&>(boOut));
    if (run.wait(30000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] DISPATCH FAILED/TIMEOUT\n"); return 3; }
    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float* out = boOut.map<float*>();

    // expected total = 0.5 * (1+..+128) = 4128.0 for every v-head
    int nbad = 0; float worst = 0.0f; int worst_vh = -1;
    for (int vh = 0; vh < NV; vh++) {
        float total = out[vh * PKT + 0];
        float d = std::fabs(total - 4128.0f);
        if (d > 1e-4f) { nbad++; if (d > worst) { worst = d; worst_vh = vh; } }
        if (vh < 4 || d > 1e-4f)
            fprintf(stderr, "[probe] vh %2d total=%g (Δ=%g)\n", vh, total, d);
    }
    if (nbad == 0) {
        fprintf(stderr, "[probe] PASS: all 48 v-heads total == 4128.0  (core<->core ObjectFifo WORKS)\n");
        return 0;
    }
    fprintf(stderr, "[probe] FAIL: %d/48 v-heads wrong (worst Δ=%g at vh %d)\n", nbad, worst, worst_vh);
    return 1;
}