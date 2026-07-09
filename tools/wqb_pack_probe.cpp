// tools/wqb_pack_probe.cpp — definitive test: can ONE instruction blob issue
// MULTIPLE core runs on the existing fst_mla_wqb.xclbin?
//
// Strategy: replicate the wqb op-words NREP times in one blob, patching each
// copy's bd1 (weight, arg_idx=1) and bd2 (output, arg_idx=2) DDR_PATCH arg_off
// to that tile's offset, so one xrt::run computes NREP N-tiles. Compare against
// NREP separate single dispatches (weight via xrt::bo sub-buffer, as the engine
// does). If the packed output matches the per-dispatch output -> core re-triggers
// within one dispatch -> packing works. If it hangs (syncobj timeout) -> the
// core is single-pass-per-dispatch and packing on existing kernels is impossible.
//
// Build:
//   g++ -std=c++17 -O2 -I. -ISource/FastFlowLM-main/src/include \
//       -o tools/wqb_pack_probe tools/wqb_pack_probe.cpp \
//       -lxrt_coreutil -lxrt_core -luuid -laiebu -lpthread -ldl
// Run:
//   XILINX_XRT=/usr ./tools/wqb_pack_probe

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_ext.h>
#include "aiebu/aiebu.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

using bf16_t = uint16_t;
static float bf16f(bf16_t x){ uint32_t u=((x&0x8000)<<16)|((x&0x7fff)<<13); float f; std::memcpy(&f,&u,4); return f; }
static bf16_t f2bf(float f){ uint32_t u; std::memcpy(&u,&f,4); uint32_t r=(u&0x80000000)>>16, e=(u>>23)&0xff, m=(u>>13)&0x3ff, t=(u>>23)&1; if(e<113){ r|=(e<112)?0u:((0x4000+m)>>1); } else if(e>142){ r|=0x7fff; } else { r|=((e-112)<<10)|m; t&=0; if(t) r++; } return (bf16_t)r; }

static std::vector<uint32_t> load_insts(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> w(sz/4);
    f.read(reinterpret_cast<char*>(w.data()), sz);
    return w;
}

// Build an aiebu ELF from a blob and wrap as xrt::ext::kernel on ctx.
static xrt::ext::kernel build_kernel(xrt::hw_context& ctx, const std::vector<uint32_t>& blob,
                                     const std::string& kname) {
    char* elf_buf = nullptr;
    uint32_t esz = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        (char*)blob.data(), (uint32_t)(blob.size()*sizeof(uint32_t)),
        NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
    if (esz == 0 || !elf_buf) { std::fprintf(stderr, "aiebu_get_elf failed\n"); std::exit(1); }
    xrt::elf elf(elf_buf, esz);
    xrt::module mod(elf);
    free(elf_buf);
    return xrt::ext::kernel(ctx, mod, kname);
}

// Replicate the op-words of a base blob nrep times, patching each copy's DDR_PATCH
// arg_off for bd1 (arg_idx==1 -> w_offs[n]) and bd2 (arg_idx==2 -> o_offs[n]).
// Header: 4 words (keep w[0],w[1]; cmds=nrep*base_cmds; total bytes=nrep*op_words*4+16).
static std::vector<uint32_t> replicate_packed(const std::vector<uint32_t>& base,
                                              int nrep,
                                              const std::vector<uint32_t>& w_offs,
                                              const std::vector<uint32_t>& o_offs) {
    size_t hdr = 4;
    size_t op_words = base.size() - hdr;          // 101 for wqb
    uint32_t base_cmds = base[2];
    std::vector<uint32_t> out;
    out.reserve(hdr + nrep * op_words);
    out.push_back(base[0]); out.push_back(base[1]);
    out.push_back(base_cmds * (uint32_t)nrep);    // instr_counts
    uint32_t total_words = (uint32_t)(hdr + nrep * op_words);
    out.push_back(total_words * 4);               // total bytes (seq[3])
    for (int r = 0; r < nrep; ++r) {
        size_t base_i = hdr;
        // walk ops in the base copy, patching DDR_PATCH arg_off for bd1/bd2
        size_t i = base_i;
        while (i < base.size()) {
            uint32_t op = base[i];
            size_t lines;
            if (op == 1) lines = 12;          // BLOCKWRITE
            else if (op == 0x81) lines = 12;  // DDR_PATCH
            else if (op == 0x80) lines = 4;   // WAIT
            else if (op == 3) lines = 7;      // ISSUE_TOK
            else if (op == 0) lines = 6;      // WRITE
            else { lines = 0; break; }
            for (size_t k = 0; k < lines; ++k) {
                uint32_t word = base[i + k];
                if (op == 0x81) {  // DDR_PATCH: arg_idx at +8, arg_off at +10
                    uint32_t arg_idx = base[i + 8];
                    if (k == 10) {
                        if (arg_idx == 1) word = w_offs[r];
                        else if (arg_idx == 2) word = o_offs[r];
                    }
                }
                out.push_back(word);
            }
            i += lines;
        }
    }
    return out;
}

int main() {
    const char* xclbin_path = "fst_mla_wqb.xclbin";
    const char* insts  = "fst_mla_wqb_insts.bin";
    const int NREP = 2;
    const int M_c = 16, K_c = 1024, N_c = 2048;
    const size_t a_b = (size_t)M_c * K_c * sizeof(bf16_t);      // 32 KiB
    const size_t b_tile_b = (size_t)N_c * K_c * sizeof(bf16_t); // 4 MiB
    const size_t c_tile_b = (size_t)M_c * N_c * sizeof(bf16_t); // 64 KiB
    const size_t w_step = (size_t)N_c * K_c * sizeof(bf16_t);   // weight N-slice stride (bcol: 2048 cols * 1024 rows * 2)

    xrt::device dev(0);
    std::string xbpath(xclbin_path);
    xrt::xclbin xb(xbpath);
    dev.register_xclbin(xb);
    std::string kname;
    for (auto& k : xb.get_kernels()) if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kname = k.get_name(); break; }
    std::printf("kernel name: %s\n", kname.c_str());
    xrt::hw_context ctx(dev, xb.get_uuid());

    auto base = load_insts(insts);
    std::printf("base insts: %zu words, cmds=%u\n", base.size(), base[2]);

    // BOs
    xrt::bo bo_A(dev, a_b, xrt::bo::flags::host_only, 0);
    xrt::bo bo_W(dev, NREP * b_tile_b, xrt::bo::flags::host_only, 0);   // 2 weight tiles contiguous
    xrt::bo bo_Cbig(dev, NREP * c_tile_b, xrt::bo::flags::host_only, 0);// 2 output tiles contiguous
    xrt::bo bo_d1(dev, 1<<20, xrt::bo::flags::host_only, 0), bo_d2(dev, 1<<20, xrt::bo::flags::host_only, 0); // scratch

    // Fill A and W with random bf16
    srand(12345);
    auto* A = bo_A.map<bf16_t*>(); for (size_t i=0;i<a_b/2;i++) A[i]=f2bf((float)((rand()%2000)-1000)/100.0f);
    auto* W = bo_W.map<bf16_t*>(); for (size_t i=0;i<NREP*b_tile_b/2;i++) W[i]=f2bf((float)((rand()%2000)-1000)/100.0f);
    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto krnl_single = build_kernel(ctx, base, kname);

    // ---- Baseline: NREP separate single dispatches (weight via sub-buffer) ----
    std::vector<std::vector<bf16_t>> base_out(NREP);
    for (int r = 0; r < NREP; ++r) {
        xrt::bo w_sub(bo_W, b_tile_b, (size_t)r * b_tile_b);   // sub-buffer at tile offset
        {auto* c = bo_Cbig.map<bf16_t*>(); std::memset(c,0,NREP*c_tile_b);}
        bo_Cbig.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = krnl_single(3,0,0, static_cast<xrt::bo&>(bo_A), static_cast<xrt::bo&>(w_sub),
                               static_cast<xrt::bo&>(bo_Cbig), static_cast<xrt::bo&>(bo_d1), static_cast<xrt::bo&>(bo_d2));
        run.wait();
        bo_Cbig.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto* c = bo_Cbig.map<bf16_t*>();
        base_out[r].resize(c_tile_b/2);
        std::memcpy(base_out[r].data(), c, c_tile_b);   // tile r at offset 0 (single writes to offset 0)
    }
    std::printf("baseline (single dispatches) done\n");

    // ---- Packed: 1 dispatch, NREP-copy blob with arg_off patches ----
    std::vector<uint32_t> w_offs(NREP), o_offs(NREP);
    for (int r = 0; r < NREP; ++r) { w_offs[r] = (uint32_t)(r * b_tile_b); o_offs[r] = (uint32_t)(r * c_tile_b); }
    auto packed = replicate_packed(base, NREP, w_offs, o_offs);
    std::printf("packed blob: %zu words, cmds=%u\n", packed.size(), packed[2]);
    auto krnl_packed = build_kernel(ctx, packed, kname);

    {auto* c = bo_Cbig.map<bf16_t*>(); std::memset(c,0,NREP*c_tile_b);}
    bo_Cbig.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t0 = std::chrono::steady_clock::now();
    auto run = krnl_packed(3,0,0, static_cast<xrt::bo&>(bo_A), static_cast<xrt::bo&>(bo_W),
                           static_cast<xrt::bo&>(bo_Cbig), static_cast<xrt::bo&>(bo_d1), static_cast<xrt::bo&>(bo_d2));
    run.wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("packed dispatch returned in %.2f ms\n", ms);
    bo_Cbig.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* c = bo_Cbig.map<bf16_t*>();

    // Compare per-tile
    int bad = 0; double maxrel = 0;
    for (int r = 0; r < NREP; ++r) {
        bf16_t* pc = (bf16_t*)((char*)c + r * c_tile_b);
        for (size_t i = 0; i < c_tile_b/2; ++i) {
            float a = bf16f(pc[i]), b = bf16f(base_out[r][i]);
            float rel = (b!=0)? std::fabs(a-b)/std::fabs(b) : std::fabs(a);
            if (rel > 1e-2f && std::fabs(a-b) > 1e-3f) { bad++; maxrel = std::max(maxrel,(double)rel); }
        }
    }
    std::printf("RESULT: %d mismatches, maxrel=%.5f  -> %s\n", bad, maxrel,
                bad==0 ? "PACKING WORKS (1 blob == NREP dispatches)" :
                         (bad>0 ? "MISMATCH (see maxrel)" : "?"));
    return 0;
}