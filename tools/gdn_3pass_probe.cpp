// tools/gdn_3pass_probe.cpp — hardware validation + latency of the ONE-XCLBIN
// 3-pass-sequential GDN kernel (fst_gdn_3pass.xclbin).
//
// Same 48-v-head distinct-data synthesis + per-v-head fp32 reference as
// tools/gdn_probe.cpp (so results are directly comparable), but packs into the
// one-kernel BO layout and dispatches ONE krnl call (1 xrt::run) instead of 3.
//
//   f_in_S : [48 × 32 × 1040] = 32 8-row blocks/v-head (16 passA + 16 passB),
//            each block = 8 rows × [S_row(128)|kn_i|qn_i]  (S0 read twice)
//   f_in_par: [48 × 386] = [v(128)|kn(128)|qn(128)|gdec|beta]
//   f_out  : [48 × 17 × 1024] = 1 y packet [y@0:128] + 16 S2 packets (8 rows × 128)
//
// PASS = max|Δy|<1e-4 AND max|ΔS|<1e-4 (rel) across all 48 v-heads — the same
// gate as the 3-pass probe.  Reports steady-state latency over 3 dispatches.
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

constexpr int HV = 128, NROWS = 128, NV = 48;
constexpr int RPB = 8, NBLK = NROWS / RPB;     // 16 blocks/pass
constexpr int SIN = HV + 2;                    // 130 = [S_row|kn_i|qn_i]
constexpr int SBLK = RPB * SIN;                // 1040
constexpr int PAR = 3 * HV + 2;                // 386
constexpr int OUTPKT = 1024;
constexpr int NS = NV * (2 * NBLK);            // 48*32 S-blocks
constexpr int NOUT = NV * (1 + NBLK);          // 48*17 out-packets

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
    std::string DX = std::string(dir) + "/fst_gdn_3pass.xclbin";
    std::string IX = std::string(dir) + "/fst_gdn_3pass_insts.bin";

    // ---- synthesize 48 v-heads of distinct data (same as gdn_probe) ----
    std::vector<float> S0((size_t)NV * HV * HV), kn((size_t)NV * HV), qn((size_t)NV * HV),
                        vv((size_t)NV * HV), gdec(NV), beta(NV);
    for (int vh = 0; vh < NV; vh++) {
        float s = 0.0001f * (vh + 1);
        for (int i = 0; i < HV; i++) {
            kn[vh*HV + i] = 0.001f * (i + 1) * (vh + 1);
            qn[vh*HV + i] = 0.001f * (i + 1) * (vh + 1) / std::sqrt(128.0f);
            vv [vh*HV + i] = 0.01f  * (i + 1) * (vh + 1);
            for (int j = 0; j < HV; j++) S0[vh*HV*HV + i*HV + j] = s * (i + 1) * (j + 1);
        }
        gdec[vh] = std::exp(-1.0 - 0.01 * vh);
        beta[vh] = 1.0f / (1.0f + std::exp(-(-0.5f + 0.01f * vh)));
    }

    // ---- pack f_in_S: [vh][blk 0..31][8 rows × 130] (blk 0..15 passA, 16..31 passB) ----
    std::vector<float> inS((size_t)NS * SBLK, 0.0f);
    for (int vh = 0; vh < NV; vh++)
        for (int blk = 0; blk < 2 * NBLK; blk++) {
            float* b = inS.data() + ((size_t)vh * (2 * NBLK) + blk) * SBLK;
            int base = (blk % NBLK) * RPB;          // passA and passB read the SAME S0 rows
            for (int r = 0; r < RPB; r++) {
                int i = base + r;
                float* row = b + (size_t)r * SIN;
                for (int j = 0; j < HV; j++) row[j] = S0[vh*HV*HV + i*HV + j];
                row[HV + 0] = kn[vh*HV + i];
                row[HV + 1] = qn[vh*HV + i];        // unused in passB, packed for the unified stream
            }
        }
    // ---- pack f_in_par: [vh][386] = [v|kn|qn|gdec|beta] ----
    std::vector<float> inP((size_t)NV * PAR, 0.0f);
    for (int vh = 0; vh < NV; vh++) {
        float* p = inP.data() + (size_t)vh * PAR;
        memcpy(p + 0,       vv.data() + vh*HV, HV * 4);   // v
        memcpy(p + HV,      kn.data() + vh*HV, HV * 4);   // kn
        memcpy(p + 2 * HV,  qn.data() + vh*HV, HV * 4);   // qn
        p[3 * HV + 0] = gdec[vh];
        p[3 * HV + 1] = beta[vh];
    }

    xrt::device dev(0);
    XclbinKern kX = load_xclbin(dev, DX, read_file(IX));

    xrt::ext::bo boS(dev, (size_t)NS * SBLK * 4);
    xrt::ext::bo boP(dev, (size_t)NV * PAR * 4);
    xrt::ext::bo boO(dev, (size_t)NOUT * OUTPKT * 4);
    { auto* p = boS.map<float*>(); memcpy(p, inS.data(), inS.size() * 4); boS.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boP.map<float*>(); memcpy(p, inP.data(), inP.size() * 4); boP.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p = boO.map<float*>(); memset(p, 0, (size_t)NOUT * OUTPKT * 4); boO.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    fprintf(stderr, "[probe] dispatch one-xclbin 3-pass GDN (48 v-heads, 1 xrt::run) x3...\n");
    double ms = 0.0;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        auto run = kX.krnl(3, 0, 0, static_cast<xrt::bo&>(boS),
                                    static_cast<xrt::bo&>(boP),
                                    static_cast<xrt::bo&>(boO));
        if (run.wait(60000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] DISPATCH FAILED/TIMEOUT\n"); return 3; }
        double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[probe] iter %d latency = %.2f ms\n", it, d);
        if (it == 2) ms = d;
    }
    boO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float* out = boO.map<float*>();   // [vh*17*1024 + pkt*1024 + ...]

    // ---- per-v-head fp32 reference + compare (same math as gdn_probe) ----
    double maxdy = 0, maxdS = 0; int wy = -1, wS = -1;
    for (int vh = 0; vh < NV; vh++) {
        const float* Sv  = S0.data() + (size_t)vh*HV*HV;
        const float* knv = kn.data() + vh*HV;
        const float* qnv = qn.data() + vh*HV;
        const float* vvr = vv.data() + vh*HV;
        float g = gdec[vh], bt = beta[vh];
        float a[128], b[128];
        for (int j = 0; j < HV; j++) { a[j] = 0; b[j] = 0;
            for (int i = 0; i < HV; i++) { a[j] += Sv[i*HV+j]*knv[i]; b[j] += Sv[i*HV+j]*qnv[i]; } }
        float c = 0; for (int i = 0; i < HV; i++) c += knv[i]*qnv[i];
        float delta_ref[128], y_ref[128];
        for (int j = 0; j < HV; j++) {
            float kvm = g * a[j];
            delta_ref[j] = (vvr[j] - kvm) * bt;
            y_ref[j] = g * b[j] + delta_ref[j] * c;
        }
        const float* y_npu = out + (size_t)vh * (1 + NBLK) * OUTPKT;   // pkt 0, y @ [0:128]
        for (int j = 0; j < HV; j++) {
            double d = std::fabs((double)y_npu[j] - y_ref[j]) / std::max(std::fabs((double)y_ref[j]), 1.0);
            if (d > maxdy) { maxdy = d; wy = vh; }
        }
        for (int blk = 0; blk < NBLK; blk++)
            for (int r = 0; r < RPB; r++) {
                int i = blk * RPB + r;
                const float* s2row = out + (size_t)vh * (1 + NBLK) * OUTPKT + (1 + blk) * OUTPKT + r * HV;
                for (int j = 0; j < HV; j++) {
                    float ref = g * Sv[i*HV+j] + knv[i] * delta_ref[j];
                    double d = std::fabs((double)s2row[j] - ref) / std::max(std::fabs((double)ref), 1.0);
                    if (d > maxdS) { maxdS = d; wS = vh; }
                }
            }
    }
    fprintf(stderr, "[probe] max|Δy|=%.3e (vh %d)  max|ΔS|=%.3e (vh %d)\n", maxdy, wy, maxdS, wS);
    bool ok = (maxdy < 1e-4) && (maxdS < 1e-4);
    printf("maxdy=%.3e maxdS=%.3e steady=%.2fms %s\n", maxdy, maxdS, ms, ok ? "PASS" : "FAIL");
    if (ok) { fprintf(stderr, "[probe] PASS bit-correct (rel <1e-4), steady latency %.2f ms (vs 3-pass ~90ms)\n", ms); return 0; }
    fprintf(stderr, "[probe] FAIL\n"); return 4;
}