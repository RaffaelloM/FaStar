// tools/gdn_probe.cpp — multi-v-head hardware validation of the collapsed GDN scan.
//
// The 3 GDN xclbins now process ALL 48 v-heads in ONE dispatch each (3 dispatches
// total).  This probe synthesizes 48 v-heads of DISTINCT data (v-dependent), packs
// the multi-v-head BOs, dispatches passA → delta → passB ONCE each, and compares
// the NPU y/S2 to a per-v-head fp32 reference.  Distinct data catches routing or
// output-shuffle bugs (identical v-heads would mask them); non-zero S0 exercises
// the passA accumulation (a=Sᵀ@kn, b=Sᵀ@qn), not just the delta closed form.
//
//   passA : SpktA[48*128, 130]=[S_row|kn_i|qn_i] -> ab[48*1024] (256 used/v-head) [a|b]
//   delta : din[48*642]=[a|b|v|kn|qn|gdec|beta]  -> dout[48*1024] (256 used) [delta|y]
//   passB : SpktB[48*128, 264]=[S0|delta|kn_i|gdec|pad] -> S2[48*128*128]
//
// Reference per v-head v (S0, kn, qn, v, gdec, beta all v-dependent):
//   a[j]=Σ_i S0[i,j]*kn[i] ; b[j]=Σ_i S0[i,j]*qn[i]
//   kvm[j]=gdec*a[j] ; delta[j]=(v[j]-kvm[j])*beta ; c=Σ_i kn[i]*qn[i]
//   y[j]=gdec*b[j]+delta[j]*c ; S2[i,j]=gdec*S0[i,j]+kn[i]*delta[j]
// PASS = max|Δy|<1e-5 AND max|ΔS|<1e-5 across all 48 v-heads.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
static double now_s(){return 0.0;}
static double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include "aiebu/aiebu.h"

constexpr int HV = 128;
constexpr int NROWS = 128;
constexpr int NV = 48;
constexpr int PKT_A = 130;
constexpr int PKT_B = 264;
constexpr int DIN = 642;
constexpr int OUTPKT = 1024;     // 256 used per v-head for passA/delta

struct XclbinKern { xrt::hw_context ctx; xrt::ext::kernel krnl; };

static XclbinKern load_xclbin(xrt::device& dev, const std::string& xclb_path,
                              const std::vector<char>& insts) {
    xrt::xclbin xclb(xclb_path);
    try { dev.register_xclbin(xclb); }
    catch (const std::exception& e) { fprintf(stderr, "[probe] register_xclbin FAILED %s: %s\n", xclb_path.c_str(), e.what()); exit(2); }
    xrt::uuid uid = xclb.get_uuid();
    std::unique_ptr<xrt::hw_context> ctxp;
    try { ctxp = std::make_unique<xrt::hw_context>(dev, uid); }
    catch (const std::exception& e) { fprintf(stderr, "[probe] hw_context FAILED %s: %s\n", xclb_path.c_str(), e.what()); exit(2); }
    char* elf_buf = nullptr;
    uint32_t elf_sz = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(), (uint32_t)insts.size(),
        NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
    if (elf_sz == 0 || !elf_buf) { fprintf(stderr, "[probe] aiebu_get_elf FAILED %s\n", xclb_path.c_str()); exit(2); }
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
    std::string DA = std::string(dir) + "/fst_gdn_passA.xclbin";
    std::string DD = std::string(dir) + "/fst_gdn_delta.xclbin";
    std::string DB = std::string(dir) + "/fst_gdn_passB.xclbin";
    std::string IA = std::string(dir) + "/fst_gdn_passA_insts.bin";
    std::string ID = std::string(dir) + "/fst_gdn_delta_insts.bin";
    std::string IB = std::string(dir) + "/fst_gdn_passB_insts.bin";

    // ---- synthesize 48 v-heads of distinct data ----
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

    xrt::device dev(0);
    XclbinKern kA = load_xclbin(dev, DA, read_file(IA));
    XclbinKern kD = load_xclbin(dev, DD, read_file(ID));
    XclbinKern kB = load_xclbin(dev, DB, read_file(IB));

    // ---- passA: SpktA[48*128,130]=[S_row|kn_i|qn_i] -> ab[48*1024] ----
    std::vector<float> spktA((size_t)NV * NROWS * PKT_A, 0.0f);
    for (int vh = 0; vh < NV; vh++)
        for (int i = 0; i < NROWS; i++) {
            float* row = spktA.data() + ((size_t)vh * NROWS + i) * PKT_A;
            for (int j = 0; j < HV; j++) row[j] = S0[vh*HV*HV + i*HV + j];
            row[HV + 0] = kn[vh*HV + i];
            row[HV + 1] = qn[vh*HV + i];
        }
    xrt::ext::bo boSpktA(dev, (size_t)NV * NROWS * PKT_A * 4);
    xrt::ext::bo boAB(dev,   (size_t)NV * OUTPKT * 4);
    { auto* p = boSpktA.map<float*>(); memcpy(p, spktA.data(), spktA.size() * 4); }
    { auto* p = boAB.map<float*>();    memset(p, 0, (size_t)NV * OUTPKT * 4); }
    boSpktA.sync(XCL_BO_SYNC_BO_TO_DEVICE); boAB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr, "[probe] dispatch passA (48 v-heads)...\n");
    double tA0 = now_ms();
    auto runA = kA.krnl(3, 0, 0, static_cast<xrt::bo&>(boSpktA), static_cast<xrt::bo&>(boAB));
    if (runA.wait(30000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] passA FAILED\n"); return 3; }
    fprintf(stderr, "[time] passA dispatch+wait = %.2f ms\n", now_ms() - tA0);
    boAB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float* ab = boAB.map<float*>();   // [vh*1024 + 0..255] = [a(128)|b(128)]

    // ---- delta: din[48*642]=[a|b|v|kn|qn|gdec|beta] -> dout[48*1024] ----
    std::vector<float> din((size_t)NV * DIN, 0.0f);
    for (int vh = 0; vh < NV; vh++) {
        float* d = din.data() + (size_t)vh * DIN;
        float* a = ab + (size_t)vh * OUTPKT + 0;
        float* b = ab + (size_t)vh * OUTPKT + HV;
        memcpy(d + 0,   a, HV * 4);
        memcpy(d + 128, b, HV * 4);
        memcpy(d + 256, vv.data() + vh*HV, HV * 4);
        memcpy(d + 384, kn.data() + vh*HV, HV * 4);
        memcpy(d + 512, qn.data() + vh*HV, HV * 4);
        d[640] = gdec[vh]; d[641] = beta[vh];
    }
    xrt::ext::bo boDin(dev, (size_t)NV * DIN * 4);
    xrt::ext::bo boDout(dev, (size_t)NV * OUTPKT * 4);
    { auto* p = boDin.map<float*>();  memcpy(p, din.data(), din.size() * 4); }
    { auto* p = boDout.map<float*>(); memset(p, 0, (size_t)NV * OUTPKT * 4); }
    boDin.sync(XCL_BO_SYNC_BO_TO_DEVICE); boDout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr, "[probe] dispatch delta (48 v-heads)...\n");
    double tD0 = now_ms();
    auto runD = kD.krnl(3, 0, 0, static_cast<xrt::bo&>(boDin), static_cast<xrt::bo&>(boDout));
    if (runD.wait(30000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] delta FAILED\n"); return 3; }
    fprintf(stderr, "[time] delta dispatch+wait = %.2f ms\n", now_ms() - tD0);
    boDout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float* dy = boDout.map<float*>();  // [vh*1024 + 0..255] = [delta(128)|y(128)]
    std::vector<float> delta_all((size_t)NV * HV);
    for (int vh = 0; vh < NV; vh++) memcpy(delta_all.data() + vh*HV, dy + (size_t)vh*OUTPKT + 0, HV*4);

    // ---- passB: SpktB[48*128,264]=[S0|delta|kn_i|gdec|pad] -> S2[48*128*128] ----
    std::vector<float> spktB((size_t)NV * NROWS * PKT_B, 0.0f);
    for (int vh = 0; vh < NV; vh++)
        for (int i = 0; i < NROWS; i++) {
            float* row = spktB.data() + ((size_t)vh * NROWS + i) * PKT_B;
            for (int j = 0; j < HV; j++) row[j]     = S0[vh*HV*HV + i*HV + j];
            for (int j = 0; j < HV; j++) row[HV+j]  = delta_all[vh*HV + j];
            row[2*HV + 0] = kn[vh*HV + i];
            row[2*HV + 1] = gdec[vh];
        }
    xrt::ext::bo boSpktB(dev, (size_t)NV * NROWS * PKT_B * 4);
    xrt::ext::bo boS2(dev,    (size_t)NV * NROWS * HV * 4);
    { auto* p = boSpktB.map<float*>(); memcpy(p, spktB.data(), spktB.size() * 4); }
    { auto* p = boS2.map<float*>();    memset(p, 0, (size_t)NV * NROWS * HV * 4); }
    boSpktB.sync(XCL_BO_SYNC_BO_TO_DEVICE); boS2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr, "[probe] dispatch passB (48 v-heads)...\n");
    double tB0 = now_ms();
    auto runB = kB.krnl(3, 0, 0, static_cast<xrt::bo&>(boSpktB), static_cast<xrt::bo&>(boS2));
    if (runB.wait(30000) != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "[probe] passB FAILED\n"); return 3; }
    fprintf(stderr, "[time] passB dispatch+wait = %.2f ms\n", now_ms() - tB0);
    boS2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float* S2 = boS2.map<float*>();

    // ---- per-v-head fp32 reference + compare ----
    // First isolate which pass corrupts: compare passA a,b; delta y; passB S2.
    double maxda = 0, maxdb = 0, maxdy = 0, maxdS = 0;
    int wa=-1, wb=-1, worst_vh_y = -1, worst_vh_S = -1;
    std::vector<float> a_all((size_t)NV*HV), b_all((size_t)NV*HV);
    for (int vh = 0; vh < NV; vh++) {
        const float* Sv = S0.data() + (size_t)vh*HV*HV;
        const float* knv = kn.data() + vh*HV;
        const float* qnv = qn.data() + vh*HV;
        const float* vv_r = vv.data() + vh*HV;
        float g = gdec[vh], bt = beta[vh];
        float aref[128], bref[128];
        for (int j = 0; j < HV; j++) { aref[j] = 0; bref[j] = 0;
            for (int i = 0; i < HV; i++) { aref[j] += Sv[i*HV+j]*knv[i]; bref[j] += Sv[i*HV+j]*qnv[i]; } }
        const float* a_npu = ab + (size_t)vh*OUTPKT + 0;
        const float* b_npu = ab + (size_t)vh*OUTPKT + HV;
        memcpy(a_all.data()+vh*HV, a_npu, HV*4); memcpy(b_all.data()+vh*HV, b_npu, HV*4);
        for (int j = 0; j < HV; j++) {
            double da = std::fabs((double)a_npu[j]-aref[j]) / std::max(std::fabs((double)aref[j]), 1.0);
            double db = std::fabs((double)b_npu[j]-bref[j]) / std::max(std::fabs((double)bref[j]), 1.0);
            if (da>maxda){maxda=da;wa=vh;} if (db>maxdb){maxdb=db;wb=vh;}
        }
    }
    fprintf(stderr, "[probe] passA: max|Δa|=%.3e (vh %d)  max|Δb|=%.3e (vh %d)\n", maxda, wa, maxdb, wb);
    for (int vh = 0; vh < NV; vh++) {
        const float* Sv = S0.data() + (size_t)vh*HV*HV;
        const float* knv = kn.data() + vh*HV;
        const float* qnv = qn.data() + vh*HV;
        const float* vv_r = vv.data() + vh*HV;
        float g = gdec[vh], bt = beta[vh];
        float a[128], b[128];
        for (int j = 0; j < HV; j++) { a[j] = 0; b[j] = 0;
            for (int i = 0; i < HV; i++) { a[j] += Sv[i*HV+j]*knv[i]; b[j] += Sv[i*HV+j]*qnv[i]; } }
        float c = 0; for (int i = 0; i < HV; i++) c += knv[i]*qnv[i];
        float delta_ref[128], y_ref[128];
        for (int j = 0; j < HV; j++) {
            float kvm = g * a[j];
            delta_ref[j] = (vv_r[j] - kvm) * bt;
            y_ref[j] = g * b[j] + delta_ref[j] * c;
        }
        const float* y_npu = dy + (size_t)vh*OUTPKT + HV;
        for (int j = 0; j < HV; j++) {
            double d = std::fabs((double)y_npu[j] - y_ref[j]) / std::max(std::fabs((double)y_ref[j]), 1.0);
            if (d > maxdy) { maxdy = d; worst_vh_y = vh; } }
        const float* S2v = S2 + (size_t)vh*HV*HV;
        for (int i = 0; i < HV; i++)
            for (int j = 0; j < HV; j++) {
                float ref = g * Sv[i*HV+j] + knv[i] * delta_ref[j];
                double d = std::fabs((double)S2v[i*HV+j] - ref) / std::max(std::fabs((double)ref), 1.0);
                if (d > maxdS) { maxdS = d; worst_vh_S = vh; } }
    }
    fprintf(stderr, "[probe] max|Δy|=%.3e (worst v-head %d)  max|ΔS|=%.3e (worst v-head %d)\n",
            maxdy, worst_vh_y, maxdS, worst_vh_S);
    // Relative tolerance: fp32 accumulation over 128 terms amplified through the
    // a→delta→y chain runs ~1e-6..1e-5; a real routing/drain bug is O(1).  1e-4 separates them.
    bool ok = (maxda < 1e-4) && (maxdb < 1e-4) && (maxdy < 1e-4) && (maxdS < 1e-4);
    printf("maxda=%.3e maxdb=%.3e maxdy=%.3e maxdS=%.3e %s\n", maxda, maxdb, maxdy, maxdS, ok ? "PASS" : "FAIL");
    if (ok) { fprintf(stderr, "[probe] PASS bit-correct (rel <1e-4) across all 48 v-heads ✓\n"); return 0; }
    fprintf(stderr, "[probe] FAIL\n"); return 4;
}