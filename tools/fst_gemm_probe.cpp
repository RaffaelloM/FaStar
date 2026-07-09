// fst_gemm_probe.cpp — Standalone NPU GEMM correctness probe.
//
// Runs ONE MLA kvc GEMM (M=8, K=4096, N=512) on the NPU with known inputs:
//   A = all ones   [8, 4096]
//   B = identity   [4096, 512]  (B[k,n] = 1 if k==n<512 else 0)
//   C = A @ B      [8, 512]     => every C[m,n] must be 1.0 for n<512
//
// (sum_k A[m,k]*B[k,n] = A[m,n]*1 = 1, since only k==n contributes.)
//
// If C is all 1s, the kvc kernel + DMA fill TAPs are correct for this shape
// and the engine's ~3.7x/33x error is elsewhere.  If C is NOT all 1s, the
// NPU GEMM path itself is broken — isolating the bug from the engine.
//
// Build:
//   g++ -std=c++17 -O2 -o fst_gemm_probe fst_gemm_probe.cpp \
//       -I. -I/usr/include/xrt -lxrt_coreutil -lxrt_core -luuid -laiebu -lpthread
// Run:
//   XILINX_XRT=/usr LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./fst_gemm_probe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>
#include "fst_aiebu_cache.hpp"

using bf16_t = uint16_t;
static inline float bf16f(bf16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static inline bf16_t f2bf(float f){uint32_t b;memcpy(&b,&f,4);return(bf16_t)(b>>16);}

int main(int argc, char** argv) {
    const char* kname = argc > 1 ? argv[1] : "kvc";        // kvc | kvc64 | kvcrep | ...
    const int M = std::getenv("PROBE_M") ? atoi(std::getenv("PROBE_M")) : 8;
    const int K = 4096, N = argc > 2 ? atoi(argv[2]) : 512;
    const char* xclbin = argc > 3 ? argv[3] : "fst_mla_unified.xclbin";
    // FFN expert GEMM uses B stored [N,K] (transposed) and computes C=A@Bᵀ.
    // PROBE_FFN: lay out B as [N,K] identity (B[n,k]=1 if n==k<min(N,K)).
    // PROBE_K overrides the reduction dim (FFN down uses K=2048).
    const bool bcol = std::getenv("PROBE_FFN") != nullptr;
    const int Keff = std::getenv("PROBE_K") ? atoi(std::getenv("PROBE_K")) : K;
    char insts[256];
    if (const char* ei = std::getenv("PROBE_INSTS")) snprintf(insts, sizeof(insts), "%s", ei);
    else if (bcol) snprintf(insts, sizeof(insts), "fst_expert_gemm_%s_insts.bin",
                       strcmp(kname,"down")==0 ? "down" : "vec");
    else snprintf(insts, sizeof(insts), "fst_mla_%s_insts.bin", kname);

    xrt::device dev(0);
    AiebuKernelCache cache(dev, xclbin);
    cache.register_kernel(kname, insts, xclbin);
    int grp = cache.data_group_id();

    auto& krnl = cache.get(kname);
    const size_t MB = 1024*1024;
    xrt::ext::bo bo_d1(dev, 1*MB), bo_d2(dev, 1*MB);
    const int Kr = Keff;   // reduction dim (PROBE_K overrides; defaults to K)
    const size_t a_sz = (size_t)M*Kr*sizeof(bf16_t);
    const size_t b_sz = bcol ? (size_t)N*Kr*sizeof(bf16_t)   // B [N,Kr]
                             : (size_t)Kr*N*sizeof(bf16_t);  // B [Kr,N]
    const size_t c_sz = (size_t)M*N*sizeof(bf16_t);
    const int TILE_N = 64;
    const int N_div_n = (N + TILE_N - 1) / TILE_N;
    const bool replicate = std::getenv("PROBE_REPLICATE") != nullptr;
    const size_t a_bo_elems = replicate ? (size_t)N_div_n * M * Kr : (size_t)M * Kr;
    xrt::ext::bo bo_A(dev, a_bo_elems * sizeof(bf16_t));
    xrt::bo      bo_B(dev, b_sz, xrt::bo::flags::host_only, grp);
    xrt::ext::bo bo_C(dev, c_sz);

    // A = ones (replicated N_div_n times if PROBE_REPLICATE is set)
    {auto* a = bo_A.map<bf16_t*>();
     bf16_t one = f2bf(1.0f);
     for (size_t i=0;i<a_bo_elems;i++) a[i]=one;}
    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const bool b_ones = std::getenv("PROBE_B_ONES") != nullptr;
    {auto* b = bo_B.map<bf16_t*>();
     if (b_ones) { bf16_t one = f2bf(1.0f); for (size_t i=0;i<(size_t)N*Kr;i++) b[i]=one; }
     else if (bcol) { // B [N,Kr] identity: B[n,k]=1 if n==k<min(N,Kr)
        std::memset(b, 0, b_sz); bf16_t one = f2bf(1.0f);
        for (int n=0;n<N && n<Kr;n++) b[(size_t)n*Kr + n] = one;
     } else { // B [Kr,N] identity: B[k,n]=1 if k==n<N
        std::memset(b, 0, b_sz); bf16_t one = f2bf(1.0f);
        for (int n=0;n<N && n<Kr;n++) b[(size_t)n*N + n] = one;
     }}
    bo_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // C = 0
    {auto* c = bo_C.map<bf16_t*>(); std::memset(c,0,c_sz);}
    bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    fprintf(stderr, "[probe] running %s: A[%d,%d]=ones, B=%s identity, C=A@B -> expect all 1.0\n",
            kname, M, Kr, bcol ? "[N,Kr]T" : "[Kr,N]");
    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(bo_A), static_cast<xrt::bo&>(bo_B),
        static_cast<xrt::bo&>(bo_C),
        static_cast<xrt::bo&>(bo_d1), static_cast<xrt::bo&>(bo_d2));
    ert_cmd_state s = run.wait(30000);
    if (s != ERT_CMD_STATE_COMPLETED) {
        fprintf(stderr, "[probe] FAIL: run.wait -> %d (timeout/error)\n", (int)s);
        return 2;
    }
    bo_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* c = bo_C.map<bf16_t*>();
    const float expect = b_ones ? (float)Kr : 1.0f;
    int n_bad = 0, n_first_bad = -1;
    float max_abs = 0, min_abs = 1e9;
    for (int m=0;m<M;m++) for (int n=0;n<N;n++) {
        float v = bf16f(c[(size_t)m*N+n]);
        if (std::fabs(v - expect) > 0.05f * std::max(1.0f, expect)) { if (n_first_bad<0) n_first_bad=m*N+n; n_bad++; }
        if (std::fabs(v) > max_abs) max_abs = std::fabs(v);
        if (std::fabs(v) < min_abs) min_abs = std::fabs(v);
    }
    fprintf(stderr, "[probe] C stats (expect %.1f): max|C|=%.4f min|C|=%.4f  bad=%d/%d  first_bad_idx=%d\n",
            expect, max_abs, min_abs, n_bad, M*N, n_first_bad);
    // Dump row 0 in 64-col chunks (one chunk per N-tile) to see the per-tile pattern
    fprintf(stderr, "[probe] C[0,*] by N-tile (64 cols each):\n");
    for (int t=0;t<(N+63)/64;t++) {
        fprintf(stderr, "  tile %d (cols %d..%d):", t, t*64, std::min(t*64+63,N-1));
        for (int n=0;n<std::min(64,N-t*64);n++) {
            float v = bf16f(c[(size_t)0*N + t*64 + n]);
            if (((n)%8)==0) fprintf(stderr, " |");
            fprintf(stderr, " %.2f", v);
        }
        fprintf(stderr, "\n");
    }
    if (n_bad == 0) { fprintf(stderr, "[probe] PASS — C is all 1.0; NPU GEMM correct for %s.\n", kname); return 0; }
    fprintf(stderr, "[probe] FAIL — C is NOT all 1.0; NPU GEMM broken for %s.\n", kname);
    return 1;
}