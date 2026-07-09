// tools/mc_gemm_probe.cpp — verify the 3x-stride batched multi-core expert GEMM
// (fst_expert_gemm_vec_mc gate/up + fst_expert_gemm_down_mc down) on hardware.
//
// Builds a random bf16 batched-B BO in the gate|up|down 3x-stride layout (expert i
// gate at offset i*3*N*K, up at +N*K, down at +2*N*K), a shared or per-worker A,
// dispatches the IRON multi-core kernel, reads the batched C, and compares to a
// float32 CPU reference (A @ B^T, b_col_maj).  Pass = cos>0.999 for every op.
//
// Modes:  "gate"  -> vec_mc,  A=[M,K] shared (replicated), B gate slice @0
//          "up"   -> vec_mc,  A=[M,K] shared, B sub-buffer @N*K (up slice)
//          "down" -> down_mc, A=[E*M,K] per-worker, B sub-buffer @2*N*K (down slice)
// NO mockups: real xclbin, real IRON insts, real hw run, real cos check.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include "aiebu/aiebu.h"

static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); uint32_t r=(u+0x8000)>>16; return (uint16_t)r; }
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

// deterministic pseudo-random bf16 in [-1,1]
static uint16_t rnd(int i){ uint32_t s=(uint32_t)(i*2654435761u + 12345u); float f=((s>>16)/65535.0f)*2.0f-1.0f; return f2b(f); }

struct KCtx {
    xrt::hw_context ctx;
    xrt::ext::kernel krnl;
};

static KCtx load_kernel(xrt::device& dev, const char* xclb_path, const char* insts_path) {
    std::ifstream f(insts_path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr,"[probe] MISSING %s\n", insts_path); std::exit(5); }
    std::vector<char> inst((size_t)f.tellg()); f.seekg(0); f.read(inst.data(), inst.size());
    char* elf_buf=nullptr;
    uint32_t elf_sz = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        inst.data(), (uint32_t)inst.size(),
        NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
    if (elf_sz==0 || !elf_buf){ fprintf(stderr,"aiebu_get_elf FAILED for %s\n", insts_path); std::exit(2); }
    std::string xp(xclb_path);
    xrt::xclbin xclb(xp); dev.register_xclbin(xclb);
    xrt::uuid uid = xclb.get_uuid();
    xrt::hw_context ctx(dev, uid);
    xrt::elf elf(elf_buf, elf_sz);
    xrt::module mod(elf); free(elf_buf);
    std::string kname; for (auto& k : xclb.get_kernels()) kname = k.get_name();
    xrt::ext::kernel krnl(ctx, mod, kname);
    fprintf(stderr,"[probe] loaded %s  kernel=%s  insts=%zuB elf=%uB\n", xclb_path, kname.c_str(), inst.size(), elf_sz);
    return {std::move(ctx), std::move(krnl)};
}

// CPU ref: C[i*M+m, n] = sum_k A[i*M+m,k] * B_i[n,k]   (b_col_maj, A@B^T)
// a_rows: A row count per expert (M for shared-replicated read at row m, but for
// per-worker A each expert has its own M rows).  We pass A as [E*M, K] always;
// for gate/up all experts read the SAME first M rows (replicated).
static void cpu_ref(float* C, const std::vector<uint16_t>& A, const std::vector<uint16_t>& B,
                    int E, int M, int K, int N, size_t b_off_elem, bool a_per_worker) {
    for (int i=0;i<E;i++) for (int m=0;m<M;m++) for (int n=0;n<N;n++){
        double acc=0;
        int arow = a_per_worker ? (i*M+m) : m;   // shared: every expert reads row m
        for (int k=0;k<K;k++)
            acc += (double)b2f(A[(size_t)arow*K + k]) * (double)b2f(B[b_off_elem + (size_t)i*3*N*K + (size_t)n*K + k]);
        C[(size_t)(i*M+m)*N + n] = (float)acc;
    }
}

int main(int argc, char** argv){
    std::string mode = (argc>1) ? argv[1] : "gate";
    constexpr int M=16;
    int E, K, N; bool a_per_worker; const char* xclb, *insts; bool single=false;
    if (mode=="down"){ E=6; K=2048; N=4096; a_per_worker=true;  xclb="fst_expert_gemm_down_mc.xclbin"; insts="fst_expert_gemm_down_mc_insts.bin"; }
    else if (mode=="up"){ E=6; K=4096; N=2048; a_per_worker=false; xclb="fst_expert_gemm_vec_mc.xclbin"; insts="fst_expert_gemm_vec_mc_insts.bin"; }
    else if (mode=="gate1"){ E=1; K=4096; N=2048; a_per_worker=false; single=true; xclb="fst_expert_gemm_vec.xclbin"; insts="fst_expert_gemm_vec_insts.bin"; }
    else if (mode=="down1"){ E=1; K=2048; N=4096; a_per_worker=false; single=true; xclb="fst_expert_gemm_down.xclbin"; insts="fst_expert_gemm_down_insts.bin"; }
    else              { E=6; K=4096; N=2048; a_per_worker=false; xclb="fst_expert_gemm_vec_mc.xclbin"; insts="fst_expert_gemm_vec_mc_insts.bin"; }

    // A: [E*M, K] (per-worker) or [M, K] (shared).  BO always E*M*K (shared: first M rows valid, rest 0).
    const size_t A_ELEM = (size_t)E*M*K, A_BYTES = A_ELEM*2;
    // single-core: B is [N,K] flat (no 3x stride).  multicore: E*3*N*K (3x stride).
    const size_t B_ELEM = single ? (size_t)K*N : (size_t)E*3*K*N;
    const size_t B_BYTES = B_ELEM*2;
    const size_t C_ELEM = (size_t)E*M*N, C_BYTES = C_ELEM*2;

    std::vector<uint16_t> A(A_ELEM, 0);
    for (size_t i=0;i<A_ELEM;i++) A[i] = a_per_worker ? rnd((int)i) : (i < (size_t)M*K ? rnd((int)i) : 0);

    // B: 3x-stride (multicore) or flat [N,K] (single).  Fill the slice this mode
    // reads: gate@0, up@N*K, down@2*N*K.  single: flat at 0 (no stride/slice).
    size_t b_off_elem = single ? 0 : ((mode=="down") ? (size_t)2*N*K : (mode=="up" ? (size_t)N*K : 0));
    std::vector<uint16_t> B(B_ELEM, 0);
    for (int i=0;i<E;i++) for (int n=0;n<N;n++) for (int k=0;k<K;k++)
        B[b_off_elem + (size_t)i*3*N*K + (size_t)n*K + k] = rnd((int)((i*1000+n)*K+k+7));

    xrt::device dev(0);
    xrt::ext::bo boD1(dev, 1<<20), boD2(dev, 1<<20);
    auto kc = load_kernel(dev, xclb, insts);

    // For up/down the kernel reads relative to the sub-buffer base, which we shift
    // so worker i lands on the right slice.  gate: base=0.  up: base=N*K.  down: base=2*N*K.
    // single: no shift (whole B at 0).
    size_t b_sub_off_elem = b_off_elem;
    size_t b_sub_elems    = B_ELEM - b_sub_off_elem;
    xrt::ext::bo boA(dev, A_BYTES);
    xrt::ext::bo boB(dev, B_BYTES);
    xrt::ext::bo boC(dev, C_BYTES);
    { auto* p=boA.map<uint16_t*>(); memcpy(p, A.data(), A_BYTES); boA.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p=boB.map<uint16_t*>(); memcpy(p, B.data(), B_BYTES); boB.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    { auto* p=boC.map<uint16_t*>(); memset(p,0,C_BYTES); boC.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    xrt::bo boBsub(boB, b_sub_elems*2, b_sub_off_elem*2);
    fprintf(stderr,"[probe] dispatch %s  A=%zuB B_sub@%zu(%zuB) C=%zuB\n",
            mode.c_str(), A_BYTES, b_sub_off_elem*2, b_sub_elems*2, C_BYTES);
    double t0=now();
    auto run = kc.krnl(3,0,0,
        static_cast<xrt::bo&>(boA), static_cast<xrt::bo&>(boBsub),
        static_cast<xrt::bo&>(boC), static_cast<xrt::bo&>(boD1), static_cast<xrt::bo&>(boD2));
    ert_cmd_state st = ERT_CMD_STATE_COMPLETED;
    try { st = run.wait(20000); } catch (const std::exception& e){ fprintf(stderr,"EXC: %s\n",e.what()); return 3; }
    fprintf(stderr,"[probe] wait state=%d after %.3fs\n",(int)st, now()-t0);
    if (st != ERT_CMD_STATE_COMPLETED){ fprintf(stderr,"[probe] FAILED state=%d\n",(int)st); return 3; }
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<uint16_t> C(C_ELEM); memcpy(C.data(), boC.map<char*>(), C_BYTES);

    std::vector<float> Cref(C_ELEM, 0.0f);
    cpu_ref(Cref.data(), A, B, E, M, K, N, b_off_elem, a_per_worker);

    double dot=0,na=0,nb=0; float maxdiff=0; int wi=0; double rmax=0;
    for (size_t i=0;i<C_ELEM;i++){ float a=b2f(C[i]),b=Cref[i]; double d=std::fabs(a-b); if(d>maxdiff){maxdiff=d;wi=(int)i;} dot+=(double)a*b; na+=(double)a*a; nb+=(double)b*b; }
    for (size_t i=0;i<C_ELEM;i++){ double r=std::fabs(b2f(C[i]))/(std::fabs(Cref[i])+1e-12); if(r>rmax)rmax=r; }
    double cos = (na<1e-30||nb<1e-30)?0.0 : dot/(std::sqrt(na)*std::sqrt(nb));
    fprintf(stderr,"[probe] %s  cos=%.6f  max|diff|=%.4g @[%d]  |np|/|ref|=%.4f\n",
            mode.c_str(), cos, maxdiff, wi, rmax);
    printf("cos=%.6f maxdiff=%.4g\n", cos, maxdiff);
    if (cos > 0.999){ fprintf(stderr,"[probe] PASS cos>0.999 ✓\n"); return 0; }
    fprintf(stderr,"[probe] FAIL cos<=0.999\n");
    for (int t=0;t<3;t++) fprintf(stderr,"  C[%d] npu=%.4f ref=%.4f\n", t, b2f(C[t]), Cref[t]);
    return 4;
}