// tools/gdn_8kslab_canary_probe.cpp — stack-S persistence diagnostic.
// Packs S0 = 0.5 (bf16) for 48 v-heads, runs the canary kernel (S += 1.0 × K=8
// steps), reads back snew, and histograms the value:
//   snew ≈ 8.5  ⇒ stack S PERSISTS across the K-loop (S0 0.5 + 8 increments)
//   snew ≈ 1.5  ⇒ only the last increment landed (S does NOT persist between steps)
//   snew ≈ 0.5  ⇒ passB never writes the stack S
//   snew = garbage ⇒ stack S is being clobbered by the DMA/ObjectFifo path
//   XILINX_XRT=/usr ./tools/gdn_8kslab_canary_probe [Nruns=5]
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

constexpr int HV=128, COLS=32, K=8, NVH=48, V=8;
constexpr int PAR_SZ = 2*HV+COLS+3, YD_SZ = 2*COLS;
constexpr int S0_ELEM = HV*COLS, PAR_ELEM = K*PAR_SZ, YD_ELEM = K*YD_SZ;

static inline uint16_t f2bf(float f){ uint32_t u; std::memcpy(&u,&f,4); uint32_t lsb=(u>>16)&1; uint32_t bias=0x7FFFu+lsb; return (uint16_t)((u+bias)>>16); }
static inline float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; std::memcpy(&f,&u,4); return f; }

static std::vector<char> read_file(const std::string& p){
    std::ifstream f(p, std::ios::binary|std::ios::ate); std::vector<char> d((size_t)f.tellg());
    f.seekg(0); f.read(d.data(), d.size()); return d;
}

int main(int argc, char** argv){
    int NRUNS = (argc>1)?std::atoi(argv[1]):5;
    const float S0VAL = 0.5f;
    const size_t S0_BO  = (size_t)NVH*S0_ELEM;
    const size_t PAR_BO = (size_t)NVH*PAR_ELEM;
    const size_t YD_BO  = (size_t)NVH*YD_ELEM;
    std::vector<uint16_t> s0bo(S0_BO, f2bf(S0VAL)), parbo(PAR_BO, f2bf(0.1f));
    std::vector<uint16_t> snewbo(S0_BO,0), ydbo(YD_BO,0);

    xrt::device dev(0);
    xrt::xclbin xclb(std::string("kernels/fst_gdn_8kslab_canary.xclbin")); dev.register_xclbin(xclb);
    xrt::uuid uid=xclb.get_uuid(); xrt::hw_context ctx(dev,uid);
    auto insts=read_file("kernels/fst_gdn_8kslab_canary_insts.bin"); char* eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kname; for(auto&k:xclb.get_kernels()) kname=k.get_name();
    xrt::ext::kernel krnl(ctx,mod,kname);
    xrt::ext::bo boS0(dev,S0_BO*2), boPar(dev,PAR_BO*2), boSn(dev,S0_BO*2), boYd(dev,YD_BO*2);
    auto* pS0=boS0.map<uint16_t*>(); auto* pPar=boPar.map<uint16_t*>();
    std::memcpy(pS0, s0bo.data(), S0_BO*2); std::memcpy(pPar, parbo.data(), PAR_BO*2);
    boS0.sync(XCL_BO_SYNC_BO_TO_DEVICE); boPar.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    double best=1e9;
    for(int r=0;r<NRUNS+1;r++){
        auto t0=std::chrono::steady_clock::now();
        auto run=krnl(3,0,0, static_cast<xrt::bo&>(boS0), static_cast<xrt::bo&>(boPar),
                           static_cast<xrt::bo&>(boSn), static_cast<xrt::bo&>(boYd));
        if(run.wait(60000)!=ERT_CMD_STATE_COMPLETED){ fprintf(stderr,"TIMEOUT\n"); return 3; }
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        if(r>0 && ms<best) best=ms;
    }
    boSn.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* pSn=boSn.map<uint16_t*>();
    std::memcpy(snewbo.data(), pSn, S0_BO*2);

    // histogram snew values (rounded to nearest 0.25 bucket)
    fprintf(stderr,"=== gdn_8kslab CANARY (S0=%.2f, S+=1 ×K=%d steps, expect %.2f if persists) ===\n",S0VAL,K,S0VAL+K);
    fprintf(stderr,"canary latency = %.2f ms (best of %d)\n", best, NRUNS);
    std::vector<float> vals; vals.reserve(S0_BO);
    for(size_t i=0;i<S0_BO;i++) vals.push_back(bf2f(snewbo[i]));
    std::sort(vals.begin(), vals.end());
    float mn=vals.front(), mx=vals.back(), med=vals[vals.size()/2];
    // count near expected landmarks
    auto near=[](float v,float t)->bool{ return std::fabs(v-t)<0.125f; };
    long n_persist=near(med, S0VAL+K)?1:0, n_last=0, n_zero=0, n_nan=0;
    for(float v:vals){ if(!std::isfinite(v)) n_nan++; else if(near(v,S0VAL+K)) n_persist++; else if(near(v,S0VAL+1)) n_last++; else if(near(v,S0VAL)) n_zero++; }
    fprintf(stderr,"snew: min=%.4f med=%.4f max=%.4f  (S0=%.2f expect_persist=%.2f)\n", mn,med,mx, S0VAL, S0VAL+K);
    fprintf(stderr,"counts: persist(~%.2f)=%ld  last-only(~%.2f)=%ld  no-write(~%.2f)=%ld  nan/inf=%ld  /total=%zu\n",
            S0VAL+K, n_persist, S0VAL+1, n_last, S0VAL, n_zero, n_nan, vals.size());
    if(n_persist == (long)vals.size())      fprintf(stderr,"VERDICT: stack S PERSISTS (snew=S0+K). Bug is in passA/delta/par-reading, NOT S lifetime.\n");
    else if(n_last == (long)vals.size())    fprintf(stderr,"VERDICT: S does NOT persist between steps (only last write landed).\n");
    else if(n_zero == (long)vals.size())   fprintf(stderr,"VERDICT: passB never writes the stack S.\n");
    else if(n_nan)                          fprintf(stderr,"VERDICT: stack S clobbered to garbage (NaN/Inf). DMA/ObjectFifo corrupts the stack.\n");
    else                                    fprintf(stderr,"VERDICT: MIXED — see histogram. Sample first 16:");
    if(n_persist!=(long)vals.size() && n_last!=(long)vals.size() && n_zero!=(long)vals.size() && !n_nan){
        for(int i=0;i<16;i++) fprintf(stderr," %.3f", vals[i]); fprintf(stderr,"\n");
    }
    return 0;
}