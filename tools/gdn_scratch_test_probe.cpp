// tools/gdn_scratch_test_probe.cpp — Stage 1.1d persistent scratchpad probe.
// Fills N=8 input packets [addend=1, gdec=1], dispatches, drains N outputs.
// PERSISTENCE: the persistent S[0] = 0 then += 1 each of 8 calls -> last out[0] = 8.
//   If S did NOT persist (re-zeroed each call), every out[0] = 1.
// SPEED: latency per call should be ~stack-array speed (~1.6 ms), NOT 600x slow.
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
constexpr int N=8, INPKT=2, OUTPKT=1;
static std::vector<char> read_file(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);std::vector<char> d((size_t)f.tellg());f.seekg(0);f.read(d.data(),d.size());return d;}
int main(int argc,char**argv){
    const char* dir=(argc>1)?argv[1]:".";
    std::string DX=std::string(dir)+"/fst_gdn_scratch_test.xclbin";
    std::string IX=std::string(dir)+"/fst_gdn_scratch_test_insts.bin";
    std::vector<float> in((size_t)N*INPKT,0.0f);
    for(int c=0;c<N;c++){in[c*INPKT+0]=1.0f; in[c*INPKT+1]=1.0f;}   // addend=1, gdec=1
    xrt::device dev(0);
    xrt::xclbin xclb(DX); dev.register_xclbin(xclb); xrt::uuid uid=xclb.get_uuid();
    xrt::hw_context ctx(dev,uid);
    auto insts=read_file(IX); char* eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kn; for(auto&k:xclb.get_kernels()) kn=k.get_name();
    xrt::ext::kernel krnl(ctx,mod,kn);
    xrt::ext::bo boIn(dev,(size_t)N*INPKT*4), boOut(dev,(size_t)N*OUTPKT*4);
    {auto*p=boIn.map<float*>();memcpy(p,in.data(),in.size()*4);} boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    {auto*p=boOut.map<float*>();memset(p,0,(size_t)N*OUTPKT*4);} boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t0=std::chrono::steady_clock::now();
    auto run=krnl(3,0,0,static_cast<xrt::bo&>(boIn),static_cast<xrt::bo&>(boOut));
    if(run.wait(60000)!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"TIMEOUT\n");return 3;}
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE); float*out=boOut.map<float*>();
    fprintf(stderr,"per-call out[0]:");
    for(int c=0;c<N;c++) fprintf(stderr," %.0f", out[c*OUTPKT+0]);
    fprintf(stderr,"\n  latency=%.2f ms (8 re-calls, %.2f ms/call)\n", ms, ms/N);
    float last=out[(N-1)*OUTPKT+0];
    if(std::fabs(last-(float)N)<1e-3f){
        fprintf(stderr,"PASS: persistent scratchpad — last out[0]=%g == N=%d (S persists across re-calls)\n",last,N);
        if(ms < 50.0f) fprintf(stderr,"  AND FAST (%.2f ms, not the 600x Buffer slowdown)\n",ms);
        else fprintf(stderr,"  WARN: latency %.2f ms — check vs stack-array 1.6 ms (slow Buffer path?)\n",ms);
        return 0;
    }
    fprintf(stderr,"FAIL: last out[0]=%g != N=%d (S did NOT persist across re-calls)\n",last,N);return 1;
}