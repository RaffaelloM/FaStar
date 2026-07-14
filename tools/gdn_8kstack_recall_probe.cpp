// tools/gdn_8kstack_recall_probe.cpp — re-call threshold probe.
// Usage: gdn_8kstack_recall_probe <dir> <N>
// Dispatches the 8 KB-frame fn N times (range_(N)); each call does 48 v-heads.
// Checks all N*48 outputs == 264.0.  Reveals the re-call breakage threshold.
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
constexpr int NVH=48, PKT=8, TOTAL=NVH*PKT;
static std::vector<char> read_file(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);std::vector<char> d((size_t)f.tellg());f.seekg(0);f.read(d.data(),d.size());return d;}
int main(int argc,char**argv){
    const char* dir=(argc>1)?argv[1]:".";
    int N=(argc>2)?atoi(argv[2]):8;
    std::string DX=std::string(dir)+"/fst_gdn_8kstack_recall.xclbin";
    std::string IX=std::string(dir)+"/fst_gdn_8kstack_recall_insts.bin";
    std::vector<float> in((size_t)N*TOTAL,0.0f);
    for(int c=0;c<N;c++) for(int v=0;v<NVH;v++){in[c*TOTAL+v*PKT+0]=0.5f; in[c*TOTAL+v*PKT+1]=0.0f;}
    xrt::device dev(0);
    xrt::xclbin xclb(DX); dev.register_xclbin(xclb); xrt::uuid uid=xclb.get_uuid();
    xrt::hw_context ctx(dev,uid);
    auto insts=read_file(IX); char* eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kn; for(auto&k:xclb.get_kernels()) kn=k.get_name();
    xrt::ext::kernel krnl(ctx,mod,kn);
    xrt::ext::bo boIn(dev,(size_t)N*TOTAL*4), boOut(dev,(size_t)N*TOTAL*4);
    {auto*p=boIn.map<float*>();memcpy(p,in.data(),in.size()*4);} boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    {auto*p=boOut.map<float*>();memset(p,0,(size_t)N*TOTAL*4);} boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t0=std::chrono::steady_clock::now();
    auto run=krnl(3,0,0,static_cast<xrt::bo&>(boIn),static_cast<xrt::bo&>(boOut));
    if(run.wait(120000)!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"N=%d TIMEOUT\n",N);return 3;}
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE); float*out=boOut.map<float*>();
    int nbad=0; float worst=0; int worst_c=-1, first_bad_c=-1;
    for(int c=0;c<N;c++) for(int v=0;v<NVH;v++){
        float p=out[c*TOTAL+v*PKT+0]; float d=std::fabs(p-264.0f);
        if(d>1e-3f){nbad++; if(first_bad_c<0)first_bad_c=c; if(d>worst){worst=d;worst_c=c;}}
    }
    fprintf(stderr,"N=%2d: v0 per call:",N);
    for(int c=0;c<N&&c<16;c++) fprintf(stderr," %g",out[c*TOTAL+0]);
    fprintf(stderr,"\n  latency=%.2f ms  nbad=%d/%d",ms,nbad,N*NVH);
    if(nbad) fprintf(stderr,"  first_bad_call=%d worst_d=%g@call%d",first_bad_c,worst,worst_c);
    fprintf(stderr,"\n");
    if(nbad==0){fprintf(stderr,"PASS N=%d (all %d x 48 == 264.0)\n",N,N);return 0;}
    fprintf(stderr,"FAIL N=%d\n",N);return 1;
}