// tools/gdn_8kstack_probe.cpp — single-call 8 KB-stack internal-loop probe.
// in[384] = 48 v-heads x [gdec=0.5, rowoff=0, pad x6]; expects out[v*8+0]==264.0 for all 48.
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
constexpr int NVH=48, PKT=8, TOTAL=NVH*PKT;
static std::vector<char> read_file(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);std::vector<char> d((size_t)f.tellg());f.seekg(0);f.read(d.data(),d.size());return d;}
int main(int argc,char**argv){
    const char* dir=(argc>1)?argv[1]:".";
    std::string DX=std::string(dir)+"/fst_gdn_8kstack_micro.xclbin";
    std::string IX=std::string(dir)+"/fst_gdn_8kstack_micro_insts.bin";
    std::vector<float> in((size_t)TOTAL,0.0f);
    for(int v=0;v<NVH;v++){in[v*PKT+0]=0.5f; in[v*PKT+1]=0.0f;}
    xrt::device dev(0);
    xrt::xclbin xclb(DX); dev.register_xclbin(xclb); xrt::uuid uid=xclb.get_uuid();
    xrt::hw_context ctx(dev,uid);
    auto insts=read_file(IX); char* eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kn; for(auto&k:xclb.get_kernels()) kn=k.get_name();
    xrt::ext::kernel krnl(ctx,mod,kn);
    xrt::ext::bo boIn(dev,(size_t)TOTAL*4), boOut(dev,(size_t)TOTAL*4);
    {auto*p=boIn.map<float*>();memcpy(p,in.data(),in.size()*4);} boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    {auto*p=boOut.map<float*>();memset(p,0,(size_t)TOTAL*4);} boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t0=std::chrono::steady_clock::now();
    auto run=krnl(3,0,0,static_cast<xrt::bo&>(boIn),static_cast<xrt::bo&>(boOut));
    if(run.wait(60000)!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"TIMEOUT\n");return 3;}
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE); float*out=boOut.map<float*>();
    int nbad=0; float worst=0;
    for(int v=0;v<NVH;v++){float p=out[v*PKT+0];float d=std::fabs(p-264.0f);if(d>1e-3f){nbad++;if(d>worst)worst=d;}if(v<4||d>1e-3f)fprintf(stderr,"v %2d partial=%g (Δ=%g)\n",v,p,d);}
    fprintf(stderr,"latency=%.2f ms\n",ms);
    if(nbad==0){fprintf(stderr,"PASS: all 48 == 264.0 (8KB stack internal-loop OK)\n");return 0;}
    fprintf(stderr,"FAIL: %d/48 wrong (worst Δ=%g)\n",nbad,worst);return 1;
}