// tools/qwopus_ffn_noop8.cpp — measures NPU 8-tile AGGREGATE weight-read DDR
// bandwidth (THE number that sets the NPU-FFN ceiling).  8 tiles each stream
// 6.27 MB (50.1 MB total) of weight packets with no h-array access; one vector
// load per row forces the DMA read.  Reports aggregate = total / time.
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

constexpr int HV_K=5120, GROUPS=HV_K/32, PAD_BYTES=18, ROW_BYTES=GROUPS*PAD_BYTES;
constexpr int RPB=4, N_TILE=2176, NPKT=N_TILE/RPB, WPKT=RPB*ROW_BYTES, NT=8;
constexpr size_t ONE_W = (size_t)NPKT*WPKT;
constexpr size_t W_TOTAL = NT*ONE_W;

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"[probe] MISSING %s\n",path.c_str());exit(5);}
    std::vector<char> d((size_t)f.tellg()); f.seekg(0); f.read(d.data(),d.size()); return d;
}
int main(int argc,char**argv){
    const char* dir=(argc>1)?argv[1]:"kernels";
    std::string DX=std::string(dir)+"/fst_qwopus_ffn_noop8.xclbin";
    std::string IX=std::string(dir)+"/fst_qwopus_ffn_noop8_insts.bin";
    xrt::device dev(0);
    xrt::xclbin xclb(DX); dev.register_xclbin(xclb); xrt::uuid uid=xclb.get_uuid();
    auto ctxp=std::make_unique<xrt::hw_context>(dev,uid);
    char* eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
        read_file(IX).data(),(uint32_t)read_file(IX).size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    if(!es||!eb){fprintf(stderr,"[probe] aiebu_get_elf FAILED\n");exit(2);}
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kn; for(auto&k:xclb.get_kernels()) kn=k.get_name();
    fprintf(stderr,"[probe] %s kernel=%s (elf %u B)\n",DX.c_str(),kn.c_str(),es);
    xrt::ext::kernel krnl(*ctxp,mod,kn);

    xrt::ext::bo boH(dev,(size_t)NT*HV_K*4);
    xrt::ext::bo boW(dev,W_TOTAL);
    xrt::ext::bo boO(dev,(size_t)NT*N_TILE*4);
    {auto*p=boH.map<float*>();memset(p,0,(size_t)NT*HV_K*4);boH.sync(XCL_BO_SYNC_BO_TO_DEVICE);}
    {auto*p=boW.map<float*>();memset(p,1,W_TOTAL);boW.sync(XCL_BO_SYNC_BO_TO_DEVICE);} // nonzero
    {auto*p=boO.map<float*>();memset(p,0,(size_t)NT*N_TILE*4);boO.sync(XCL_BO_SYNC_BO_TO_DEVICE);}

    fprintf(stderr,"[probe] 8-tile no-op: %.1f MB total weight across 8 tiles x3...\n",(double)W_TOTAL/1e6);
    double ms=0;
    for(int it=0; it<3; ++it){
        auto t0=std::chrono::steady_clock::now();
        auto run=krnl(3,0,0,static_cast<xrt::bo&>(boH),static_cast<xrt::bo&>(boW),static_cast<xrt::bo&>(boO));
        if(run.wait(60000)!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"[probe] TIMEOUT\n");return 3;}
        double d=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        fprintf(stderr,"[probe] iter %d = %.2f ms\n",it,d);
        if(it==2) ms=d;
    }
    double gbs=((double)W_TOTAL/1e9)/(ms/1e3);
    fprintf(stderr,"[probe] 8-tile AGGREGATE weight-read bandwidth = %.2f GB/s  (host FFN ~4.7 GB/s)\n",gbs);
    printf("aggregate8=%.2fGB/s total=%.1fMB steady=%.2fms\n",gbs,(double)W_TOTAL/1e6,ms);
    return 0;
}