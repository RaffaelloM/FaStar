// tools/qwopus_ffn_nibprobe.cpp — A3 vector-nibble correctness probe.
//
// Validates the hardest IRON element of the float-fifo NPU FFN path: the proven
// fused-FFN dequant pattern (aie::load_v-as-uint8 byte-perfect vector load +
// vector::operator[] register extract + scalar FP4[]) on a FLOAT-typed
// ObjectFifo, where the Phase-2 attempt failed only because it load_v'd as
// FLOAT (wrong reinterpret).  Loads fst_ffn_nibprobe_vec (the path under test)
// and fst_ffn_nibprobe_sca (scalar memory reads — the postmortem claimed these
// return a ramp on a float fifo; tested here), packs 16 weight bytes into the
// first 16 B of each 128-B float packet, and compares both to a host reference
// (FP4_LUT[nib]).
//
//   PASS = nib_vec matches host ref (the load_v-as-uint8 + operator[] dequant
//          works on a float fifo -> the float-fifo NPU FFN path is de-risked).
//   nib_sca is reported separately: if it ALSO matches, scalar memory reads of
//          a float fifo deliver the bytes (contradicts the postmortem's "ramp"
//          claim — a finding, not a failure).
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

constexpr int PKT_FLOATS = 32, OUT_FLOATS = 32, NPKT = 16;
constexpr int NBYTE = 16;   // weight bytes per packet

static const float FP4_LUT[16] = {
    0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
    0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f };

static double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static std::vector<char> read_file(const std::string&p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"[probe] MISSING %s\n",p.c_str());exit(5);}
    std::vector<char> d((size_t)f.tellg()); f.seekg(0); f.read(d.data(),d.size()); return d;
}
struct XK{ xrt::hw_context ctx; xrt::ext::kernel krnl; };
static XK load(xrt::device&dev,const std::string&xp,const std::vector<char>&insts){
    xrt::xclbin xclb(xp); dev.register_xclbin(xclb); xrt::uuid uid=xclb.get_uuid();
    auto ctxp=std::make_unique<xrt::hw_context>(dev,uid);
    char*eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    if(!es||!eb){fprintf(stderr,"[probe] aiebu FAILED %s\n",xp.c_str());exit(2);}
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kn; for(auto&k:xclb.get_kernels()) kn=k.get_name();
    fprintf(stderr,"[probe] %s kernel=%s (elf %u B)\n",xp.c_str(),kn.c_str(),es);
    xrt::ext::kernel krnl(*ctxp,mod,kn);
    return {std::move(*ctxp),std::move(krnl)};
}

// Compare an NPU output to the host ref; return max abs diff + argmax match.
static void cmp(const float*out,const std::vector<float>&ref,const char*tag,
                double&maxabs,int&argmatch)
{
    maxabs=0; int ref_am=0; float ref_mx=ref[0];
    for(size_t i=1;i<ref.size();i++) if(ref[i]>ref_mx){ref_mx=ref[i];ref_am=i;}
    int npu_am=0; float npu_mx=out[0];
    for(size_t i=1;i<ref.size();i++) if(out[i]>npu_mx){npu_mx=out[i];npu_am=i;}
    argmatch=(ref_am==npu_am)?1:0;
    int firstbad=-1;
    for(size_t i=0;i<ref.size();i++){
        double d=std::fabs((double)out[i]-ref[i]);
        if(d>maxabs)maxabs=d;
        if(firstbad<0 && d>1e-4) firstbad=(int)i;
    }
    fprintf(stderr,"[probe] %-7s max|Δ|=%.4e argmax ref=%d npu=%d %s  firstbad=%d\n",
            tag,maxabs,ref_am,npu_am,argmatch?"MATCH":"DIFF",
            firstbad);
}

int main(int argc,char**argv){
    const char*dir=(argc>1)?argv[1]:"kernels";
    xrt::device dev(0);
    XK kVec=load(dev,std::string(dir)+"/fst_ffn_nibprobe_vec.xclbin",
                 read_file(std::string(dir)+"/fst_ffn_nibprobe_vec_insts.bin"));
    XK kSca=load(dev,std::string(dir)+"/fst_ffn_nibprobe_sca.xclbin",
                 read_file(std::string(dir)+"/fst_ffn_nibprobe_sca_insts.bin"));

    // Build the float input BO: NPKT packets × 32 floats.  First 16 B of each
    // packet = 16 deterministic weight bytes (full 0..255 range); rest = 0.
    const size_t IN_BYTES=(size_t)NPKT*PKT_FLOATS*4;
    const size_t OUT_BYTES=(size_t)NPKT*OUT_FLOATS*4;
    std::vector<uint8_t> wbytes((size_t)NPKT*NBYTE);
    std::vector<float> inp(NPKT*PKT_FLOATS, 0.0f);
    for(int p=0;p<NPKT;p++)
        for(int i=0;i<NBYTE;i++){
            uint8_t b=(uint8_t)((p*13 + i*37) & 0xFF);   // deterministic, full-range
            wbytes[p*NBYTE+i]=b;
        }
    // Pack weight bytes into the first 16 B of each packet (raw byte view of float BO).
    uint8_t* inpB=reinterpret_cast<uint8_t*>(inp.data());
    for(int p=0;p<NPKT;p++)
        memcpy(inpB + (size_t)p*PKT_FLOATS*4, &wbytes[(size_t)p*NBYTE], NBYTE);

    // Host reference: 32 FP4 floats per packet ([16 low-nibble, 16 high-nibble]).
    std::vector<float> ref((size_t)NPKT*OUT_FLOATS);
    for(int p=0;p<NPKT;p++){
        const uint8_t*wb=&wbytes[(size_t)p*NBYTE];
        for(int i=0;i<NBYTE;i++){
            ref[p*OUT_FLOATS + i]      = FP4_LUT[wb[i] & 0x0F];
            ref[p*OUT_FLOATS + 16 + i] = FP4_LUT[(wb[i] >> 4) & 0x0F];
        }
    }

    xrt::ext::bo boIn(dev,IN_BYTES), boOut(dev,OUT_BYTES);
    {auto*p=boIn.map<float*>();memcpy(p,inp.data(),IN_BYTES);}
    boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run=[&](XK&k,const char*tag)->double{
        {auto*p=boOut.map<float*>();memset(p,0,OUT_BYTES);}
        boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        double s0=now_ms();
        k.krnl(3,0,0,static_cast<xrt::bo&>(boIn),static_cast<xrt::bo&>(boOut)).wait(60000);
        double d=now_ms()-s0;
        boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        fprintf(stderr,"[time] %s = %.2f ms\n",tag,d);
        return d;
    };

    double tVec=run(kVec,"nib_vec");
    const float* outV=boOut.map<float*>();
    double maV; int amV; cmp(outV,ref,"nib_vec",maV,amV);

    double tSca=run(kSca,"nib_sca");
    const float* outS=boOut.map<float*>();
    double maS; int amS; cmp(outS,ref,"nib_sca",maS,amS);

    bool vec_ok = (maV < 1e-4) && amV;        // the path under test matches ref
    bool sca_ok = (maS < 1e-4) && amS;        // scalar memory reads also work?
    bool pass = vec_ok;

    printf("vec_maxabs=%.4e vec_argmatch=%d sca_maxabs=%.4e sca_argmatch=%d "
           "vec_ms=%.2f sca_ms=%.2f %s\n",
           maV,amV,maS,amS,tVec,tSca,pass?"PASS":"FAIL");
    if(pass){
        fprintf(stderr,"[probe] PASS: load_v-as-uint8 + operator[] dequant works on a float fifo -> float-fifo NPU FFN path de-risked ✓\n");
        if(sca_ok) fprintf(stderr,"[probe] NOTE: scalar MEMORY reads ALSO match ref on this float fifo (contradicts the postmortem 'ramp' claim)\n");
        else       fprintf(stderr,"[probe] NOTE: scalar reads did NOT match (consistent with the postmortem 'ramp' claim)\n");
    } else {
        fprintf(stderr,"[probe] FAIL: vectorized dequant does NOT match ref (load_v-as-uint8 / operator[] broken on this float fifo)\n");
        return 4;
    }
    return 0;
}