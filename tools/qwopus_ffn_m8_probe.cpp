// tools/qwopus_ffn_m8_probe.cpp — M=8 SPECULATIVE-DECODING FFN matvec probe.
// Validates fst_qwopus_ffn_m8.xclbin: 2x-dequant + native <8,8,4> MMUL across 8
// independent h-vectors, 16-tile N-split.  Compares each of the 8 NPU matvec
// outputs to its own host mxfp4_matvec_f32 reference (argmax gate per h).
//
// Packet (host-packed uint8, 12576 B): [32 hdr | h0..h7 (8*4096? =8192 BF16) |
// scales(256) | nibbles(4096)].  8 DISTINCT h-vectors so cross-contamination shows.
// Output: 8*N_TILE held fp32 per tile = o[m*N_TILE + idx]; 8 matvec results.
//
// Usage: qwopus_ffn_m8_probe [dir] [shape] [scrange]
//   Env: FST_FFN_XCLBIN (default fst_qwopus_ffn_m8), FST_FFN_NOOP_XCLBIN (default
//   fst_qwopus_ffn_m8_noop).  shape=up|down, scrange=narrow (default).
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

constexpr int M_n = 16, MH = 8, G = 16, HDR = 32;            // G_pkt=16
constexpr int H_BYTES   = MH * G * 32 * 2;                  // 8192 (8 bf16 h-vectors)
constexpr int SC_BYTES = M_n * G;                            // 256
constexpr int NB_BYTES = M_n * G * 16;                      // 4096
constexpr int W_BYTES  = SC_BYTES + NB_BYTES;
constexpr int PKT_BYTES = HDR + H_BYTES + W_BYTES;          // 12576
constexpr int NT = 16;
constexpr int N_TILE_COMPILED = 1088;

struct Shape { const char* name; int N,K,GROUPS,KCHUNKS,NCHUNKS,N_TILE_USED; };
static const Shape SH_UP  = {"up",  17408, 5120, 160, 10, 68, 1088};
static const Shape SH_DOWN = {"down", 5120,17408, 544, 34, 20, 320};

static float FP4_TABLE[16] = {0};
static inline uint16_t fp32_to_bf16_trunc(float f){uint32_t u;std::memcpy(&u,&f,4);return (uint16_t)(u>>16);}
static inline void pack_bf16_row(uint8_t* dst, const float* src, int n){
    for(int i=0;i<n;i++){uint16_t b=fp32_to_bf16_trunc(src[i]);dst[2*i]=(uint8_t)(b&0xFF);dst[2*i+1]=(uint8_t)(b>>8);}
}
static void mxfp4_matvec_f32(float* out, const float* h, const uint8_t* packed, int Nrows, int K){
    const int groups=K/32;
    #pragma omp parallel for schedule(static)
    for(int n=0;n<Nrows;n++){
        const uint8_t* row=packed+(size_t)n*groups*17; float acc=0.0f;
        for(int g=0;g<groups;g++){const uint8_t* blk=row+(size_t)g*17; uint8_t sc=blk[0]; if(sc==0)continue;
            float scale=std::ldexp(1.0f,(int)sc-127); const uint8_t* nb=blk+1; const float* hb=h+g*32;
            for(int i=0;i<32;i++){uint8_t byte=nb[i>>1];uint8_t nib=(i&1)?((byte>>4)&0xF):(byte&0xF);acc+=FP4_TABLE[nib]*scale*hb[i];}}
        out[n]=acc;
    }
}
static double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static std::vector<char> read_file(const std::string&p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"[probe] MISSING %s\n",p.c_str());exit(5);}
    std::vector<char> d((size_t)f.tellg()); f.seekg(0); f.read(d.data(),d.size()); return d;
}
struct XK{ xrt::hw_context ctx; xrt::ext::kernel krnl; };
static XK load(xrt::device&dev,const std::string&xp,const std::vector<char>&insts){
    xrt::xclbin xclb(xp); dev.register_xclbin(xclb); xrt::uuid uid=xclb.get_uuid();
    auto ctxp=std::make_unique<xrt::hw_context>(dev,uid); char*eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    if(!es||!eb){fprintf(stderr,"[probe] aiebu FAILED %s\n",xp.c_str());exit(2);}
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kn; for(auto&k:xclb.get_kernels()) kn=k.get_name();
    fprintf(stderr,"[probe] %s kernel=%s (elf %u B)\n",xp.c_str(),kn.c_str(),es);
    xrt::ext::kernel krnl(*ctxp,mod,kn); return {std::move(*ctxp),std::move(krnl)};
}
int main(int argc,char**argv){
    const char*dir=(argc>1)?argv[1]:"kernels";
    std::string shape_s=(argc>2)?argv[2]:"up";
    std::string sc_s=(argc>3)?argv[3]:"narrow";
    std::string full_stem=std::getenv("FST_FFN_XCLBIN")?std::getenv("FST_FFN_XCLBIN"):"fst_qwopus_ffn_m8";
    std::string noop_stem=std::getenv("FST_FFN_NOOP_XCLBIN")?std::getenv("FST_FFN_NOOP_XCLBIN"):"fst_qwopus_ffn_m8_noop";
    const Shape* S=(shape_s=="down")?&SH_DOWN:&SH_UP;
    const int N=S->N,K=S->K,GROUPS=S->GROUPS,KCHUNKS=S->KCHUNKS,NCHUNKS=S->NCHUNKS,N_TILE_USED=S->N_TILE_USED;
    const int NPKT=NCHUNKS*KCHUNKS;
    fprintf(stderr,"[probe] M=8 shape=%s N=%d K=%d GROUPS=%d KCHUNKS=%d NCHUNKS=%d N_TILE_USED=%d NPKT=%d\n",
            S->name,N,K,GROUPS,KCHUNKS,NCHUNKS,N_TILE_USED,NPKT);
    if(NPKT!=680){fprintf(stderr,"[probe] FATAL NPKT=%d != compiled 680\n",NPKT);return 6;}
    const float FP4[16]={0,0.5,1,1.5,2,3,4,6,0,-0.5,-1,-1.5,-2,-3,-4,-6};
    std::memcpy(FP4_TABLE,FP4,sizeof(FP4));
    xrt::device dev(0);
    XK kFull=load(dev,std::string(dir)+"/"+full_stem+".xclbin",  read_file(std::string(dir)+"/"+full_stem+"_insts.bin"));
    XK kNoop=load(dev,std::string(dir)+"/"+noop_stem+".xclbin",  read_file(std::string(dir)+"/"+noop_stem+"_insts.bin"));
    // 8 distinct h-vectors
    std::vector<std::vector<float>> h(MH, std::vector<float>(K));
    for(int m=0;m<MH;m++) for(int k=0;k<K;k++) h[m][k]=(float)((((k*37+m*101)%1000)-500))/500.0f;
    std::vector<uint8_t> W((size_t)N*GROUPS*17);
    for(int n=0;n<N;n++) for(int g=0;g<GROUPS;g++){uint8_t* blk=&W[(size_t)n*GROUPS*17+(size_t)g*17];
        uint8_t sc=(uint8_t)(120+((n*7+g*13)%21)); if(((n+g)%23)==0) sc=0; blk[0]=sc;
        for(int i=0;i<16;i++) blk[1+i]=(uint8_t)((n*3+g*5+i*11)&0xFF);}
    // 8 host refs
    std::vector<std::vector<float>> ref(MH, std::vector<float>(N));
    for(int m=0;m<MH;m++) mxfp4_matvec_f32(ref[m].data(), h[m].data(), W.data(), N, K);
    // pack input: 8 h's per packet
    const size_t IN_BYTES=(size_t)NT*NPKT*PKT_BYTES;
    const size_t OUT_BYTES=(size_t)NT*MH*N_TILE_COMPILED*4;
    std::vector<uint8_t> inp(IN_BYTES);
    for(int t=0;t<NT;t++) for(int nc=0;nc<NCHUNKS;nc++) for(int kc=0;kc<KCHUNKS;kc++){
        size_t pidx=(size_t)t*NPKT+(nc*KCHUNKS+kc); uint8_t* pkt=&inp[pidx*PKT_BYTES];
        int row_base=nc*M_n; std::memcpy(pkt,&row_base,4); std::memcpy(pkt+4,&kc,4); std::memset(pkt+8,0,HDR-8);
        for(int m=0;m<MH;m++) pack_bf16_row(pkt+HDR+m*(G*32*2), &h[m][(size_t)kc*G*32], G*32); // h0..h7
        uint8_t* sc_dst=pkt+HDR+H_BYTES; uint8_t* nb_dst=sc_dst+SC_BYTES;
        for(int r=0;r<M_n;r++){int gn=t*N_TILE_USED+nc*M_n+r; for(int g=0;g<G;g++){
            const uint8_t* blk=&W[(size_t)gn*GROUPS*17+(size_t)(kc*G+g)*17];
            sc_dst[r*G+g]=blk[0]; std::memcpy(nb_dst+(size_t)(r*G+g)*16, blk+1, 16);}}
    }
    xrt::ext::bo boIn(dev,IN_BYTES), boOut(dev,OUT_BYTES);
    {auto*p=boIn.map<uint8_t*>();std::memcpy(p,inp.data(),IN_BYTES);} boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=[&](XK&k,const char* tag)->double{
        {auto*p=boOut.map<float*>();std::memset(p,0,OUT_BYTES);} boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        double s0=now_ms(); k.krnl(3,0,0,static_cast<xrt::bo&>(boIn),static_cast<xrt::bo&>(boOut)).wait(120000);
        double d=now_ms()-s0; boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        fprintf(stderr,"[time] %s = %.2f ms (%.2f GB/s over %.1f MB)\n",tag,d,(double)IN_BYTES/d/1e6,(double)IN_BYTES/1e6); return d;
    };
    double tFull=run(kFull,"ffn_matvec_m8");
    const float* outF=boOut.map<float*>();
    std::vector<float> outFull(outF, outF+NT*MH*N_TILE_COMPILED);
    double tNoop=run(kNoop,"ffn_matvec_m8_noop");
    // gather 8 NPU outputs (compiled-stride read per h)
    std::vector<std::vector<float>> outNpu(MH, std::vector<float>(N));
    for(int m=0;m<MH;m++) for(int n=0;n<N;n++){int t=n/N_TILE_USED;int i=n%N_TILE_USED;
        outNpu[m][n]=outFull[(size_t)t*MH*N_TILE_COMPILED + (size_t)m*N_TILE_COMPILED + i];}
    double dma_gb=(double)IN_BYTES/tNoop/1e6, full_gb=(double)IN_BYTES/tFull/1e6, ratio=tFull/tNoop;
    printf("M8 shape=%s scrange=%s full_ms=%.2f noop_ms=%.2f ratio=%.2f full_gb=%.2f noop_gb=%.2f\n",
           S->name,sc_s.c_str(),tFull,tNoop,ratio,full_gb,dma_gb);
    int any_fail=0;
    for(int m=0;m<MH;m++){
        double maxabs=0;int ref_am=0,npu_am=0; float ref_mx=ref[m][0],npu_mx=outNpu[m][0];
        for(int n=1;n<N;n++){if(ref[m][n]>ref_mx){ref_mx=ref[m][n];ref_am=n;} if(outNpu[m][n]>npu_mx){npu_mx=outNpu[m][n];npu_am=n;}}
        for(int n=0;n<N;n++){double d=std::fabs((double)outNpu[m][n]-ref[m][n]); if(d>maxabs)maxabs=d;}
        double maxabs_ref=0; for(int n=0;n<N;n++) maxabs_ref=std::fmax(maxabs_ref,std::fabs(ref[m][n]));
        double relmax=maxabs_ref?maxabs/maxabs_ref:maxabs;
        bool argmatch=(ref_am==npu_am), rel_sane=(relmax<0.2);
        bool ok=argmatch&&rel_sane;
        if(!ok) any_fail=1;
        printf("  h%d: max|Δ|=%.4e rel=%.3e argmax %s (ref=%d npu=%d) -> %s\n",
               m,maxabs,relmax,argmatch?"MATCH":"DIFF",ref_am,npu_am,ok?"PASS":"FAIL");
        fprintf(stderr,"  h%d: max|Δ|=%.4e rel=%.3e argmax %s (ref=%d npu=%d) -> %s\n",
                m,maxabs,relmax,argmatch?"MATCH":"DIFF",ref_am,npu_am,ok?"PASS":"FAIL");
    }
    fprintf(stderr,"\n=== M=8 RESULT [%s/%s] ===\nfull matvec : %.2f ms (%.2f GB/s)\nno-op (DMA) : %.2f ms (%.2f GB/s)\ncompute/DMA = %.2fx\n",
            S->name,sc_s.c_str(),tFull,full_gb,tNoop,dma_gb,ratio);
    fprintf(stderr,"TARGET: full < 85.6 ms (host M=8).  %s\n", (tFull<85.6)?"BEATS HOST":"LOSES vs host");
    fprintf(stderr,"per-h cost: %.2f ms (full/8) vs host 10.7 ms/h\n", tFull/8.0);
    return any_fail?4:0;
}