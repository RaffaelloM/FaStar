// tools/gdn_8kslab_1tile_probe.cpp — 1-tile REAL-DMA isolation test.
// Runs the 1-tile slab kernel (fst_gdn_8kslab_1tile.xclbin) on REAL layer-0 GDN
// intermediates for ONE 32-col stripe (tile 0, cols 0..31).  Separates:
//   (a) 4-tile placement/memory pressure (43KB×4 ObjectFifo buffers) from
//   (b) the kernel's real-DMA parf reading + K-loop on-tile stack-S persistence.
// If 1-tile-real-DMA is correct + ~80ms ⇒ bug is 4-tile placement.
// If 1-tile-real-DMA ALSO blows up ⇒ bug is real-DMA parf / S-persistence.
//
//   XILINX_XRT=/usr ./tools/gdn_8kslab_1tile_probe <dump_dir> [start=0] [Nruns=5]
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
constexpr int PAR_SZ = 2*HV+COLS+8;   // 296 (padded to mult of 8 for par alignment)
constexpr int YD_SZ  = 2*COLS;        // 64
constexpr int S0_ELEM = HV*COLS;      // 4096
constexpr int PAR_ELEM = K*PAR_SZ;    // 2328
constexpr int YD_ELEM  = K*YD_SZ;     // 512

static inline uint16_t f2bf(float f){ uint32_t u; std::memcpy(&u,&f,4); uint32_t lsb=(u>>16)&1; uint32_t bias=0x7FFFu+lsb; return (uint16_t)((u+bias)>>16); }
static inline float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; std::memcpy(&f,&u,4); return f; }

static std::vector<float> load_f32(const std::string& p, size_t n){
    std::ifstream f(p, std::ios::binary); if(!f){ fprintf(stderr,"cannot open %s\n",p.c_str()); std::exit(2);}
    std::vector<float> v(n); f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n*4));
    return v;
}

// software recurrence restricted to ONE 32-col stripe (tile 0, cols [0..31]).
// S_stripe[HV][COLS], kn/qn/v full [HV] (c=kn·qn over full HV).
static void recur_sw_stripe(const float* S0_stripe, const std::vector<float>&kn,
                             const std::vector<float>&qn, const std::vector<float>&vvec,
                             const std::vector<float>&gdec, const std::vector<float>&beta,
                             float* S_out, float* y_out, int start){
    std::vector<float> S(S0_stripe, S0_stripe+(size_t)NVH*HV*COLS);
    std::vector<float> a(COLS), b(COLS), delta(COLS), yv(COLS);
    for(int t=0;t<K;t++){
        const int pos=start+t;
        for(int vh=0; vh<NVH; vh++){
            float* Sv  = S.data()  + (size_t)vh*HV*COLS;
            float* yvh  = y_out     + (size_t)t*NVH*COLS + (size_t)vh*COLS;
            const float* knv=kn.data() +(size_t)(pos-start)*NVH*HV+(size_t)vh*HV;
            const float* qnv=qn.data() +(size_t)(pos-start)*NVH*HV+(size_t)vh*HV;
            const float* vv =vvec.data()+(size_t)(pos-start)*NVH*HV+(size_t)vh*HV;
            float g=gdec[(pos-start)*NVH+vh], be=beta[(pos-start)*NVH+vh];
            float knb[HV], qnb[HV], vb[COLS];
            for(int d=0;d<HV;d++){ knb[d]=bf2f(f2bf(knv[d])); qnb[d]=bf2f(f2bf(qnv[d])); }
            for(int c=0;c<COLS;c++) vb[c]=bf2f(f2bf(vv[c]));
            for(int j=0;j<COLS;j++){ float aa=0,bb=0; for(int i=0;i<HV;i++){ aa+=Sv[i*COLS+j]*knb[i]; bb+=Sv[i*COLS+j]*qnb[i]; } a[j]=aa; b[j]=bb; }
            float c=0; for(int i=0;i<HV;i++) c+=knb[i]*qnb[i];
            for(int j=0;j<COLS;j++){ delta[j]=(vb[j]-g*a[j])*be; yvh[j]=g*b[j]+delta[j]*c; }
            for(int i=0;i<HV;i++) for(int j=0;j<COLS;j++) Sv[i*COLS+j]=bf2f(f2bf(g*Sv[i*COLS+j]+knb[i]*delta[j]));
        }
    }
    std::memcpy(S_out, S.data(), (size_t)NVH*HV*COLS*4);
}

static std::vector<char> read_file(const std::string& p){
    std::ifstream f(p, std::ios::binary|std::ios::ate); std::vector<char> d((size_t)f.tellg());
    f.seekg(0); f.read(d.data(), d.size()); return d;
}

int main(int argc, char** argv){
    const char* dir = (argc>1)?argv[1]:"/tmp/ssm_dump8";
    int start = (argc>2)?std::atoi(argv[2]):0;
    int NRUNS = (argc>3)?std::atoi(argv[3]):5;
    const int TILE0=0;   // cols [0..31]

    std::vector<float> kn, qn, vvec, gdec, beta;
    for(int p=start; p<start+K; p++){
        auto a=load_f32(std::string(dir)+"/kn_"+std::to_string(p)+".bin",  (size_t)NVH*HV);
        auto b=load_f32(std::string(dir)+"/qn_"+std::to_string(p)+".bin",  (size_t)NVH*HV);
        auto c=load_f32(std::string(dir)+"/vvec_"+std::to_string(p)+".bin",(size_t)NVH*HV);
        auto d=load_f32(std::string(dir)+"/gdec_"+std::to_string(p)+".bin",NVH);
        auto e=load_f32(std::string(dir)+"/beta_"+std::to_string(p)+".bin",NVH);
        kn.insert(kn.end(),a.begin(),a.end()); qn.insert(qn.end(),b.begin(),b.end());
        vvec.insert(vvec.end(),c.begin(),c.end()); gdec.insert(gdec.end(),d.begin(),d.end());
        beta.insert(beta.end(),e.begin(),e.end());
    }
    auto S_pre_full = load_f32(std::string(dir)+"/S_pre_"+std::to_string(start)+".bin",  (size_t)NVH*HV*HV);
    auto S_post_full= load_f32(std::string(dir)+"/S_post_"+std::to_string(start+K-1)+".bin",(size_t)NVH*HV*HV);

    // extract tile-0 stripe (cols 0..31) from full S
    std::vector<float> S0_stripe((size_t)NVH*HV*COLS);
    for(int v=0;v<NVH;v++) for(int i=0;i<HV;i++) for(int c=0;c<COLS;c++)
        S0_stripe[(size_t)v*HV*COLS+(size_t)i*COLS+c] = S_pre_full[(size_t)v*HV*HV+(size_t)i*HV+TILE0*COLS+c];
    std::vector<float> Sref_stripe((size_t)NVH*HV*COLS), yref_stripe((size_t)K*NVH*COLS);
    recur_sw_stripe(S0_stripe.data(), kn, qn, vvec, gdec, beta, Sref_stripe.data(), yref_stripe.data(), start);
    // shipped-dump stripe reference
    std::vector<float> Sship_stripe((size_t)NVH*HV*COLS);
    for(int v=0;v<NVH;v++) for(int i=0;i<HV;i++) for(int c=0;c<COLS;c++)
        Sship_stripe[(size_t)v*HV*COLS+(size_t)i*COLS+c] = S_post_full[(size_t)v*HV*HV+(size_t)i*HV+TILE0*COLS+c];

    // ---- pack 1-tile BOs (bf16) ----
    const size_t S0_BO  = (size_t)NVH*S0_ELEM;
    const size_t PAR_BO = (size_t)NVH*PAR_ELEM;
    const size_t YD_BO  = (size_t)NVH*YD_ELEM;
    std::vector<uint16_t> s0bo(S0_BO,0), parbo(PAR_BO,0), snewbo(S0_BO,0), ydbo(YD_BO,0);

    for(int v=0;v<NVH;v++){
        uint16_t* s0 = s0bo.data() + (size_t)v*S0_ELEM;
        for(int i=0;i<HV;i++) for(int c=0;c<COLS;c++)
            s0[i*COLS+c] = f2bf(S0_stripe[(size_t)v*HV*COLS+(size_t)i*COLS+c]);
        uint16_t* par = parbo.data() + (size_t)v*PAR_ELEM;
        for(int tk=0;tk<K;tk++){
            const int pos=start+tk;
            uint16_t* pt = par + (size_t)tk*PAR_SZ;
            const float* knv=kn.data()+(size_t)(pos-start)*NVH*HV+(size_t)v*HV;
            const float* qnv=qn.data()+(size_t)(pos-start)*NVH*HV+(size_t)v*HV;
            const float* vv =vvec.data()+(size_t)(pos-start)*NVH*HV+(size_t)v*HV;
            for(int d=0;d<HV;d++){ pt[d]=f2bf(knv[d]); pt[HV+d]=f2bf(qnv[d]); }
            for(int c=0;c<COLS;c++) pt[2*HV+c]=f2bf(vv[(size_t)TILE0*COLS+c]);
            float cc=0; for(int d=0;d<HV;d++) cc+=knv[d]*qnv[d];
            pt[2*HV+COLS]  = f2bf(gdec[(pos-start)*NVH+v]);
            pt[2*HV+COLS+1]= f2bf(beta[(pos-start)*NVH+v]);
            pt[2*HV+COLS+2]= f2bf(cc);
        }
    }

    // ---- run the 1-tile slab kernel ----
    fprintf(stderr,"[init] opening device...\n"); fflush(stderr);
    xrt::device dev(0);
    xrt::xclbin xclb(std::string("kernels/fst_gdn_8kslab_1tile.xclbin")); dev.register_xclbin(xclb);
    xrt::uuid uid=xclb.get_uuid(); xrt::hw_context ctx(dev,uid);
    auto insts=read_file("kernels/fst_gdn_8kslab_1tile_insts.bin"); char* eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kname; for(auto&k:xclb.get_kernels()) kname=k.get_name();
    fprintf(stderr,"[init] kernel name='%s' insts=%zu\n", kname.c_str(), insts.size()); fflush(stderr);
    xrt::ext::kernel krnl(ctx,mod,kname);
    xrt::ext::bo boS0(dev,S0_BO*2), boPar(dev,PAR_BO*2), boSn(dev,S0_BO*2), boYd(dev,YD_BO*2);
    auto* pS0=boS0.map<uint16_t*>(); auto* pPar=boPar.map<uint16_t*>();
    std::memcpy(pS0, s0bo.data(), S0_BO*2); std::memcpy(pPar, parbo.data(), PAR_BO*2);
    boS0.sync(XCL_BO_SYNC_BO_TO_DEVICE); boPar.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr,"[init] BOs ready: S0=%zuB Par=%zuB — launching kernel\n", S0_BO*2, PAR_BO*2); fflush(stderr);

    double best=1e9;
    for(int r=0;r<NRUNS+1;r++){
        auto t0=std::chrono::steady_clock::now();
        auto run=krnl(3,0,0, static_cast<xrt::bo&>(boS0), static_cast<xrt::bo&>(boPar),
                           static_cast<xrt::bo&>(boSn), static_cast<xrt::bo&>(boYd));
        if(run.wait(60000)!=ERT_CMD_STATE_COMPLETED){ fprintf(stderr,"TIMEOUT\n"); return 3; }
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        if(r>0 && ms<best) best=ms;
    }
    boSn.sync(XCL_BO_SYNC_BO_FROM_DEVICE); boYd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* pSn=boSn.map<uint16_t*>(); auto* pYd=boYd.map<uint16_t*>();
    std::memcpy(snewbo.data(), pSn, S0_BO*2); std::memcpy(ydbo.data(), pYd, YD_BO*2);

    // ---- unpack slab S_new + y (1-tile: cols 0..31) ----
    std::vector<float> slab_S((size_t)NVH*HV*COLS,0), slab_y((size_t)K*NVH*COLS,0);
    for(int v=0;v<NVH;v++){
        const uint16_t* sn=snewbo.data()+(size_t)v*S0_ELEM;
        for(int i=0;i<HV;i++) for(int c=0;c<COLS;c++)
            slab_S[(size_t)v*HV*COLS+(size_t)i*COLS+c]=bf2f(sn[i*COLS+c]);
        const uint16_t* yd=ydbo.data()+(size_t)v*YD_ELEM;
        for(int tk=0;tk<K;tk++){
            const uint16_t* yt=yd+(size_t)tk*YD_SZ;
            for(int c=0;c<COLS;c++)
                slab_y[(size_t)tk*NVH*COLS+(size_t)v*COLS+c]=bf2f(yt[c]);
        }
    }

    auto maxd=[](const float*a,const float*b,size_t n)->std::pair<float,float>{
        float mx=0,avg=0; for(size_t i=0;i<n;i++){ float d=std::fabs(a[i]-b[i]); if(d>mx)mx=d; avg+=d; } return {mx,avg/n}; };
    auto p_y = maxd(slab_y.data(), yref_stripe.data(), (size_t)K*NVH*COLS);
    auto p_S = maxd(slab_S.data(), Sref_stripe.data(), (size_t)NVH*HV*COLS);
    auto p_shipS = maxd(Sref_stripe.data(), Sship_stripe.data(), (size_t)NVH*HV*COLS); // sw vs shipped
    // per-token y max|Δ|
    fprintf(stderr,"=== gdn_8kslab 1-TILE REAL-DMA isolation (cols 0..31, K=%d, start=%d, NRUNS=%d) ===\n",K,start,NRUNS);
    fprintf(stderr,"1-tile dispatch latency = %.2f ms (best of %d)\n", best, NRUNS);
    fprintf(stderr,"sw-stripe vs shipped-stripe S max|Δ|=%.4e (replay fidelity)\n", p_shipS.first);
    fprintf(stderr,"slab vs sw-bf16S-stripe  y max|Δ|=%.4e  avg|Δ|=%.4e\n", p_y.first, p_y.second);
    fprintf(stderr,"slab vs sw-bf16S-stripe  S max|Δ|=%.4e  avg|Δ|=%.4e\n", p_S.first, p_S.second);
    fprintf(stderr,"[diag] per-token y max|Δ| slab-vs-sw:"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*COLS;i<(size_t)(tk+1)*NVH*COLS;i++){ float d=std::fabs(slab_y[i]-yref_stripe[i]); if(d>m)m=d;} fprintf(stderr," tk%d=%.3f",tk,m);} fprintf(stderr,"\n");
    { float ms=0,mr=0; for(size_t i=0;i<(size_t)NVH*HV*COLS;i++){ if(std::fabs(slab_S[i])>ms)ms=std::fabs(slab_S[i]); if(std::fabs(Sref_stripe[i])>mr)mr=std::fabs(Sref_stripe[i]); }
      float m0=0; for(size_t i=0;i<(size_t)NVH*HV*COLS;i++) if(std::fabs(S0_stripe[i])>m0)m0=std::fabs(S0_stripe[i]);
      float mship=0; for(size_t i=0;i<(size_t)NVH*HV*COLS;i++) if(std::fabs(Sship_stripe[i])>mship)mship=std::fabs(Sship_stripe[i]);
      fprintf(stderr,"[diag] max|S|  S0(load)=%.3f  slab(snew)=%.3f  sw_ref=%.3f  shipped(S_post)=%.3f\n",m0,ms,mr,mship); }
    // does slab snew ≈ S0 (persistence failure) ? report max|slab - S0|
    { float d=0; for(size_t i=0;i<(size_t)NVH*HV*COLS;i++){ float t=std::fabs(slab_S[i]-S0_stripe[i]); if(t>d)d=t;} fprintf(stderr,"[diag] max|slab_S - S0|=%.4e  (≈0 ⇒ slab S stayed at S0, persistence FAILED)\n",d); }
    int nfinite=0,nnan=0; for(float s:slab_y){ if(!std::isfinite(s))nnan++; else nfinite++; }
    fprintf(stderr,"[diag] slab_y finite=%d nan/inf=%d\n", nfinite, nnan);
    fprintf(stderr,"--- verdict ---\n");
    bool ok = p_y.first < 1e-2f && p_S.first < 1e-2f;
    if(ok) fprintf(stderr,"CORRECT: 1-tile real-DMA matches bf16-S software (Δ<1e-2) ⇒ bug is 4-TILE PLACEMENT/memory pressure, not the kernel/DMA-streaming.\n");
    else   fprintf(stderr,"BLOWN-UP: 1-tile real-DMA STILL wrong (Δ=%.2e) ⇒ bug is in real-DMA parf reading or K-loop stack-S persistence (kernel-level).\n", p_y.first);
    return 0;
}