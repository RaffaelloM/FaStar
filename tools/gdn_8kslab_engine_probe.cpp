// tools/gdn_8kslab_engine_probe.cpp — Phase 2 (correctness) + Phase 3 (latency)
// probe for the 4-tile bf16-S slab GDN engine kernel (fst_gdn_8kslab.xclbin).
//
// Inputs: REAL layer-0 GDN intermediates dumped by the engine with FST_SSM_DUMP
// (qn/kn/vvec/gdec/beta per position, S_pre/S_post/y_scan per position — the
// shipped fp32 3-pass path).  Packs 8 sequential tokens (start..start+7) for 48
// v-heads into the slab BO layout (4 tiles × 32-col slabs), computes c=kn·qn
// host-side, runs the slab kernel once, and unpacks S_new + y[8].
//
// Compares against TWO software references replayed on the same dumped inputs:
//   ref_fp32 : fp32 kn/qn/v + fp32 S + fp32 compute  (≈ shipped default path)
//   ref_bs   : bf16-rounded kn/qn/v + bf16-S storage (RNE) + fp32 compute (matches
//              the slab's precision choices, isolating aie-vs-host compute residual)
// Reports max|Δ| for y and S:
//   slab vs ref_fp32  = total bf16-S divergence vs the shipped path (expect non-zero)
//   slab vs ref_bs    = aie-vs-host fp32 compute residual (expect ~0 ⇒ math correct)
//   ref_fp32 vs dumped = replay fidelity (expect ~1e-6 ⇒ replay ≈ shipped)
// Plus per-v-head argmax(y) match rate (a downstream-argmax proxy) and slab
// dispatch latency (best of N) — Phase 3.
//
//   XILINX_XRT=/usr ./tools/gdn_8kslab_engine_probe <dump_dir> [start_pos=0] [Nruns=5]
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

constexpr int HV=128, COLS=32, NTILE=4, K=8, NVH=48, V=8;
constexpr int PAR_SZ = 2*HV+COLS+8;   // 296 (padded to mult of 8 for par alignment)
constexpr int YD_SZ  = 2*COLS;        // 64
constexpr int S0_ELEM = HV*COLS;      // 4096
constexpr int PAR_ELEM = K*PAR_SZ;    // 2328
constexpr int YD_ELEM  = K*YD_SZ;     // 512

static inline uint16_t f2bf(float f){ uint32_t u; std::memcpy(&u,&f,4); uint32_t lsb=(u>>16)&1; uint32_t bias=0x7FFFu+lsb; return (uint16_t)((u+bias)>>16); } // RNE
static inline uint16_t f2bf_trunc(float f){ uint32_t u; std::memcpy(&u,&f,4); return (uint16_t)(u>>16); } // engine convention u>>16 (toward zero)
static inline float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; std::memcpy(&f,&u,4); return f; }

static std::vector<float> load_f32(const std::string& p, size_t n){
    std::ifstream f(p, std::ios::binary); if(!f){ fprintf(stderr,"cannot open %s\n",p.c_str()); std::exit(2);}
    std::vector<float> v(n); f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n*4));
    return v;
}

// ---- software GDN recurrence (per v-head, K tokens).  S=[HV][HV] row-major ----
// bf16_S: round S to bf16 (RNE) after every passB; bf16_in: round kn/qn/v to bf16 first.
static void recur_sw(const float* S0, const std::vector<float>&kn, const std::vector<float>&qn,
                      const std::vector<float>&vvec, const std::vector<float>&gdec,
                      const std::vector<float>&beta, float* S_out, float* y_out,
                      int start, int bf16_S, bool bf16_in, int force_gdec_zero=0){
    // bf16_S: 0=fp32 S, 1=bf16-S RNE (slab kernel's to_vector<bfloat16>), 2=bf16-S TRUNC (engine u>>16)
    auto sstore=[bf16_S](float x)->float{
        if(bf16_S==1) return bf2f(f2bf(x));
        if(bf16_S==2) return bf2f(f2bf_trunc(x));
        return x; };
    std::vector<float> S(S0, S0+(size_t)NVH*HV*HV);
    if(bf16_S) for(float&x:S) x=sstore(x);
    std::vector<float> a(HV), b(HV), delta(HV), yv(HV);
    for(int t=0;t<K;t++){
        const int pos=start+t;
        for(int vh=0; vh<NVH; vh++){
            float* Sv = S.data() + (size_t)vh*HV*HV;
            float* yvh = y_out + (size_t)(pos-start)*NVH*HV + (size_t)vh*HV;
            const float* knv = kn.data() + (size_t)(pos-start)*NVH*HV + (size_t)vh*HV;
            const float* qnv = qn.data() + (size_t)(pos-start)*NVH*HV + (size_t)vh*HV;
            const float* vv  = vvec.data()+ (size_t)(pos-start)*NVH*HV + (size_t)vh*HV;
            float g=force_gdec_zero?0.0f:gdec[(pos-start)*NVH+vh], be=beta[(pos-start)*NVH+vh];
            float knb[HV], qnb[HV], vb[HV];
            for(int d=0;d<HV;d++){ knb[d]=bf16_in?bf2f(f2bf(knv[d])):knv[d];
                                  qnb[d]=bf16_in?bf2f(f2bf(qnv[d])):qnv[d];
                                  vb[d] =bf16_in?bf2f(f2bf(vv[d])) :vv[d]; }
            for(int j=0;j<HV;j++){ float aa=0,bb=0; for(int i=0;i<HV;i++){ aa+=Sv[i*HV+j]*knb[i]; bb+=Sv[i*HV+j]*qnb[i]; } a[j]=aa; b[j]=bb; }
            float c=0; for(int i=0;i<HV;i++) c+=knb[i]*qnb[i];
            for(int j=0;j<HV;j++){ delta[j]=(vb[j]-g*a[j])*be; yvh[j]=g*b[j]+delta[j]*c; }
            for(int i=0;i<HV;i++) for(int j=0;j<HV;j++) Sv[i*HV+j]=g*Sv[i*HV+j]+knb[i]*delta[j];
            if(bf16_S) for(int i=0;i<HV;i++) for(int j=0;j<HV;j++) Sv[i*HV+j]=sstore(Sv[i*HV+j]);
        }
    }
    std::memcpy(S_out, S.data(), (size_t)NVH*HV*HV*4);
}

static std::vector<char> read_file(const std::string& p){
    std::ifstream f(p, std::ios::binary|std::ios::ate); std::vector<char> d((size_t)f.tellg());
    f.seekg(0); f.read(d.data(), d.size()); return d;
}

int main(int argc, char** argv){
    const char* dir = (argc>1)?argv[1]:"/tmp/ssm_dump8";
    int start = (argc>2)?std::atoi(argv[2]):0;
    int NRUNS = (argc>3)?std::atoi(argv[3]):5;
    const std::string STEM = (argc>4)?argv[4]:"kernels/fst_gdn_8kslab";
    const int NDUMP = start + K;   // need positions [start..start+K-1] inputs + S_post[start+K-1]

    // load dumped inputs (fp32) for positions start..start+K-1
    std::vector<float> kn, qn, vvec, gdec, beta, S_pre0, y_dump, S_post_dump;
    for(int p=start; p<start+K; p++){
        auto a=load_f32(std::string(dir)+"/kn_"+std::to_string(p)+".bin",  (size_t)NVH*HV);
        auto b=load_f32(std::string(dir)+"/qn_"+std::to_string(p)+".bin",  (size_t)NVH*HV);
        auto c=load_f32(std::string(dir)+"/vvec_"+std::to_string(p)+".bin",(size_t)NVH*HV);
        auto d=load_f32(std::string(dir)+"/gdec_"+std::to_string(p)+".bin",NVH);
        auto e=load_f32(std::string(dir)+"/beta_"+std::to_string(p)+".bin",NVH);
        auto y=load_f32(std::string(dir)+"/y_scan_"+std::to_string(p)+".bin",(size_t)NVH*HV);
        kn.insert(kn.end(),a.begin(),a.end()); qn.insert(qn.end(),b.begin(),b.end());
        vvec.insert(vvec.end(),c.begin(),c.end()); gdec.insert(gdec.end(),d.begin(),d.end());
        beta.insert(beta.end(),e.begin(),e.end()); y_dump.insert(y_dump.end(),y.begin(),y.end());
    }
    S_pre0     = load_f32(std::string(dir)+"/S_pre_"+std::to_string(start)+".bin",  (size_t)NVH*HV*HV);
    S_post_dump= load_f32(std::string(dir)+"/S_post_"+std::to_string(start+K-1)+".bin",(size_t)NVH*HV*HV);

    // ---- software references ----
    std::vector<float> Sref_fp32((size_t)NVH*HV*HV), yref_fp32((size_t)NVH*HV*K);
    recur_sw(S_pre0.data(), kn, qn, vvec, gdec, beta, Sref_fp32.data(), yref_fp32.data(), start, 0,false);
    std::vector<float> Sref_bs((size_t)NVH*HV*HV),   yref_bs((size_t)NVH*HV*K);   // RNE (slab's to_vector<bfloat16>)
    recur_sw(S_pre0.data(), kn, qn, vvec, gdec, beta, Sref_bs.data(),   yref_bs.data(),   start, 1, true);
    std::vector<float> Sref_bt((size_t)NVH*HV*HV),   yref_bt((size_t)NVH*HV*K);   // TRUNC (engine u>>16)
    recur_sw(S_pre0.data(), kn, qn, vvec, gdec, beta, Sref_bt.data(),   yref_bt.data(),   start, 2, true);
    std::vector<float> Sref_g0((size_t)NVH*HV*HV),   yref_g0((size_t)NVH*HV*K);   // gdec forced to 0 (S doesn't accumulate)
    recur_sw(S_pre0.data(), kn, qn, vvec, gdec, beta, Sref_g0.data(),   yref_g0.data(),   start, 1, true, 1);

    // ---- pack slab BOs (bf16) ----
    // layouts: s0/snew [tile][v][i*32 + c]; par [tile][v][tk*PAR_SZ + (kn|qn|v|gdec|beta|c)]; yd [tile][v][tk*YD_SZ + (y|delta)]
    const size_t S0_BO  = (size_t)NTILE*NVH*S0_ELEM;
    const size_t PAR_BO = (size_t)NTILE*NVH*PAR_ELEM;
    const size_t YD_BO  = (size_t)NTILE*NVH*YD_ELEM;
    std::vector<uint16_t> s0bo(S0_BO,0), parbo(PAR_BO,0), snewbo(S0_BO,0), ydbo(YD_BO,0);

    for(int t=0;t<NTILE;t++){
        for(int v=0;v<NVH;v++){
            uint16_t* s0 = s0bo.data() + (size_t)t*NVH*S0_ELEM + (size_t)v*S0_ELEM;
            for(int i=0;i<HV;i++) for(int c=0;c<COLS;c++)
                s0[i*COLS+c] = f2bf(S_pre0[(size_t)v*HV*HV + (size_t)i*HV + (size_t)t*COLS + c]);
            uint16_t* par = parbo.data() + (size_t)t*NVH*PAR_ELEM + (size_t)v*PAR_ELEM;
            for(int tk=0;tk<K;tk++){
                const int pos=start+tk;
                uint16_t* pt = par + (size_t)tk*PAR_SZ;
                const float* knv=kn.data()+(size_t)(pos-start)*NVH*HV+(size_t)v*HV;
                const float* qnv=qn.data()+(size_t)(pos-start)*NVH*HV+(size_t)v*HV;
                const float* vv =vvec.data()+(size_t)(pos-start)*NVH*HV+(size_t)v*HV;
                for(int d=0;d<HV;d++){ pt[d]=f2bf(knv[d]); pt[HV+d]=f2bf(qnv[d]); }
                for(int c=0;c<COLS;c++) pt[2*HV+c]=f2bf(vv[(size_t)t*COLS+c]);
                float cc=0; for(int d=0;d<HV;d++) cc+=knv[d]*qnv[d];
                pt[2*HV+COLS]  = f2bf(gdec[(pos-start)*NVH+v]);
                pt[2*HV+COLS+1]= f2bf(beta[(pos-start)*NVH+v]);
                pt[2*HV+COLS+2]= f2bf(cc);
            }
        }
    }

    // ---- run the slab kernel ----
    xrt::device dev(0);
    xrt::xclbin xclb(STEM + ".xclbin"); dev.register_xclbin(xclb);
    xrt::uuid uid=xclb.get_uuid(); xrt::hw_context ctx(dev,uid);
    auto insts=read_file(STEM + "_insts.bin"); char* eb=nullptr;
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
    boSn.sync(XCL_BO_SYNC_BO_FROM_DEVICE); boYd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* pSn=boSn.map<uint16_t*>(); auto* pYd=boYd.map<uint16_t*>();
    std::memcpy(snewbo.data(), pSn, S0_BO*2); std::memcpy(ydbo.data(), pYd, YD_BO*2);

    // ---- unpack slab S_new + y ----
    std::vector<float> slab_S((size_t)NVH*HV*HV,0), slab_y((size_t)NVH*HV*K,0);
    for(int t=0;t<NTILE;t++){
        for(int v=0;v<NVH;v++){
            const uint16_t* sn=snewbo.data()+(size_t)t*NVH*S0_ELEM+(size_t)v*S0_ELEM;
            for(int i=0;i<HV;i++) for(int c=0;c<COLS;c++)
                slab_S[(size_t)v*HV*HV+(size_t)i*HV+(size_t)t*COLS+c]=bf2f(sn[i*COLS+c]);
            const uint16_t* yd=ydbo.data()+(size_t)t*NVH*YD_ELEM+(size_t)v*YD_ELEM;
            for(int tk=0;tk<K;tk++){
                const uint16_t* yt=yd+(size_t)tk*YD_SZ;
                for(int c=0;c<COLS;c++)
                    slab_y[(size_t)tk*NVH*HV+(size_t)v*HV+(size_t)t*COLS+c]=bf2f(yt[c]);
            }
        }
    }

    // ---- compare ----
    auto maxd=[](const float*a,const float*b,size_t n)->std::pair<float,float>{
        float mx=0,avg=0; for(size_t i=0;i<n;i++){ float d=std::fabs(a[i]-b[i]); if(d>mx)mx=d; avg+=d; } return {mx,avg/n}; };
    auto md_y=[&](const std::vector<float>&ref,const char*tag)->std::pair<float,float>{
        return maxd(slab_y.data(),ref.data(),(size_t)K*NVH*HV); };
    auto md_S=[&](const std::vector<float>&ref,const char*tag)->std::pair<float,float>{
        return maxd(slab_S.data(),ref.data(),(size_t)NVH*HV*HV); };

    auto p_fp_y=maxd(yref_fp32.data(),y_dump.data(),(size_t)K*NVH*HV);   // replay vs shipped
    auto p_fp_S=maxd(Sref_fp32.data(),S_post_dump.data(),(size_t)NVH*HV*HV);
    auto p_bs_y=maxd(slab_y.data(),yref_bs.data(),(size_t)K*NVH*HV);     // slab vs bf16-S-sw
    auto p_bs_S=maxd(slab_S.data(),Sref_bs.data(),(size_t)NVH*HV*HV);
    auto p_fp2slab_y=maxd(slab_y.data(),yref_fp32.data(),(size_t)K*NVH*HV); // slab vs shipped-fp32
    auto p_fp2slab_S=maxd(slab_S.data(),Sref_fp32.data(),(size_t)NVH*HV*HV);

    // per-v-head argmax(y) match: for each (token,v-head), does slab's max-y index match ref_fp32?
    int argmatch=0, argtot=0;
    for(int tk=0;tk<K;tk++) for(int v=0;v<NVH;v++){
        const float* yr=yref_fp32.data()+(size_t)tk*NVH*HV+(size_t)v*HV;
        const float* ys=slab_y.data()   +(size_t)tk*NVH*HV+(size_t)v*HV;
        int ir=0,is=0; for(int d=1;d<HV;d++){ if(yr[d]>yr[ir])ir=d; if(ys[d]>ys[is])is=d; }
        if(ir==is)argmatch++; argtot++;
    }

    fprintf(stderr,"=== gdn_8kslab engine probe (4 tile, 8KB stack, K=%d, start=%d, NRUNS=%d, stem=%s) ===\n",K,start,NRUNS,STEM.c_str());
    fprintf(stderr,"slab dispatch latency = %.2f ms (best of %d)\n", best, NRUNS);
    fprintf(stderr,"--- replay fidelity (sw ref_fp32 vs shipped dump) ---\n");
    fprintf(stderr,"  y max|Δ|=%.4e  avg|Δ|=%.4e\n", p_fp_y.first, p_fp_y.second);
    fprintf(stderr,"  S max|Δ|=%.4e  avg|Δ|=%.4e   (expect ~1e-6: replay ≈ shipped)\n", p_fp_S.first, p_fp_S.second);
    fprintf(stderr,"--- slab vs shipped-fp32 (total bf16-S divergence) ---\n");
    fprintf(stderr,"  y max|Δ|=%.4e  avg|Δ|=%.4e\n", p_fp2slab_y.first, p_fp2slab_y.second);
    fprintf(stderr,"  S max|Δ|=%.4e  avg|Δ|=%.4e\n", p_fp2slab_S.first, p_fp2slab_S.second);
    fprintf(stderr,"--- slab vs sw bf16-S (aie-vs-host compute residual; expect ~0 ⇒ math correct) ---\n");
    fprintf(stderr,"  y max|Δ|=%.4e  avg|Δ|=%.4e\n", p_bs_y.first, p_bs_y.second);
    fprintf(stderr,"  S max|Δ|=%.4e  avg|Δ|=%.4e\n", p_bs_S.first, p_bs_S.second);
    fprintf(stderr,"--- argmax proxy: per-(token,v-head) max-y index match = %d/%d (%.1f%%) ---\n",
            argmatch, argtot, 100.0*argmatch/argtot);
    // --- diagnostics: localize the divergence ---
    int nfinite=0,nbig=0,nnan=0; float worst=0; int wt=-1,wv=-1,wj=-1;
    for(int tk=0;tk<K;tk++) for(int v=0;v<NVH;v++) for(int j=0;j<HV;j++){
        float s=slab_y[(size_t)tk*NVH*HV+(size_t)v*HV+j];
        if(!std::isfinite(s)){nnan++; continue;} nfinite++;
        float d=std::fabs(s-yref_fp32[(size_t)tk*NVH*HV+(size_t)v*HV+j]);
        if(d>worst){worst=d;wt=tk;wv=v;wj=j;} if(d>1.0f)nbig++;
    }
    fprintf(stderr,"[diag] slab_y finite=%d nan/inf=%d  |Δ|>1 count=%d  worst@tk=%d v=%d j=%d\n",nfinite,nnan,nbig,wt,wv,wj);
    fprintf(stderr,"[diag] tk=0 v=0 y slab vs ref_fp32 (first 16 j):\n  slab:");
    for(int j=0;j<16;j++) fprintf(stderr," %.4f",slab_y[(size_t)0*NVH*HV+0*HV+j]); fprintf(stderr,"\n  ref :");
    for(int j=0;j<16;j++) fprintf(stderr," %.4f",yref_fp32[(size_t)0*NVH*HV+0*HV+j]); fprintf(stderr,"\n");
    // per-token y max|Δ|
    fprintf(stderr,"[diag] per-token y max|Δ| slab-vs-fp32:"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*HV;i<(size_t)(tk+1)*NVH*HV;i++){ float d=std::fabs(slab_y[i]-yref_fp32[i]); if(d>m)m=d;} fprintf(stderr," tk%d=%.3f",tk,m);} fprintf(stderr,"\n");
    fprintf(stderr,"[diag] per-token y max|Δ| ref_bs(RNE)-vs-fp32:"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*HV;i<(size_t)(tk+1)*NVH*HV;i++){ float d=std::fabs(yref_bs[i]-yref_fp32[i]); if(d>m)m=d;} fprintf(stderr," tk%d=%.3f",tk,m);} fprintf(stderr,"\n");
    fprintf(stderr,"[diag] per-token y max|Δ| ref_bt(TRUNC)-vs-fp32:"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*HV;i<(size_t)(tk+1)*NVH*HV;i++){ float d=std::fabs(yref_bt[i]-yref_fp32[i]); if(d>m)m=d;} fprintf(stderr," tk%d=%.3f",tk,m);} fprintf(stderr,"\n");
    fprintf(stderr,"[diag] per-token y max|Δ| slab-vs-ref_bt(TRUNC):"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*HV;i<(size_t)(tk+1)*NVH*HV;i++){ float d=std::fabs(slab_y[i]-yref_bt[i]); if(d>m)m=d;} fprintf(stderr," tk%d=%.3f",tk,m);} fprintf(stderr,"\n");
    // magnitude: is S exploding (|S| grows) or same-magnitude-wrong (corruption)?
    fprintf(stderr,"[diag] per-token max|y|  slab:"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*HV;i<(size_t)(tk+1)*NVH*HV;i++) if(std::fabs(slab_y[i])>m)m=std::fabs(slab_y[i]); fprintf(stderr," tk%d=%.2f",tk,m);} fprintf(stderr,"\n");
    fprintf(stderr,"[diag] per-token max|y|  ref :"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*HV;i<(size_t)(tk+1)*NVH*HV;i++) if(std::fabs(yref_fp32[i])>m)m=std::fabs(yref_fp32[i]); fprintf(stderr," tk%d=%.2f",tk,m);} fprintf(stderr,"\n");
    { float ms=0,mr=0; for(size_t i=0;i<(size_t)NVH*HV*HV;i++){ if(std::fabs(slab_S[i])>ms)ms=std::fabs(slab_S[i]); if(std::fabs(Sref_fp32[i])>mr)mr=std::fabs(Sref_fp32[i]); }
      fprintf(stderr,"[diag] final max|S|  slab=%.3f  ref=%.3f  (slab/ref ratio indicates explosion vs corruption)\n",ms,mr);
      float mg0=0; for(size_t i=0;i<(size_t)NVH*HV*HV;i++) if(std::fabs(Sref_g0[i])>mg0)mg0=std::fabs(Sref_g0[i]);
      fprintf(stderr,"[diag] final max|S|  ref_g0(gdec=0)=%.3f\n", mg0); }
    // ---- gdec=0 hypothesis: does slab ≈ ref_g0 (S doesn't accumulate)? ----
    auto p_g0_y=maxd(slab_y.data(),yref_g0.data(),(size_t)K*NVH*HV);
    auto p_g0_S=maxd(slab_S.data(),Sref_g0.data(),(size_t)NVH*HV*HV);
    fprintf(stderr,"[diag] gdec=0 hypothesis: slab vs ref_g0  y max|Δ|=%.3e  S max|Δ|=%.3e  (≈0 ⇒ slab sees gdec≈0)\n", p_g0_y.first, p_g0_S.first);
    fprintf(stderr,"[diag] per-token y max|Δ| slab-vs-ref_g0:"); for(int tk=0;tk<K;tk++){ float m=0; for(size_t i=(size_t)tk*NVH*HV;i<(size_t)(tk+1)*NVH*HV;i++){ float d=std::fabs(slab_y[i]-yref_g0[i]); if(d>m)m=d;} fprintf(stderr," tk%d=%.3f",tk,m);} fprintf(stderr,"\n");
    fprintf(stderr,"[diag] sample gdec[start..start+K-1] v0:"); for(int tk=0;tk<K;tk++) fprintf(stderr," %.4f", gdec[(start+tk)*NVH+0]); fprintf(stderr,"\n");
    fprintf(stderr,"[diag] sample beta[start..start+K-1] v0:"); for(int tk=0;tk<K;tk++) fprintf(stderr," %.4f", beta[(start+tk)*NVH+0]); fprintf(stderr,"\n");
    // per-tile S max|Δ| (does one 32-col stripe diverge?)
    fprintf(stderr,"[diag] per-tile S max|Δ|:"); for(int t=0;t<NTILE;t++){ float m=0; for(int v=0;v<NVH;v++) for(int i=0;i<HV;i++) for(int c=0;c<COLS;c++){ float d=std::fabs(slab_S[(size_t)v*HV*HV+(size_t)i*HV+(size_t)t*COLS+c]-Sref_fp32[(size_t)v*HV*HV+(size_t)i*HV+(size_t)t*COLS+c]); if(d>m)m=d;} fprintf(stderr," tile%d=%.3f",t,m);} fprintf(stderr,"\n");
    fprintf(stderr,"--- verdict ---\n");
    bool math_ok = p_bs_y.first < 1e-3f && p_bs_S.first < 1e-3f;
    bool replay_ok = p_fp_y.first < 1e-3f && p_fp_S.first < 1e-3f;
    if(math_ok && replay_ok)
        fprintf(stderr,"CORRECT: slab math matches bf16-S software (Δ<1e-3); bf16-S vs fp32 divergence is the expected precision gap. Proceed to argmax/latency assessment.\n");
    else
        fprintf(stderr,"SUSPECT: slab-vs-bf16S Δ=%.2e (math %s), replay Δ=%.2e (%s) — investigate before wiring.\n",
                p_bs_y.first, math_ok?"ok":"BAD", p_fp_y.first, replay_ok?"ok":"BAD");
    return 0;
}