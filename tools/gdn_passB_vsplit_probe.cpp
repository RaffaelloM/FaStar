// tools/gdn_passB_vsplit_probe.cpp — Step 1: measure v-head-split passB latency
// vs the shipped 1-tile passB, on the SAME packed BO, same machine, same data.
//
// The shipped passB (fst_gdn_passB.xclbin, 1 worker, 768 packets) is DMA-bound on
// its single shim's S0-read+S2-write (~33.6 ms).  The v-head-split variants
// (fst_gdn_passB_vsplit_{4,8}t.xclbin) split the 48 v-heads across N tiles, each
// on its own shim columns ⇒ N× aggregate DMA bandwidth.  Math is byte-identical
// (gdn_passB_block reused verbatim), so S2 should match the shipped S_post dump.
//
// Inputs: FST_SSM_DUMP layer-0 GDN intermediates at /tmp/ssm_dump8.  For position
// `start`: S0 = S_pre[start]; delta = (v - gdec*a)*beta where a = S0ᵀ@kn (host,
// fp32, same as the shipped delta kernel); passB ⇒ S2, compared to S_post[start].
//
//   XILINX_XRT=/usr ./tools/gdn_passB_vsplit_probe <dump_dir> [start=0] [Nruns=5]
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

constexpr int HV=128, NVH=48, V=8, NVEC=HV/V;   // 16
constexpr int ROWS_PER_PKT=8, NPKT_V=HV/ROWS_PER_PKT;  // 16 blocks/v-head
constexpr int PKT_B=264, PKT_B_BLK=PKT_B*ROWS_PER_PKT; // 2112
constexpr int OUT_BLK=HV*ROWS_PER_PKT;                  // 1024
constexpr int NPKT=NVH*NPKT_V;                          // 768
constexpr int IN_TOTAL =NPKT*PKT_B_BLK;                 // 768*2112 fp32 (= NVH*HV*PKT_B)
constexpr int OUT_TOTAL=NPKT*OUT_BLK;                   // 768*1024  (= NVH*HV*HV)

static std::vector<float> load_f32(const std::string& p, size_t n){
    std::ifstream f(p, std::ios::binary); if(!f){ fprintf(stderr,"cannot open %s\n",p.c_str()); std::exit(2);}
    std::vector<float> v(n); f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n*4));
    return v;
}
static std::vector<char> read_file(const std::string& p){
    std::ifstream f(p, std::ios::binary|std::ios::ate); std::vector<char> d((size_t)f.tellg());
    f.seekg(0); f.read(d.data(), d.size()); return d;
}

// Run one xclbin variant on the pre-packed in-BO; return best latency (ms) and
// copy out-BO into `out`.  Returns -1 on timeout.
static double run_variant(xrt::device& dev, const std::string& xcl_path,
                          const std::string& inst_path, xrt::ext::bo& inBO, xrt::ext::bo& outBO,
                          std::vector<float>& out, int NRUNS){
    xrt::xclbin xclb(xcl_path); dev.register_xclbin(xclb);
    xrt::uuid uid=xclb.get_uuid(); xrt::hw_context ctx(dev,uid);
    auto insts=read_file(inst_path); char* eb=nullptr;
    uint32_t es=aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
        insts.data(),(uint32_t)insts.size(),NULL,0,(void**)&eb,NULL,0,"","",NULL,0);
    xrt::elf elf(eb,es); xrt::module mod(elf); free(eb);
    std::string kname; for(auto&k:xclb.get_kernels()) kname=k.get_name();
    xrt::ext::kernel krnl(ctx,mod,kname);

    double best=1e9;
    for(int r=0;r<NRUNS+1;r++){
        auto t0=std::chrono::steady_clock::now();
        auto run=krnl(3,0,0, static_cast<xrt::bo&>(inBO), static_cast<xrt::bo&>(outBO));
        if(run.wait(60000)!=ERT_CMD_STATE_COMPLETED){ fprintf(stderr,"  [TIMEOUT %s]\n",xcl_path.c_str()); return -1; }
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        if(r>0 && ms<best) best=ms;
    }
    outBO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* po=outBO.map<float*>();
    std::memcpy(out.data(), po, (size_t)OUT_TOTAL*4);
    return best;
}

int main(int argc, char** argv){
    const char* dir=(argc>1)?argv[1]:"/tmp/ssm_dump8";
    int start=(argc>2)?std::atoi(argv[2]):0;
    int NRUNS=(argc>3)?std::atoi(argv[3]):5;

    // ---- load inputs for position `start` ----
    auto S0 = load_f32(std::string(dir)+"/S_pre_"+std::to_string(start)+".bin",  (size_t)NVH*HV*HV);
    auto Spost=load_f32(std::string(dir)+"/S_post_"+std::to_string(start)+".bin",(size_t)NVH*HV*HV);
    auto kn  = load_f32(std::string(dir)+"/kn_"+std::to_string(start)+".bin",   (size_t)NVH*HV);
    auto qn  = load_f32(std::string(dir)+"/qn_"+std::to_string(start)+".bin",   (size_t)NVH*HV);
    auto vv  = load_f32(std::string(dir)+"/vvec_"+std::to_string(start)+".bin", (size_t)NVH*HV);
    auto gdec= load_f32(std::string(dir)+"/gdec_"+std::to_string(start)+".bin", NVH);
    auto beta= load_f32(std::string(dir)+"/beta_"+std::to_string(start)+".bin", NVH);

    // ---- compute delta host-side (fp32, same as shipped delta kernel) ----
    // a[j]=Σ_i S0[vh,i,j]*kn[vh,i]; delta[j]=(v[j]-gdec*a[j])*beta
    std::vector<float> delta((size_t)NVH*HV, 0.0f);
    std::vector<float> a((size_t)NVH*HV, 0.0f);
    for(int vh=0;vh<NVH;++vh){
        const float* Sv = S0.data() + (size_t)vh*HV*HV;
        const float* knv= kn.data() + (size_t)vh*HV;
        for(int j=0;j<HV;++j){ float aa=0; for(int i=0;i<HV;++i) aa+=Sv[i*HV+j]*knv[i]; a[vh*HV+j]=aa; }
        const float* vvec=vv.data()+(size_t)vh*HV; float g=gdec[vh], be=beta[vh];
        for(int j=0;j<HV;++j) delta[vh*HV+j]=(vvec[j]-g*a[vh*HV+j])*be;
    }

    // ---- pack the in-BO: 768 packets, v-head-major / block-major ----
    // packet k = (vh, b): 8 rows [b*8 .. b*8+7]; row = [S0_row(128)|delta(128)|kn_i(1)|gdec(1)|pad(5)] = 264
    std::vector<float> inBO(IN_TOTAL, 0.0f), outBO(OUT_TOTAL, 0.0f);
    for(int vh=0;vh<NVH;++vh){
        for(int b=0;b<NPKT_V;++b){
            const int k = vh*NPKT_V + b;
            float* pk = inBO.data() + (size_t)k*PKT_B_BLK;
            for(int r=0;r<ROWS_PER_PKT;++r){
                const int i = b*ROWS_PER_PKT + r;
                float* row = pk + (size_t)r*PKT_B;
                const float* srow = S0.data() + (size_t)vh*HV*HV + (size_t)i*HV;
                const float* dv    = delta.data() + (size_t)vh*HV;
                std::memcpy(row, srow, (size_t)HV*4);            // [0,128) S0_row
                std::memcpy(row+HV, dv, (size_t)HV*4);           // [128,256) delta
                row[2*HV+0] = kn[vh*HV + i];                     // 256 kn_i
                row[2*HV+1] = gdec[vh];                          // 257 gdec
                // [258..263] pad 0
            }
        }
    }

    xrt::device dev(0);
    xrt::ext::bo inB(dev,(size_t)IN_TOTAL*4), outB(dev,(size_t)OUT_TOTAL*4);
    { auto* pi=inB.map<float*>(); std::memcpy(pi, inBO.data(), (size_t)IN_TOTAL*4);
      inB.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    fprintf(stderr,"=== passB v-head-split A/B (pos=%d, NRUNS=%d, 48 v-heads, 768 pkts) ===\n",start,NRUNS);
    fprintf(stderr,"  in-BO=%.1f MB  out-BO=%.1f MB\n", (double)IN_TOTAL*4/1e6, (double)OUT_TOTAL*4/1e6);

    struct Var { const char* tag; const char* xcl; const char* inst; };
    Var vars[] = {
        {"1-tile SHIPPED", "kernels/fst_gdn_passB.xclbin",           "kernels/fst_gdn_passB_insts.bin"},
        {"4-tile v-split", "kernels/fst_gdn_passB_vsplit_4t.xclbin", "kernels/fst_gdn_passB_vsplit_4t_insts.bin"},
        {"8-tile v-split", "kernels/fst_gdn_passB_vsplit_8t.xclbin", "kernels/fst_gdn_passB_vsplit_8t_insts.bin"},
    };
    std::vector<float> ref_out; // first variant's out = reference (shipped)
    for(auto& v : vars){
        std::vector<float> out(OUT_TOTAL, 0.0f);
        double ms = run_variant(dev, v.xcl, v.inst, inB, outB, out, NRUNS);
        // correctness vs shipped S_post
        float mx=0,avg=0; int nbad=0;
        for(int vh=0;vh<NVH;++vh) for(int b=0;b<NPKT_V;++b) for(int r=0;r<ROWS_PER_PKT;++r){
            const int i=b*ROWS_PER_PKT+r; const int k=vh*NPKT_V+b;
            const float* orow = out.data() + (size_t)k*OUT_BLK + (size_t)r*HV;
            const float* srow = Spost.data()+ (size_t)vh*HV*HV + (size_t)i*HV;
            for(int j=0;j<HV;++j){ float d=std::fabs(orow[j]-srow[j]); if(d>mx)mx=d; avg+=d; if(d>1e-3f)nbad++; }
        }
        avg /= (double)NVH*HV*HV;
        fprintf(stderr,"  %-16s latency=%7.2f ms   S2 vs S_post: max|Δ|=%.4e avg|Δ|=%.4e bad=%d\n",
                v.tag, ms, mx, avg, nbad);
        if(ref_out.empty()) ref_out=out;
    }
    // cross-variant consistency (4t/8t vs shipped 1t)
    {
        std::vector<float> out4(OUT_TOTAL,0), out8(OUT_TOTAL,0);
        // re-run 4t and 8t to capture their outs (run_variant already overwrote; redo for diff)
        double m4=run_variant(dev, vars[1].xcl, vars[1].inst, inB, outB, out4, 1);
        double m8=run_variant(dev, vars[2].xcl, vars[2].inst, inB, outB, out8, 1);
        (void)m4;(void)m8;
        float d48=0,d18=0;
        for(size_t i=0;i<(size_t)OUT_TOTAL;++i){
            float a=std::fabs(out4[i]-out8[i]); if(a>d48)d48=a;
            float b=std::fabs(ref_out[i]-out8[i]); if(b>d18)d18=b;
        }
        fprintf(stderr,"  cross-variant: 4t-vs-8t max|Δ|=%.4e   1t-vs-8t max|Δ|=%.4e (expect ~0 ⇒ all byte-identical)\n", d48, d18);
    }
    fprintf(stderr,"--- verdict ---\n");
    return 0;
}