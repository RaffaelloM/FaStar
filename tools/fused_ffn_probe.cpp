// tools/fused_ffn_probe.cpp — Gate 3 hardware validation of fst_fused_ffn_direct.
//
// Loads the raw-MLIR fused dequant+GEMM xclbin, builds its host control seq
// (insts.bin) via the npu_sequence API (validated byte-identical to IRON's
// rmsnorm output), assembles it to an ELF via aiebu, creates the xrt kernel,
// runs the single-tile probe (M=16 K=64 N=64) on hardware, and compares the
// NPU output C to a float32 CPU reference (dequant MXFP4 + GEMM).  Pass = cos>0.999.
//
// NO mockups: real xclbin, real insts, real hw run, real cos check.
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#include "npu_utils/npu_instr_utils.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include "aiebu/aiebu.h"

// ---- bf16 helpers (float<->bf16 bit tricks) ----
static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); uint32_t r=(u+0x8000)>>16; return (uint16_t)r; }
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

// ---- MXFP4 dequant (mirrors fst_fused_ffn_direct.cc) ----
static const float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static float e8m0_to_f(uint8_t s){
    if (s==0 || s==255) return 0.0f;
    // e8m0 power-of-2: bf16 = (uint16_t)s << 7
    uint16_t u = (uint16_t)s << 7; return b2f(u);
}
// W_q4 layout: [N][K/BLK][BLK_BYTES], N=64, K/BLK=2, BLK_BYTES=17, BLK=32
// B[k,n] = scale(n, k/32) * FP4_LUT[nibble]; nibble from W[n*34 + (k/32)*17 + 1 + k%32/2]
static float B_kn(const uint8_t* W, int k, int n){
    int blk_id = k/32, blk_off = k%32;
    const uint8_t* blk = W + (size_t)n*34 + (size_t)blk_id*17;
    float scale = e8m0_to_f(blk[0]);
    uint8_t byte = blk[1 + blk_off/2];
    float v = (blk_off%2==0) ? FP4_LUT[byte & 0x0F] : FP4_LUT[(byte>>4)&0x0F];
    return scale * v;
}

int main(int argc, char** argv){
    std::string mode = (argc>1) ? argv[1] : "fused";
    const bool do_rmsnorm = (mode == "rmsnorm");
    bool binsp = false; if (const char* e=getenv("FST_INSP")) binsp=atoi(e);
    constexpr int M=16, K=64, N=64;
    constexpr size_t W_BYTES = 2176, A_BYTES = 1024*sizeof(uint16_t), C_BYTES = 1024*sizeof(uint16_t);
    // FST_REP: MM2S repeat factor (size[0]) for the fused kernel. 1=repeat0
    // (one-shot, original), 2=repeat1, 8=repeat7 (rmsnorm-style).  C is always
    // one-shot: the core computes once -> 1 C produced regardless of MM2S feed.
    int rep = 1; if (const char* e=getenv("FST_REP")) rep=atoi(e);
    if (rep<1) rep=1;

    // ---- deterministic test data ----
    std::vector<uint8_t> W(W_BYTES);
    for (size_t i=0;i<W_BYTES;i++) W[i] = (uint8_t)((i*131 + 7) & 0xFF);
    // Scales: VISIBLE by default (130..133 -> 2^3..2^6 = 8..64) so dequanted B
    // prints non-zero.  FST_TINY_SCL=1 restores the original tiny 1..250 range.
    bool tiny_scl = false; if (const char* e=getenv("FST_TINY_SCL")) tiny_scl=atoi(e);
    for (int n=0;n<N;n++) for (int b=0;b<2;b++){
        uint8_t& s = W[n*34 + b*17];
        s = tiny_scl ? (uint8_t)(1 + ((n*7+b*3+1)%250))
                     : (uint8_t)(130 + ((n*7+b*3)%4));
    }
    std::vector<uint16_t> A(M*K);
    for (int i=0;i<M*K;i++) A[i] = f2b(((i%9)*0.125f - 0.5f));   // small range

    // ---- host control seq (insts.bin): generated for fused, shipped for rmsnorm ----
    std::vector<char> inst_raw;
    if (do_rmsnorm) {
        std::ifstream f("fst_ew_rmsnorm_insts.bin", std::ios::binary | std::ios::ate);
        inst_raw.resize((size_t)f.tellg());
        f.seekg(0); f.read(inst_raw.data(), inst_raw.size());
    } else {
        // Compiler-generated insts (fst_fused_ffn_rt.mlir runtime_sequence):
        // configures BOTH shim DMA (0x1d000/20/40 + 0x1d204/14/1c queues) AND
        // core DMA (0x21d000/20/40 + 0x21de04/0c/14 queues) + DDR_PATCH + token.
        // Matches the working npu-xrt dma_configure_task_lock reference.
        std::ifstream f("fst_fused_ffn_rt_insts.bin", std::ios::binary | std::ios::ate);
        if (!f) { fprintf(stderr,"[probe] MISSING fst_fused_ffn_rt_insts.bin\n"); return 5; }
        inst_raw.resize((size_t)f.tellg());
        f.seekg(0); f.read(inst_raw.data(), inst_raw.size());
    }

    // ---- aiebu: insts.bin -> ELF ----
    char* elf_buf=nullptr;
    uint32_t elf_sz = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        inst_raw.data(), (uint32_t)inst_raw.size(),
        NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
    if (elf_sz==0 || !elf_buf){ fprintf(stderr,"aiebu_get_elf FAILED\n"); return 2; }
    fprintf(stderr,"[probe] aiebu ELF %u bytes\n", elf_sz);

    // ---- xrt: device, xclbin, hw_context, module, kernel ----
    xrt::device dev(0);
    std::string xclb_path = do_rmsnorm ? "fst_ew_unified.xclbin" : "fst_fused_ffn_rt.xclbin";
    if (const char* e=getenv("FST_XCLB")) xclb_path = e;
    xrt::xclbin xclb(xclb_path);
    dev.register_xclbin(xclb);
    xrt::uuid uid = xclb.get_uuid();
    xrt::hw_context ctx(dev, uid);
    xrt::elf elf(elf_buf, elf_sz);
    xrt::module mod(elf);
    free(elf_buf);
    // single kernel symbol in the xclbin
    std::string kname;
    for (auto& k : xclb.get_kernels()) kname = k.get_name();
    fprintf(stderr,"[probe] kernel symbol: %s\n", kname.c_str());
    xrt::ext::kernel krnl(ctx, mod, kname);

    // ---- BOs + data ----
    xrt::ext::bo boD1(dev, 1<<20), boD2(dev, 1<<20);
    std::vector<uint16_t> C;
    std::vector<float> Cref;
    double cos=0, rmax=0; float maxdiff=0; int worst_m=0,worst_n=0;

    if (do_rmsnorm) {
        // rmsnorm [8,4096]: in, weight, out (each 32768 bf16 = 65536 B)
        constexpr int RN = 8*4096; constexpr size_t B = RN*2;
        std::vector<uint16_t> xin(RN), wt(RN);
        for (int i=0;i<RN;i++){ xin[i]=f2b(((i*131)%17)*0.1f - 0.8f); wt[i]=f2b(((i*7)%13)*0.1f + 0.2f); }
        xrt::ext::bo boI(dev,B), boWt(dev,B), boO(dev,B);
        auto* pi=boI.map<uint16_t*>(); auto* pw=boWt.map<uint16_t*>(); auto* po=boO.map<uint16_t*>();
        memcpy(pi,xin.data(),B); memcpy(pw,wt.data(),B); memset(po,0,B);
        boI.sync(XCL_BO_SYNC_BO_TO_DEVICE); boWt.sync(XCL_BO_SYNC_BO_TO_DEVICE); boO.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        fprintf(stderr,"[probe] dispatching rmsnorm...\n");
        double t0 = now();
        auto run = krnl(3,0,0, static_cast<xrt::bo&>(boI), static_cast<xrt::bo&>(boWt),
                        static_cast<xrt::bo&>(boO), static_cast<xrt::bo&>(boD1), static_cast<xrt::bo&>(boD2));
        ert_cmd_state st = run.wait(5000);
        fprintf(stderr,"[probe] rmsnorm wait state=%d after %.3fs\n",(int)st, now()-t0);
        if (st != ERT_CMD_STATE_COMPLETED){ fprintf(stderr,"[probe] FAILED state=%d\n",(int)st); return 3; }
        boO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        C.assign(RN,0); memcpy(C.data(),po,B);
        // CPU RMSNorm (per-row, d=4096): out = x/rms(x)*w, rms=sqrt(mean(x^2)+eps)
        Cref.assign(RN,0.0f);
        const float eps=1e-6f;
        for (int r=0;r<8;r++){
            double s=0; for(int i=0;i<4096;i++){ float x=b2f(xin[r*4096+i]); s+=x*x; }
            float rms=sqrtf((float)(s/4096.0)+eps);
            for(int i=0;i<4096;i++) Cref[r*4096+i]=b2f(xin[r*4096+i])/rms*b2f(wt[r*4096+i]);
        }
        for (int i=0;i<RN;i++){ float a=b2f(C[i]),b=Cref[i]; double d=std::fabs(a-b); if(d>maxdiff){maxdiff=d;worst_m=i/4096;worst_n=i%4096;} }
    } else {
        // Pad W/A BOs to rep× (fill data repeated) so a multi-iteration MM2S BD
        // (FST_REP>1) reads valid data each iter instead of past-BO garbage.  The
        // core computes once and consumes the first iter's data, so cos vs the
        // 1-iter CPU ref is unaffected.
        xrt::ext::bo boW(dev, (size_t)rep*W_BYTES);
        xrt::ext::bo boA(dev, (size_t)rep*A_BYTES);
        xrt::ext::bo boC(dev, C_BYTES);
        {auto* p=boW.map<char*>(); for(int r=0;r<rep;r++) memcpy(p+(size_t)r*W_BYTES, W.data(), W_BYTES);}
        boW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        {auto* p=boA.map<char*>(); for(int r=0;r<rep;r++) memcpy(p+(size_t)r*A_BYTES, A.data(), A_BYTES);}
        boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        {auto* c=boC.map<char*>(); memset(c,0,C_BYTES);} boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        fprintf(stderr,"[probe] dispatching fused kernel...\n");
        auto run = krnl(3, 0, 0,
            static_cast<xrt::bo&>(boW), static_cast<xrt::bo&>(boA),
            static_cast<xrt::bo&>(boC), static_cast<xrt::bo&>(boD1),
            static_cast<xrt::bo&>(boD2));
        double t0 = now();
        ert_cmd_state st = ERT_CMD_STATE_COMPLETED;
        try { st = run.wait(5000); }
        catch (const std::exception& e) {
            double dt = now()-t0;
            fprintf(stderr,"[probe] EXCEPTION after %.3fs: %s\n", dt, e.what());
            try { fprintf(stderr,"[probe] run state=%d\n",(int)run.state()); } catch(...){}
            return 3;
        }
        fprintf(stderr,"[probe] wait returned state=%d after %.3fs\n",(int)st, now()-t0);
        if (st != ERT_CMD_STATE_COMPLETED){ fprintf(stderr,"[probe] FAILED state=%d\n",(int)st); return 3; }
        boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        C.assign(M*N,0); memcpy(C.data(), boC.map<char*>(), C_BYTES);
        Cref.assign(M*N,0.0f);
        bool mmul1 = false; if (const char* e=getenv("FST_MMUL1")) mmul1=atoi(e);
        bool bkk   = false; if (const char* e=getenv("FST_KK"))   bkk=atoi(e);
        bool bramp = false; if (const char* e=getenv("FST_RAMP")) bramp=atoi(e);
        bool bkt   = false; if (const char* e=getenv("FST_KT"))   bkt=atoi(e);
        bool bneg  = false; if (const char* e=getenv("FST_NEG"))  bneg=atoi(e);
        bool bdump = false; if (const char* e=getenv("FST_DUMP")) bdump=atoi(e);
        bool bnt   = false; if (const char* e=getenv("FST_NT"))   bnt=atoi(e);
        bool bgk   = false; if (const char* e=getenv("FST_GK"))   bgk=atoi(e);
        int klim = K; if (const char* e=getenv("FST_KLIM")) klim=atoi(e)*8; if (klim>K) klim=K;
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){
            float acc=0.0f;
            if (mmul1)      { for (int k=0;k<K;k++) acc += b2f(A[m*K+k]) * 1.0f; }
            else if (bkk)   { for (int kt=0;kt<8;kt++) for (int k=0;k<8;k++) acc += b2f(A[m*K+kt*8+k]) * (float)k; }
            else if (bramp) { for (int kk=0;kk<K;kk++) acc += b2f(A[m*K+kk]) * (float)((kk%8)*4 + (n%4)); }
            else if (bkt)   { for (int kt=0;kt<8;kt++){ float rs=0; for(int k=0;k<8;k++) rs+=b2f(A[m*K+kt*8+k]); acc += (float)kt * rs; } }
            else if (bnt)   { float rs=0; for(int kk=0;kk<K;kk++) rs+=b2f(A[m*K+kk]); acc = (float)(n/4) * rs; }
            else if (bgk)  { float s1=0,s2=0; for(int kk=0;kk<K;kk++){ float a=b2f(A[m*K+kk]); s1+=a*(float)kk; s2+=a; } acc = 64.0f*s1 + (float)n*s2; }
            else if (bneg)  { // matches kernel D[] pattern: large/negative/zero, all tiles same
                static const float D[32] = {0,8,-8,16,-16,32,-32,64,-64,128,-128,384,-384,24,-24,12,
                                           -12,48,-48,96,-96,192,-192,16,-16,8,-8,4,-4,2,-2,1};
                for (int kt=0;kt<8;kt++) for (int k=0;k<8;k++) acc += b2f(A[m*K+kt*8+k]) * D[k*4 + (n%4)];
            }
            else if (bdump) { acc = 0.0f; }  // filled below
            else            { for (int k=0;k<klim;k++) acc += b2f(A[m*K+k]) * B_kn(W.data(), k, n); }
            Cref[m*N+n] = acc;
        }
        if (bdump) {
            // Kernel wrote C_out linear [0..1023] = 32 tiles, t=0..31 -> (kt=t/4, nt=(t%4)*5).
            // tile i (0..31) = B[kt*8 + i/4, nt*4 + i%4].  C is M*N row-major: L=m*64+n.
            for (int L=0; L<M*N; L++){
                int t=L/32, i=L%32;
                int kt=t/4, nt=(t%4)*5;
                Cref[L] = B_kn(W.data(), kt*8 + i/4, nt*4 + i%4);
            }
        }
        for (int i=0;i<M*N;i++){ float a=b2f(C[i]),b=Cref[i]; double d=std::fabs(a-b); if(d>maxdiff){maxdiff=d;worst_m=i/N;worst_n=i%N;} }
        // CPU B_kn reference for the dequant-only diagnostic (first tile kt=0,nt=0:
        // bbuf[k*4+n]=B[k,n]).  C_out[0..7] should equal B[k=0..1, n=0..3].
        fprintf(stderr,"[probe] CPU B_kn(k,0) for k=0..7 and A[1,k]:\n");
        for (int k=0;k<8;k++)
            fprintf(stderr,"  k=%d B[k,0]=%.4f  A[1,k]=%.4f  term=%.4f\n",
                k, B_kn(W.data(), k, 0), b2f(A[1*K+k]),
                B_kn(W.data(), k, 0)*b2f(A[1*K+k]));
        float sum10=0; for(int k=0;k<8;k++) sum10 += B_kn(W.data(),k,0)*b2f(A[1*K+k]);
        fprintf(stderr,"[probe] expected mul(m=1,n=0,kt=0) = %.4f\n", sum10);
    }

    // ---- compare ----
    if (binsp) {
        // kt=0-only mmul diagnostic.  Dumped: C_out[0..3]=mmul C[0,0..3];
        // C_out[960..991]=bbuf tile(0,0) (B[0..7,0..3]); C_out[992..999]=A[m=0,k=0..7].
        fprintf(stderr,"[insp] mmul C[0,0..3] (kt=0 only):");
        for (int n=0;n<4;n++) fprintf(stderr," %.3f", b2f(C[0*N+n]));
        fprintf(stderr,"\n");
        float Bd[8][4], Ad[8];
        for (int k=0;k<8;k++) for (int n=0;n<4;n++) Bd[k][n]=b2f(C[960 + k*4 + n]);
        for (int k=0;k<8;k++) Ad[k]=b2f(C[992 + k]);
        fprintf(stderr,"[insp] dumped A[m=0,k=0..7] (host):");
        for (int k=0;k<8;k++) fprintf(stderr," %.3f", b2f(A[0*K+k]));
        fprintf(stderr,"\n");
        fprintf(stderr,"[insp] dumped atile[0..7] (what mmul gathered):");
        for (int k=0;k<8;k++) fprintf(stderr," %.3f", b2f(C[1000 + k]));
        fprintf(stderr,"\n");
        for (int k=0;k<8;k++){
            fprintf(stderr,"  k=%d  B[k,0..3]=%.3f %.3f %.3f %.3f  A[0,k]=%.3f\n",
                k, Bd[k][0],Bd[k][1],Bd[k][2],Bd[k][3], Ad[k]);
        }
        fprintf(stderr,"[insp] expected C[0,n] = sum_k A[0,k]*B[k,n] (from DUMPED A,B):\n");
        for (int n=0;n<4;n++){
            float acc=0; for (int k=0;k<8;k++) acc += Ad[k]*Bd[k][n];
            fprintf(stderr,"  C[0,%d] expected=%.3f  npu=%.3f  %s\n",
                n, acc, b2f(C[0*N+n]),
                std::fabs(acc-b2f(C[0*N+n]))<1.0f?"OK":"MISMATCH");
        }
        fprintf(stderr,"[insp] expected C[0,n] from HOST B_kn + host A:\n");
        for (int n=0;n<4;n++){
            float acc=0; for (int k=0;k<8;k++) acc += b2f(A[0*K+k]) * B_kn(W.data(), k, n);
            fprintf(stderr,"  C[0,%d] host=%.3f  npu=%.3f\n", n, acc, b2f(C[0*N+n]));
        }
        return 0;
    }
    {
        size_t Nn = C.size(); double dot=0,na=0,nb=0;
        for (size_t i=0;i<Nn;i++){ float a=b2f(C[i]),b=Cref[i]; dot+=(double)a*b; na+=(double)a*a; nb+=(double)b*b; }
        cos = (na<1e-30||nb<1e-30)?0.0 : dot/(std::sqrt(na)*std::sqrt(nb));
        for (size_t i=0;i<Nn;i++){ double r=std::fabs(b2f(C[i]))/(std::fabs(Cref[i])+1e-12); if(r>rmax)rmax=r; }
    }
    fprintf(stderr,"[probe] %s  cos=%.6f  max|diff|=%.4g @[%d,%d]  |np|/|ref|=%.4f\n",
        mode.c_str(), cos, maxdiff, worst_m, worst_n, rmax);
    printf("cos=%.6f maxdiff=%.4g\n", cos, maxdiff);
    if (cos > 0.999){ fprintf(stderr,"[probe] PASS cos>0.999 ✓\n"); return 0; }
    fprintf(stderr,"[probe] FAIL cos<=0.999\n");
    for (int m=0;m<2;m++) for (int n=0;n<8;n++)
        fprintf(stderr,"  C[%d][%d] npu=%.4f ref=%.4f\n", m,n,b2f(C[m*(do_rmsnorm?4096:N)+n]),Cref[m*(do_rmsnorm?4096:N)+n]);
    return 4;
}