#include "fst_engine.h"
#include "fst_aiebu_cache.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <fstream>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_ini.h"
#include "xrt/experimental/xrt_ext.h"

static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static inline float bf16f(bf16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static inline bf16_t f2bf(float f){uint32_t b;memcpy(&b,&f,4);return(bf16_t)(b>>16);}

/* ── AVX2+FMA SIMD helpers for the host projection matvecs ────────────────
 * bf16 is the top 16 bits of fp32, so bf16→fp32 = zero-extend u16→u32 then
 * <<16 — bit-identical to bf16f() above.  The GEMM inner loop accumulates
 * 8 elements/iter with _mm256_fmadd_ps (fused multiply-add) and a tree
 * horizontal reduce.  This is NOT bit-identical to the serial k-loop (FMA +
 * tree reduction change the rounding/order), but these helpers have no DS4
 * caller (all HY3: attention q/k/v/o + dense + shared + NextN + host-FFN),
 * so DS4 stays byte-identical; HY3 argmax-stability is verified empirically. */
#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 bf16x8_to_f32x8(const bf16_t* p) {
    __m128i v16 = _mm_loadu_si128((const __m128i*)p);          /* 8 bf16 */
    __m256i v32 = _mm256_cvtepu16_epi32(v16);                  /* u16→u32 */
    return _mm256_castsi256_ps(_mm256_slli_epi32(v32, 16));    /* <<16 = fp32 */
}
static inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif
/* FP16 (IEEE half) → FP32 conversion.  Q8_0 scales are stored as FP16, NOT
 * BF16.  Interpreting them as BF16 (<<16) gives denormal garbage (1e-31). */
static inline float fp16f(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000)<<16;
    uint32_t exp=(h>>10)&0x1f;
    uint32_t mant=h&0x3ff;
    if(exp==0){if(mant==0)return sign?0.0f:0.0f;
        /* subnormal: convert to normal */
        uint32_t e=0;uint32_t m=mant;
        while(!(m&0x400)){m<<=1;e++;}
        m&=0x3ff;exp=1-e;
        uint32_t bits=sign|((exp+112)<<23)|(mant<<13);
        float f;memcpy(&f,&bits,4);return f;}
    if(exp==31)return sign?0.0f:0.0f; /* inf/nan → 0 for safety */
    uint32_t bits=sign|((exp+112)<<23)|(mant<<13);
    float f;memcpy(&f,&bits,4);return f;
}
static std::vector<uint8_t> pread(int fd,size_t sz,off_t off){std::vector<uint8_t> b(sz);ssize_t r=::pread(fd,b.data(),sz,off);(void)r;return b;}

struct FSTH{char m[4];uint32_t v,hd,nl,ne,tk,qh,kh,hd2,id,ns,r0;uint64_t vs;float re,rf;uint64_t sdo,sdc,eo,eb,es,et,r1,r2;};
struct FSTE{uint32_t tid;uint16_t lid,sid,qt,nd;uint32_t pad;uint64_t s0,s1,s2,off,sz,rsv;};

/* Proper IEEE-754 binary16 (fp16) -> float32 conversion.
 * The Q8_0 shared-tensor format stores per-block fp16 scales (uint16 bits).
 * The old code used (uint16<<16)->float, which is the BF16->float trick and is
 * WRONG for fp16: it produces denormals (~0) that trip the `sf<1e-12 -> 1.f`
 * fallback, dropping the scale entirely so the weight dequants to its raw int8
 * values (~1200x too large).  This caused the draft main_proj (mtp.0.main_proj,
 * Q8_0) to project a ~300x-too-large input into the draft, corrupting stage 0
 * (proj rms 314 vs correct 0.26).  See POST_MORTEM_RESUME_AIEBU. */
static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {  /* denormal: normalize into a normal float */
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant = (mant & 0x3ffu) << 13;
            f = sign | ((exp + (127 - 15 + 1)) << 23) | mant;
        }
    } else if (exp == 0x1fu) {
        f = sign | 0x7f800000u | (mant << 13);  /* inf/nan */
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);  /* normal */
    }
    float r; memcpy(&r, &f, 4);
    return r;
}

ModelWeightsC::~ModelWeightsC(){
    free(embedding_table);
    free(lm_head);
    free(final_norm);
    embedding_table=nullptr;
    lm_head=nullptr;
    final_norm=nullptr;
}

static void rms_cpu(bf16_t*o,const bf16_t*i,const bf16_t*w,int d,int n,float e=1e-6f){
    for(int t=0;t<n;t++){float ss=0;for(int j=0;j<d;j++){float v=bf16f(i[t*d+j]);ss+=v*v;}
        float r=1.f/sqrtf(ss/(float)d+e);for(int j=0;j<d;j++)o[t*d+j]=f2bf(bf16f(i[t*d+j])*r*bf16f(w[j]));}}
static void silu_cpu(bf16_t*o,const bf16_t*g,const bf16_t*u,int n){
    for(int i=0;i<n;i++){float x=bf16f(g[i]);o[i]=f2bf(0.5f*x*(1.f+tanhf(0.5f*x))*bf16f(u[i]));}}
/* Host reference GEMM for ARCH_HY3 (correctness-first, pre-NPU).  B is stored
 * native [N,K] row-major (= the .fst weight layout [out,in]), so out[m,n] =
 * Σ_k bf16(A[m,k])·bf16(B[n,k]) accumulated in fp32.  A is [M,K], B is [N,K],
 * out is [M,N].  fp32 output keeps q/k/v full precision through norm+RoPE. */
static void host_gemm_bnk_f32(float* out, const bf16_t* A, const bf16_t* B,
                              int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const bf16_t* arow = A + (size_t)m * K;
        float* orow = out + (size_t)m * N;
        /* Parallelize over output rows n (OpenMP) + vectorize the k-dot with
         * AVX2+FMA (8 elems/iter).  Each n writes a distinct orow[n] so the
         * OpenMP split is race-free; the SIMD accumulate is a tree reduction
         * (NOT bit-identical to the serial k-loop — see bf16x8_to_f32x8), but
         * this helper has no DS4 caller so DS4 stays byte-identical and the HY3
         * argmax is verified empirically.  This is the HY3 decode floor: the
         * q/k/v/o attention projections are M=1 matvecs (~107M MACs/layer). */
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const bf16_t* brow = B + (size_t)n * K;
            float s;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 af = bf16x8_to_f32x8(arow + k);
                __m256 bf = bf16x8_to_f32x8(brow + k);
                acc = _mm256_fmadd_ps(af, bf, acc);
            }
            s = hsum256(acc);
            for (; k < K; k++) s += bf16f(arow[k]) * bf16f(brow[k]);   /* tail */
#else
            s = 0.0f;
            for (int k = 0; k < K; k++) s += bf16f(arow[k]) * bf16f(brow[k]);
#endif
            orow[n] = s;
        }
    }
}
/* Same as host_gemm_bnk_f32 but A is fp32 (used by the o-projection, whose
 * input is the fp32 attention output).  B is still native [N,K] bf16. */
static void host_gemm_xnk_f32(float* out, const float* A, const bf16_t* B,
                              int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const float* arow = A + (size_t)m * K;
        float* orow = out + (size_t)m * N;
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const bf16_t* brow = B + (size_t)n * K;
            float s;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 af = _mm256_loadu_ps(arow + k);          /* A is fp32 */
                __m256 bf = bf16x8_to_f32x8(brow + k);
                acc = _mm256_fmadd_ps(af, bf, acc);
            }
            s = hsum256(acc);
            for (; k < K; k++) s += arow[k] * bf16f(brow[k]);
#else
            s = 0.0f;
            for (int k = 0; k < K; k++) s += arow[k] * bf16f(brow[k]);
#endif
            orow[n] = s;
        }
    }
}
/* Host fp32×fp32 GEMM for ARCH_HY3 routed-expert FFN (dequanted weights are
 * fp32).  A is [M,K], B is [N,K] native row-major (= [out,in]); out[m,n]=Σ_k A[m,k]·B[n,k]. */
static void host_gemm_f32f32(float* out, const float* A, const float* B,
                             int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const float* arow = A + (size_t)m * K;
        float* orow = out + (size_t)m * N;
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const float* brow = B + (size_t)n * K;
            float s;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 af = _mm256_loadu_ps(arow + k);
                __m256 bf = _mm256_loadu_ps(brow + k);
                acc = _mm256_fmadd_ps(af, bf, acc);
            }
            s = hsum256(acc);
            for (; k < K; k++) s += arow[k] * brow[k];
#else
            s = 0.0f;
            for (int k = 0; k < K; k++) s += arow[k] * brow[k];
#endif
            orow[n] = s;
        }
    }
}

/* ── Host MXFP4 dense-block dequant (ARCH_HY3 expert bank) ─────────────────
 * Inverse of fst_converter.py::_float32_to_dense_blocks.  Each 17-byte block
 * = 1 e8m0 scale byte + 16 nibble bytes (32 FP4 elements, low nibble first):
 *   element[i] = FP4_TABLE[nibble_i] × 2^(e8m0-127)   (e8m0=0 ⇒ dead block ⇒ 0)
 * Blocks group 32 consecutive elements along the IN dim: block (row r, group g)
 * covers in-elems [g*32, g*32+32).  packed is the expert-bank tensor slice;
 * out is fp32 [out_dim, in_dim] row-major. */
static const float FP4_TABLE[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                                   0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
static void dequant_mxfp4_dense(float* out, const uint8_t* packed,
                                int out_dim, int in_dim) {
    const int groups = in_dim / 32;
    for (int r = 0; r < out_dim; r++) {
        float* orow = out + (size_t)r * in_dim;
        for (int g = 0; g < groups; g++) {
            const uint8_t* blk = packed + (size_t)(r * groups + g) * 17;
            uint8_t sc = blk[0];
            float scale = (sc == 0) ? 0.0f : std::ldexp(1.0f, (int)sc - 127);
            const uint8_t* nb = blk + 1;
            for (int i = 0; i < 32; i++) {
                uint8_t byte = nb[i >> 1];
                uint8_t nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                orow[g * 32 + i] = FP4_TABLE[nib] * scale;
            }
        }
    }
}

/* HY3 sigmoid+bias router (host; the 192-wide GEMM is tiny vs the expert FFN).
 * router_w is [hd, n_experts]=[K,N] (transposed at load, matches router_gemm).
 * Selection score = sigmoid(logit)+bias (top-k); weight = UNBIASED sigmoid of
 * the winners, normalized ÷Σ then × ew_scale (HY3 gotcha: bias is selection-only). */
static void hy3_router_host(int* eids, float* wts, const bf16_t* hidden,
                            const float* router_w, const float* router_bias,
                            int M, int n_experts, int top_k, int hd, float ew_scale) {
    std::vector<float> s(n_experts);
    std::vector<int> ord(n_experts);
    for (int m = 0; m < M; m++) {
        const bf16_t* hm = hidden + (size_t)m * hd;
        for (int n = 0; n < n_experts; n++) {
            float l = 0.0f;
            for (int k = 0; k < hd; k++) l += bf16f(hm[k]) * router_w[(size_t)k * n_experts + n];
            s[n] = 1.0f / (1.0f + std::exp(-l));   /* sigmoid */
        }
        for (int i = 0; i < n_experts; i++) ord[i] = i;
        std::partial_sort(ord.begin(), ord.begin() + top_k, ord.end(), [&](int a, int b){
            float sa = s[a] + (router_bias ? router_bias[a] : 0.0f);
            float sb = s[b] + (router_bias ? router_bias[b] : 0.0f);
            return sa > sb;
        });
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) { eids[m * top_k + k] = ord[k]; wts[m * top_k + k] = s[ord[k]]; sum += s[ord[k]]; }
        float inv = ew_scale / (sum + 1e-12f);
        for (int k = 0; k < top_k; k++) wts[m * top_k + k] *= inv;
    }
}

/* SwiGLU on BF16 weights (dense L0 + always-on shared expert): out[M,hd] = down(SiLU(gate·h)·up·h).
 * gate/up are [inter,hd] native; down is [hd,inter] native.  WRITES out (caller adds). */
static void swiglu_bf16_host(const bf16_t* h, const bf16_t* gate, const bf16_t* up,
                             const bf16_t* down, int M, int hd, int inter, float* out) {
    std::vector<float> g((size_t)M * inter), u((size_t)M * inter), s((size_t)M * inter);
    host_gemm_bnk_f32(g.data(), h, gate, M, inter, hd);
    host_gemm_bnk_f32(u.data(), h, up,   M, inter, hd);
    for (size_t i = 0; i < (size_t)M * inter; i++) { float gv = g[i]; s[i] = (gv / (1.0f + std::exp(-gv))) * u[i]; }
    host_gemm_xnk_f32(out, s.data(), down, M, hd, inter);
}
/* SwiGLU on fp32 dequanted weights (routed experts): h_f is [M,hd] fp32;
 * gate/up [inter,hd], down [hd,inter], all fp32 dequanted.  WRITES out. */
static void swiglu_f32_host(const float* hf, const float* gate, const float* up,
                            const float* down, int M, int hd, int inter, float* out) {
    std::vector<float> g((size_t)M * inter), u((size_t)M * inter), s((size_t)M * inter);
    host_gemm_f32f32(g.data(), hf, gate, M, inter, hd);
    host_gemm_f32f32(u.data(), hf, up,   M, inter, hd);
    for (size_t i = 0; i < (size_t)M * inter; i++) { float gv = g[i]; s[i] = (gv / (1.0f + std::exp(-gv))) * u[i]; }
    host_gemm_f32f32(out, s.data(), down, M, hd, inter);
}
static void router_cpu(int*ids,float*wts,const bf16_t*hn,const float*rw,int M,int ne,int tk,int hd){
    for(int m=0;m<M;m++){std::vector<float> lg(ne);
        for(int e=0;e<ne;e++){float s=0;for(int d=0;d<hd;d++)s+=bf16f(hn[m*hd+d])*rw[e*hd+d];
            lg[e]=sqrtf(fmaxf(0.f,logf(1.f+expf(s))));}
        std::vector<int> ix(ne);std::iota(ix.begin(),ix.end(),0);
        std::partial_sort(ix.begin(),ix.begin()+tk,ix.end(),[&](int a,int b){return lg[a]>lg[b];});
        float sw=0;for(int k=0;k<tk;k++){ids[m*tk+k]=ix[k];wts[m*tk+k]=lg[ix[k]];sw+=wts[m*tk+k];}
        for(int k=0;k<tk;k++)wts[m*tk+k]/=(sw+1e-12f);}}

/* Transpose a [rows, cols] float vector in-place (for router weights). */
static void transpose_f32(std::vector<float>& v, int rows, int cols) {
    if ((size_t)rows * cols != v.size()) return;  /* weight not present */
    std::vector<float> tmp((size_t)rows * cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            tmp[(size_t)c * rows + r] = v[(size_t)r * cols + c];
    v = std::move(tmp);
}

/* Transpose a [rows, cols] BF16 vector in-place.  The GEMM kernels read
 * weight matrices as [K, N] row-major, but HF stores them as [out=N, in=K].
 * Reading [N, K] data as [K, N] yields wrong elements, so we transpose at
 * load time to get the correct [K, N] = [in, out] layout. */
static void transpose_bf16(std::vector<bf16_t, AlignedAllocator<bf16_t>>& v, int rows, int cols) {
    if ((size_t)rows * cols != v.size()) return;  /* weight not present in .fst */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> tmp((size_t)rows * cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            tmp[(size_t)c * rows + r] = v[(size_t)r * cols + c];
    v = std::move(tmp);
}

// ── NPU dispatch helpers ───────────────────────────────────────────────────
static void npu_sync_to(xrt::bo& bo) { bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
static void npu_sync_from(xrt::bo& bo) { bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); }

/* ── Dispatch packing helpers ──────────────────────────────────────────────
 * Replicate a base _insts.bin blob's op-words nrep times into ONE blob, patching
 * each copy's DDR_PATCH arg_off for arg_idx==1 (weight) and arg_idx==2 (output)
 * to w_offs[r]/o_offs[r].  This turns nrep separate XRT dispatches (one per
 * N-tile) into a single run_blob dispatch — proven byte-identical to the
 * per-dispatch path by tools/wqb_pack_probe on fst_mla_wqb.xclbin.  The blob
 * header is rewritten: instr_counts = nrep*base_cmds, seq[3] = total bytes. */
static std::vector<uint32_t> load_insts_words(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[pack] cannot open %s\n", path); return {}; }
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> w(sz / 4);
    f.read(reinterpret_cast<char*>(w.data()), sz);
    return w;
}

/* ── Kernel directory resolution ──────────────────────────────────────────
 * The engine loads _insts.bin / .xclbin by relative path.  FST_KERNEL_DIR
 * (or --kernel_dir) overrides the location; otherwise we look for a ./kernels
 * directory next to the binary, then ./kernels in the CWD, then fall back to
 * the CWD (legacy behaviour).  ZERO hard-coded absolute paths. */
static std::string resolve_kernel_dir() {
    if (const char* e = std::getenv("FST_KERNEL_DIR")) return e;
    struct stat st;
    if (::stat("kernels", &st) == 0 && S_ISDIR(st.st_mode)) return "kernels";
    return ".";
}
static std::string kpath(const char* name) {
    if (!name || !*name || name[0] == '/') return name ? std::string(name) : std::string();
    const std::string d = resolve_kernel_dir();
    if (d.empty() || d == ".") return std::string(name);
    return d + "/" + name;
}
static std::vector<uint32_t> replicate_packed_blob(const std::vector<uint32_t>& base,
                                                   int nrep,
                                                   const std::vector<uint32_t>& a_offs,
                                                   const std::vector<uint32_t>& w_offs,
                                                   const std::vector<uint32_t>& o_offs) {
    if (base.size() < 4) return {};
    const size_t hdr = 4;
    const size_t op_words = base.size() - hdr;
    const uint32_t base_cmds = base[2];
    std::vector<uint32_t> out;
    out.reserve(hdr + (size_t)nrep * op_words);
    out.push_back(base[0]); out.push_back(base[1]);
    out.push_back(base_cmds * (uint32_t)nrep);
    out.push_back((uint32_t)((hdr + (size_t)nrep * op_words) * 4));
    for (int r = 0; r < nrep; ++r) {
        size_t i = hdr;
        while (i < base.size()) {
            uint32_t op = base[i];
            size_t lines = (op == 1) ? 12 : (op == 0x81) ? 12 : (op == 0x80) ? 4
                      : (op == 3) ? 7 : (op == 0) ? 6 : 0;
            if (!lines) break;
            uint32_t arg_idx = (op == 0x81) ? base[i + 8] : 0;
            for (size_t k = 0; k < lines; ++k) {
                uint32_t word = base[i + k];
                if (op == 0x81 && k == 10) {
                    if (arg_idx == 0 && !a_offs.empty()) word = a_offs[r];
                    else if (arg_idx == 1) word = w_offs[r];
                    else if (arg_idx == 2) word = o_offs[r];
                }
                out.push_back(word);
            }
            i += lines;
        }
    }
    return out;
}

static void npu_run_wait(xrt::run& run) {
    /* 30 s ceiling: a healthy scalar AIE op finishes in well under 1 s,
     * so hitting this means the NPU deadlocked (wrong bank / context /
     * instruction mismatch) — throw a diagnosable error instead of
     * hanging the whole engine on a syncobj that will never fire. */
    ert_cmd_state s = run.wait(30000);
    if (s != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error(
            std::string("NPU kernel failed: state ") + std::to_string((int)s) +
            " (timeout=" + std::to_string((int)s) + ")");
}

// ── Async pipeline dependency barrier ───────────────────────────────────────
// Wait on every run that was start()ed but not yet waited on, then clear the
// pipeline.  Called at layer end and before reusing a BO that is in flight.
void FSTEngine::flush_pending_runs() {
    for (auto& r : pending_runs_) {
        ert_cmd_state s = r.wait();
        if (s != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error(
                std::string("async NPU run failed: state ") + std::to_string((int)s));
    }
    pending_runs_.clear();
}

// ── BO-to-BO async elementwise (NO host round-trip) ─────────────────────────
// silu / rmsnorm-style unary ops: in_bo (device) -> out_bo (device), pushed
// onto pending_runs_.  Caller must flush before reusing either BO.
void FSTEngine::npu_ew_async(const char* kname, xrt::bo& in_bo, xrt::bo& out_bo) {
    auto& krnl = kernel_cache_->get(kname);
    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(in_bo), static_cast<xrt::bo&>(out_bo));
    pending_runs_.push_back(std::move(run));
}

// mul-style binary op: in_a × in_b -> out_bo, all on device, async.
void FSTEngine::npu_ew_bin_async(const char* kname, xrt::bo& in_a, xrt::bo& in_b,
                                  xrt::bo& out_bo) {
    auto& krnl = kernel_cache_->get(kname);
    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(in_a), static_cast<xrt::bo&>(in_b),
        static_cast<xrt::bo&>(out_bo),
        static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
    pending_runs_.push_back(std::move(run));
}

// ── NPU GEMM with automatic N-tiling ────────────────────────────────────────
// AIE DMA BD dimensions must each be [1:64], so compiled kernels are capped at
// N=2048.  When N exceeds that, we tile along N.  B is [K, N] row-major, so
// each tile requires a strided copy (row by row) — NOT a flat memcpy.
void FSTEngine::npu_gemm(const char* kname, bf16_t* C, const bf16_t* A,
                          const bf16_t* B, int M, int N, int K) {
    constexpr int MAX_N_TILE = 2048;
    if (N <= MAX_N_TILE) {
        /* ── Single tile (fast path) ── */
        auto& krnl = kernel_cache_->get(kname);
        const size_t a_sz = (size_t)M * K * sizeof(bf16_t);
        const size_t b_sz = (size_t)K * N * sizeof(bf16_t);
        const size_t c_sz = (size_t)M * N * sizeof(bf16_t);

        if (bo_scratch_in_.size() < a_sz)
            bo_scratch_in_ = xrt::ext::bo(npu_device_, a_sz);
        if (bo_scratch_big_.size() < b_sz)
            bo_scratch_big_ = xrt::bo(npu_device_, b_sz, xrt::bo::flags::host_only, grp_);
        if (bo_scratch_out_.size() < c_sz)
            bo_scratch_out_ = xrt::ext::bo(npu_device_, c_sz);

        bool ov = std::getenv("FST_DISPATCH_TIME");
        double t_sync=0, t_wait=0, t_rb=0, t_tot = ov ? now() : 0;
        double s0 = ov ? now() : 0;
        memcpy(bo_scratch_in_.map<char*>(), A, a_sz); npu_sync_to(bo_scratch_in_);
        memcpy(bo_scratch_big_.map<char*>(), B, b_sz); npu_sync_to(bo_scratch_big_);
        memset(bo_scratch_out_.map<char*>(), 0, c_sz); npu_sync_to(bo_scratch_out_);
        if (ov) t_sync = now() - s0;

        double d0 = ov ? now() : 0;
        auto run = krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_scratch_big_),
            static_cast<xrt::bo&>(bo_scratch_out_),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
        flush_pending_runs();
        if (ov) t_wait = now() - d0;

        double r0 = ov ? now() : 0;
        npu_sync_from(bo_scratch_out_);
        memcpy(C, bo_scratch_out_.map<char*>(), c_sz);
        if (ov) { t_rb = now() - r0; t_tot = now() - t_tot;
            static int sc=0;
            if (sc++ < 40)
                fprintf(stderr, "[disp] %s M=%d N=%d K=%d  tot=%.2fms sync=%.2f wait=%.2f rb=%.2f\n",
                        kname, M, N, K, t_tot*1000, t_sync*1000, t_wait*1000, t_rb*1000);
        }
        return;
    }

    /* ── Multi-tile: tile along N with strided B/C copies ── */
    auto& krnl = kernel_cache_->get(kname);
    const int num_tiles = (N + MAX_N_TILE - 1) / MAX_N_TILE;
    const size_t a_sz = (size_t)M * K * sizeof(bf16_t);

    /* A is [M, K] — uploaded once, shared by all tiles. */
    if (bo_scratch_in_.size() < a_sz)
        bo_scratch_in_ = xrt::ext::bo(npu_device_, a_sz);
    memcpy(bo_scratch_in_.map<char*>(), A, a_sz);
    npu_sync_to(bo_scratch_in_);

    for (int t = 0; t < num_tiles; t++) {
        const int n_off = t * MAX_N_TILE;
        const int n_cur = std::min(MAX_N_TILE, N - n_off);
        const size_t b_sz = (size_t)K * n_cur * sizeof(bf16_t);
        const size_t c_sz = (size_t)M * n_cur * sizeof(bf16_t);

        if (bo_scratch_big_.size() < b_sz)
            bo_scratch_big_ = xrt::bo(npu_device_, b_sz, xrt::bo::flags::host_only, grp_);
        if (bo_scratch_out_.size() < c_sz)
            bo_scratch_out_ = xrt::ext::bo(npu_device_, c_sz);

        /* Strided copy: B is [K, N] row-major.  Tile t takes columns
         * [n_off .. n_off+n_cur) from every row. */
        char* b_dst = bo_scratch_big_.map<char*>();
        for (int k = 0; k < K; k++)
            memcpy(b_dst + (size_t)k * n_cur * sizeof(bf16_t),
                   B + (size_t)k * N + n_off,
                   n_cur * sizeof(bf16_t));
        npu_sync_to(bo_scratch_big_);

        memset(bo_scratch_out_.map<char*>(), 0, c_sz);
        npu_sync_to(bo_scratch_out_);

        auto run = krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_scratch_big_),
            static_cast<xrt::bo&>(bo_scratch_out_),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
        flush_pending_runs();

        npu_sync_from(bo_scratch_out_);
        /* Strided write-back: C is [M, N] row-major. */
        const char* c_src = bo_scratch_out_.map<char*>();
        for (int m = 0; m < M; m++)
            memcpy(C + (size_t)m * N + n_off,
                   c_src + (size_t)m * n_cur * sizeof(bf16_t),
                   n_cur * sizeof(bf16_t));
    }
}

void FSTEngine::npu_gemm(const char* kname, bf16_t* C, const bf16_t* A,
                          const xrt::bo& bo_B, size_t b_bytes, int M, int N, int K) {
    auto& krnl = kernel_cache_->get(kname);
    const size_t a_sz = (size_t)M * K * sizeof(bf16_t);
    const size_t c_sz = (size_t)M * N * sizeof(bf16_t);

    if (bo_scratch_in_.size() < a_sz)
        bo_scratch_in_ = xrt::ext::bo(npu_device_, a_sz);
    if (bo_scratch_out_.size() < c_sz)
        bo_scratch_out_ = xrt::ext::bo(npu_device_, c_sz);

    memcpy(bo_scratch_in_.map<char*>(), A, a_sz); npu_sync_to(bo_scratch_in_);
    npu_sync_to(bo_scratch_out_);

    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(bo_scratch_in_), const_cast<xrt::bo&>(bo_B),
        static_cast<xrt::bo&>(bo_scratch_out_),
        static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));

    pending_runs_.push_back(std::move(run));
    flush_pending_runs();

    npu_sync_from(bo_scratch_out_);
    memcpy(C, bo_scratch_out_.map<char*>(), c_sz);
}

// ── General GEMM on the kvc kernel (8,4096,512, K_div_k=64) ──────────────────
// C[M,N] = A[M,K] @ B[K,N], M=8.  The dedicated wq_b/oa/ob kernels (K_div_k=16)
// deadlock the DMA (state-8 timeout); kvc does not.  K is reduced in K_PAD=4096
// chunks — kvc performs the 4096-deep reduction internally, so for K>4096 we
// tile K and accumulate the per-chunk partials in float on the host.  N is
// tiled into 512-col slices.  Each (k,n) tile zero-pads to [8,4096]@[4096,512]
// -> [8,512]; zeros contribute nothing, so the result is exact.  B tile is
// host_only (kvc B-side BDs deadlock on device BOs).  Uses the reusable qk/sv
// scratch BOs — no per-call alloc, no extra hw_context, no recompile.  This is
// the O-projection (heads @ wo_a, low @ wo_b) and wq_b path.
//
// The IRON matmul_bf16 AIE kernel TAP previously shifted A per N-tile (A's
// N_div_n stride was m*k instead of 0), reading wrong K-columns for N-tile n>0
// and producing the 3.7x / sign-flip vs host float.  That TAP bug is fixed in
// compile_mla_unified.py (A's N stride = 0, matching the FFN kernel) and the
// kernels recompiled, so this NPU dispatch is now correct.  Verified by the
// wkv/wq_b/sv/oa CPU-ref audits in process_mla (ratio ~1.0, no sign flip).

// General GEMM on a DEDICATED MLA kernel (qk 8x512x128, sv 8x128x512) using the
// kernel's native K_pad/N_pad.  The kvc kernel (8x4096x512) misaccumulates when
// the real K is small (qk K=512, sv K=27) because of the 4096-deep K reduction
// over mostly-zero padding — empirically a 4.25x (qk) / 30.8x (sv) NPU-vs-CPU
// discrepancy.  The dedicated kernels reduce over K=512 (qk) / K=128 (sv) so
// the K-padding is at most 128, not 4096.  M tiled in 8; synchronous.

// ── BO-to-BO async GEMM (no host readback) ──────────────────────────────────
// Dispatch a non-tiled (N<=2048) GEMM that writes to a caller-supplied device
// output BO.  A is host data (uploaded to bo_scratch_in_), B is a device BO
// (already on device).  The run is pushed onto pending_runs_; caller flushes.
// For N>2048, falls back to the tiled readback npu_gemm (host C needed for
// strided N-tiling, so readback is unavoidable there).
void FSTEngine::npu_gemm_async(const char* kname, xrt::bo& bo_C,
                                const bf16_t* A, const xrt::bo& bo_B,
                                int M, int N, int K) {
    constexpr int MAX_N_TILE = 2048;
    if (N > MAX_N_TILE) {
        /* Tiled path: must use host intermediate for strided B/C copies.
         * Read back to a host buffer, then the caller can upload if needed. */
        std::vector<bf16_t, AlignedAllocator<bf16_t>> C_buf((size_t)M * N);
        npu_gemm(kname, C_buf.data(), A,
                 const_cast<xrt::bo&>(bo_B).map<bf16_t*>(),
                 M, N, K);
        /* Upload result to bo_C */
        const size_t c_sz = (size_t)M * N * sizeof(bf16_t);
        if (bo_C.size() < c_sz)
            bo_C = xrt::ext::bo(npu_device_, c_sz);
        memcpy(bo_C.map<char*>(), C_buf.data(), c_sz);
        npu_sync_to(bo_C);
        return;
    }
    auto& krnl = kernel_cache_->get(kname);
    const size_t a_sz = (size_t)M * K * sizeof(bf16_t);
    const size_t c_sz = (size_t)M * N * sizeof(bf16_t);

    if (bo_scratch_in_.size() < a_sz)
        bo_scratch_in_ = xrt::ext::bo(npu_device_, a_sz);
    if (bo_C.size() < c_sz)
        bo_C = xrt::ext::bo(npu_device_, c_sz);

    memcpy(bo_scratch_in_.map<char*>(), A, a_sz); npu_sync_to(bo_scratch_in_);
    memset(bo_C.map<char*>(), 0, c_sz); npu_sync_to(bo_C);

    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(bo_scratch_in_), const_cast<xrt::bo&>(bo_B),
        static_cast<xrt::bo&>(bo_C),
        static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
    pending_runs_.push_back(std::move(run));
}

// Fully on-device async GEMM: A, B, C are all device BOs, N<=2048.
// The MLA qc/kvc kernels this serves are compiled for a fixed M_TILE=8, so the
// M dimension is tiled here in chunks of exactly 8.  The AMDXDNA driver rejects
// sub-BO kernel args, so each 8-row tile is run into a reusable 8-row output BO
// (bo_mla_ctile_), read back, and scattered into bo_C.  This is synchronous
// (run.wait per tile) — correctness first; the async pipeline can return once
// the math is proven.  For M<=8 the original single-shot async path is kept.

// ── Vectorized MLA GEMM (canonical kernels.mm + zero, 100% aie::mmul) ────────
// One dispatch for all 4 MLA xclbins.  Each kernel is AOT-compiled for a fixed
// (M_c, K_c, N_c); the engine pads A/B to those dims (zeros contribute nothing)
// and M-tiles in M_c chunks.  bcol kernels (qc/wqb/ob) read B as [N_c, K_c]
// (native weight layout — NO transpose, matches the FFN fix); brow (qksv) reads
// B as [K_c, N_c].  B is a HOST buffer [N,K] (bcol) or [K,N] (brow), copied into
// a reusable host_only BO padded to [N_c,K_c]/[K_c,N_c].  Synchronous (run.wait
// per tile) — correctness first; the async pipeline can return once the math is
// proven.  NO CPU GEMM, NO scalar kernel.
//
// Kernel configs (must match compile_mla_vec.py MLA_DIMS):
//   qc   M_c=32  K_c=4096 N_c=1024 bcol  (qc, kvc[N-pad], oa — all K=4096,N=1024)
//   wqb  M_c=32  K_c=1024 N_c=2048 bcol  (wq_b, engine N-tiles 16x for N=32768)
//   ob   M_c=32  K_c=8192 N_c=2048 bcol  (ob,  engine N-tiles 2x  for N=4096)
//   qksv M_c=2048 K_c=512 N_c=512 brow  (qk[N-pad], sv[K-pad] — 1 call, no M-tile)
void FSTEngine::npu_gemm_mla_vec(const char* kname, bf16_t* C,
                                   const bf16_t* A, const bf16_t* B_host,
                                   int M, int N, int K,
                                   const xrt::bo* B_dev, size_t B_dev_off) {
    int M_c, K_c, N_c; bool bcol;
    /* qc is AOT-compiled with K_c=1024 (K_div_k=16) to stay under the NPU2
     * per-BD K-step limit; the engine host-loops the K=4096 reduction in
     * K_c-sized chunks accumulating partial C's in float32 (see K-loop below).
     * wqb/ob/qksv are compiled at their full K (K_c==K, 1 chunk, fp32-acc of a
     * single bf16 result is a no-op round trip). */
    /* M=16 standardization: all MLA projection kernels compiled at problem M=16
     * (tile m=16, 1 M-tile). qksv stays M=2048 (out of scope). */
    if      (!std::strcmp(kname, "qc"))   { M_c = 16;   K_c = 1024; N_c = 1024; bcol = true; }
    else if (!std::strcmp(kname, "wqb"))  { M_c = 16;   K_c = 1024; N_c = 2048; bcol = true; }
    else if (!std::strcmp(kname, "ob"))   { M_c = 16;   K_c = 1024; N_c = 2048; bcol = true; }
    else if (!std::strcmp(kname, "qksv")) { M_c = 2048; K_c = 512;  N_c = 512;  bcol = false; }
    /* qck: device-side K-split prototype -- compiled at FULL K (no host K-loop).
     * K_c==K so the K-loop below is one iteration (no split, exact round trip). */
    else if (!std::strcmp(kname, "qck"))  { M_c = 16;   K_c = 4096; N_c = 1024; bcol = true; }
    else { fprintf(stderr, "[mla] unknown kernel %s\n", kname); return; }

    auto& krnl = kernel_cache_->get(kname);
    const int M_pad = ((M + M_c - 1) / M_c) * M_c;
    const size_t a_tile_b = (size_t)M_c * K_c * sizeof(bf16_t);
    const size_t b_tile_b = bcol ? (size_t)N_c * K_c * sizeof(bf16_t)
                                 : (size_t)K_c * N_c * sizeof(bf16_t);
    const size_t c_tile_b = (size_t)M_c * N_c * sizeof(bf16_t);
    if (bo_mla_atile_.size() < a_tile_b) bo_mla_atile_ = xrt::ext::bo(npu_device_, a_tile_b);
    if (bo_mla_ctile_.size() < c_tile_b) bo_mla_ctile_ = xrt::ext::bo(npu_device_, c_tile_b);
    if (bo_mla_bhost_.size() < b_tile_b) bo_mla_bhost_ = xrt::bo(npu_device_, b_tile_b, xrt::bo::flags::host_only, grp_);

    /* Host K-split: K_c is the kernel's baked reduction dim; loop the actual K
     * in K_c chunks and accumulate partial C's in float32 (kernel zeros C each
     * call, so C_partial = A[:,k_chunk] @ B[k_chunk,:]; the fp32 sum is the full
     * GEMM).  When K==K_c this is one iteration and the fp32 round trip is exact.
     *
     * NOTE on async: an earlier attempt pipelined the K-chunks with ping-pong
     * BO sets (dispatch chunk k before waiting chunk k-1).  It measured ~1.5x
     * faster (0.04 -> 0.06 tok/s) but introduced a NON-DETERMINISTIC race --
     * same binary gave prefill-last 671 ("The") on one run and 29506 (garbage)
     * on the next, from npu_sync_to/sync_from on the shared MLA BOs contending
     * with in-flight NPU runs on the same hw_context.  The 1.5x was not worth
     * garbage output, so the K-loop is kept SERIAL (flush+readback per chunk).
     * The genuine async opportunity is limited: the single-chunk paths
     * (qck/wqb/oa, the majority) get no overlap from K-chunk pipelining, and
     * the multi-chunk path (ob, 8 chunks) is the only beneficiary -- ~25% of
     * per-layer flushes.  The 0.04 tok/s floor is the 43-layer NPU dispatch +
     * FFN dequant, not the per-GEMM wait. */
    std::vector<float> c_acc((size_t)M_pad * N, 0.0f);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> a_tile(M_c * K_c, 0);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> c_tile(M_c * N_c);
    const int Nn = std::min(N, N_c);

    /* Persistent-BO path: valid only for single-K-chunk kernels (K_c == K),
     * where the [N_c,K] tile is a contiguous sub-buffer of the device-resident
     * weight BO at B_dev_off — no host copy, no bo_mla_bhost_.sync.  K-split
     * kernels (K_c < K, e.g. ob) fall back to the host strided-copy path. */
    const bool use_dev = (B_dev != nullptr && *B_dev && K_c == K);

    for (int k_off = 0; k_off < K; k_off += K_c) {
        const int Kk = std::min(K_c, K - k_off);
        /* B tile BO for this k-chunk: persistent sub-buffer (no copy/sync) or
         * host strided copy into bo_mla_bhost_ + sync. */
        xrt::bo b_tile_bo;
        if (use_dev) {
            b_tile_bo = xrt::bo(*B_dev, b_tile_b, B_dev_off);
        } else {
            auto* b = bo_mla_bhost_.map<bf16_t*>();
            std::memset(b, 0, b_tile_b);
            if (bcol) {
                for (int n = 0; n < Nn; n++)
                    std::memcpy(b + (size_t)n * K_c, B_host + (size_t)n * K + k_off,
                                (size_t)Kk * sizeof(bf16_t));
            } else {
                for (int k = 0; k < Kk; k++)
                    std::memcpy(b + (size_t)k * N_c, B_host + (size_t)(k_off + k) * N,
                                (size_t)Nn * sizeof(bf16_t));
            }
            bo_mla_bhost_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            b_tile_bo = bo_mla_bhost_;
        }

        for (int mt = 0; mt < M_pad; mt += M_c) {
            const int m_cur = std::min(M_c, M - mt);
            /* A tile [M_c, K_c]: real rows m_cur, real cols Kk at k_off; zero-pad. */
            std::memset(a_tile.data(), 0, a_tile_b);
            for (int m = 0; m < m_cur; m++)
                std::memcpy(a_tile.data() + (size_t)m * K_c,
                            A + (size_t)(mt + m) * K + k_off,
                            (size_t)Kk * sizeof(bf16_t));
            std::memcpy(bo_mla_atile_.map<char*>(), a_tile.data(), a_tile_b);
            npu_sync_to(bo_mla_atile_);
            {auto* c = bo_mla_ctile_.map<bf16_t*>(); std::memset(c, 0, c_tile_b);}
            npu_sync_to(bo_mla_ctile_);
            auto run = krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_mla_atile_), static_cast<xrt::bo&>(b_tile_bo),
                static_cast<xrt::bo&>(bo_mla_ctile_),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
            flush_pending_runs();
            npu_sync_from(bo_mla_ctile_);
            std::memcpy(c_tile.data(), bo_mla_ctile_.map<char*>(), c_tile_b);
            for (int m = 0; m < m_cur; m++)
                for (int n = 0; n < Nn; n++)
                    c_acc[(size_t)(mt + m) * N + n] += bf16f(c_tile[(size_t)m * N_c + n]);
        }
    }
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C[(size_t)m * N + n] = f2bf(c_acc[(size_t)m * N + n]);
}

/* ── Random-matrix GEMM correctness probe (env FST_RANDMAT_TEST) ────────────
 * The ones@identity probe is useless for layout bugs: every permutation of the
 * identity gives the same result.  This feeds a RANDOM bf16 B [N,K] (row-major,
 * the exact layout npu_gemm_mla_vec / the FFN BO-to-BO path provide to the
 * b_col_maj kernels) through the real NPU GEMM and compares to a float32 CPU
 * reference C[m,n] = sum_k A[m,k]*B[n,k].  cos < 0.999 on a random matrix is a
 * LAYOUT bug (wrong tile stride / [N,K] vs [K,N] / wrong b_col_maj), never bf16
 * precision.  Tests the FFN "gemm" kernel (verified-correct baseline) and the
 * MLA qc/wqb/ob kernels (the cos-0.962 suspects).  Pure diagnostic, exits 0. */
void FSTEngine::randmat_test() {
    auto cos = [](const std::vector<float>&a, const std::vector<float>&b)->double{
        double da=0,db=0,dot=0;
        for(size_t i=0;i<a.size();i++){da+=(double)a[i]*a[i];db+=(double)b[i]*b[i];dot+=(double)a[i]*b[i];}
        return dot/(std::sqrt(da)*std::sqrt(db)+1e-30);
    };
    /* deterministic LCG so the test is reproducible (no Date/rand dependencies) */
    uint64_t st = 0x123456789abcdefULL;
    auto nxt = [&]()->uint32_t{ st ^= st<<13; st ^= st>>7; st ^= st<<17; return (uint32_t)(st>>32); };
    auto rbf = [&]()->bf16_t{ float f = ((float)(nxt()&0xffff)/65535.0f)*2.0f-1.0f; return f2bf(f); };

    const int M = 16;   // M=16 std: kernels compiled at M=16 (FFN probe calls them directly)
    struct Probe { const char* name; const char* kname; int K; int N; bool mla; };
    Probe probes[] = {
        /* FFN now runs through npu_gemm_mla_vec("ob") (host K-split, K_c=1024,
         * fp32 accum) -- the same path the engine uses for gate/up (K=4096) and
         * down (K=2048).  The old fused "gemm" kernel (K_div_k=128, cos 0.976)
         * is no longer on the FFN path. */
        {"ffn_gate/up (ob K-split)", "ob", 4096, 2048, true},
        {"ffn_down    (ob K-split)", "ob", 2048, 2048, true},
        {"qc   (MLA wq_a,   b_col_maj)",   "qc",   4096, 1024, true},
        {"wqb  (MLA wq_b,   b_col_maj)",   "wqb",  1024, 2048, true},
        {"ob   (MLA wo_b,   b_col_maj)",  "ob",   8192, 2048, true},
        /* Device-side K fix (Step 1): full K=4096 in ONE call via k_tile=128
         * (K_div_k=32).  cos~0.998 |ratio|~1.003 -- layout-correct, no host K-loop. */
        {"qck  (device k128, full K)",     "qck",  4096, 1024, true},
    };
    for (const auto& p : probes) {
        const int K=p.K, N=p.N;
        std::vector<bf16_t, AlignedAllocator<bf16_t>> A((size_t)M*K), B((size_t)N*K), C((size_t)M*N, 0);
        for (auto& v : A) v = rbf();
        for (auto& v : B) v = rbf();
        if (p.mla) npu_gemm_mla_vec(p.kname, C.data(), A.data(), B.data(), M, N, K);
        else       npu_gemm(        p.kname, C.data(), A.data(), B.data(), M, N, K);
        std::vector<float> cn((size_t)M*N), cref((size_t)M*N);
        double me=0; int wr=0,wn=0; float wa=0;
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){
            double acc=0;
            for (int k=0;k<K;k++) acc += (double)bf16f(A[(size_t)m*K+k]) * (double)bf16f(B[(size_t)n*K+k]);
            float ref=(float)acc, npu=bf16f(C[(size_t)m*N+n]);
            cn[(size_t)m*N+n]=npu; cref[(size_t)m*N+n]=ref;
            float e=std::fabs(npu-ref); if(e>me){me=e;wr=m;wn=n;}
            if(std::fabs(npu)>wa) wa=std::fabs(npu);
        }
        double c = cos(cn, cref);
        double nn=0,nr=0; for(size_t i=0;i<cn.size();i++){nn+=(double)cn[i]*cn[i];nr+=(double)cref[i]*cref[i];}
        fprintf(stderr, "[randmat] %-34s M=%d K=%d N=%d : cos=%.5f |npu|/|ref|=%.5f maxabserr=%.4f (worst m=%d n=%d npu=%.4f ref=%.4f, peak=%.3f)\n",
                p.name, M, K, N, c, std::sqrt(nn/nr), me, wr, wn, bf16f(C[(size_t)wr*N+wn]), cref[(size_t)wr*N+wn], wa);
        fflush(stderr);
    }

    /* ── FFN BO-to-BO GEMM probe (Step 2): the FFN path now runs the new
     * k_tile=128 b_col_maj kernels BO-to-BO (B in a device BO, NO host
     * readback).  Probe them directly: random A,B -> device BOs -> kernel ->
     * read C -> float32 ref.  cos<0.999 = layout bug; |npu|/|ref|!=1 = mag
     * inflation.  gate/up (M=16,K=4096,N=2048), down (M=16,K=2048,N=4096). */
    {
        struct FProbe { const char* name; const char* kname; int K; int N; };
        FProbe fprobes[] = {
            {"ffn_gate/up (BO-to-BO k128)", "gemm",      4096, 2048},
            {"ffn_down    (BO-to-BO k128)", "gemm_down", 2048, 4096},
        };
        for (const auto& p : fprobes) {
            const int K = p.K, N = p.N;
            std::vector<bf16_t, AlignedAllocator<bf16_t>> A((size_t)M*K), B((size_t)N*K), C((size_t)M*N, 0);
            for (auto& v : A) v = rbf();
            for (auto& v : B) v = rbf();
            xrt::bo bo_A = xrt::ext::bo(npu_device_, (size_t)M*K*sizeof(bf16_t));
            xrt::bo bo_B = xrt::ext::bo(npu_device_, (size_t)N*K*sizeof(bf16_t));
            xrt::bo bo_C = xrt::ext::bo(npu_device_, (size_t)M*N*sizeof(bf16_t));
            std::memcpy(bo_A.map<char*>(), A.data(), (size_t)M*K*sizeof(bf16_t)); npu_sync_to(bo_A);
            std::memcpy(bo_B.map<char*>(), B.data(), (size_t)N*K*sizeof(bf16_t)); npu_sync_to(bo_B);
            npu_sync_to(bo_C);
            auto& krnl = kernel_cache_->get(p.kname);
            auto run = krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_A), static_cast<xrt::bo&>(bo_B),
                static_cast<xrt::bo&>(bo_C),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
            flush_pending_runs();
            npu_sync_from(bo_C);
            std::memcpy(C.data(), bo_C.map<char*>(), (size_t)M*N*sizeof(bf16_t));
            std::vector<float> cn((size_t)M*N), cref((size_t)M*N);
            double me = 0;
            for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
                double a = 0;
                for (int k = 0; k < K; k++) a += (double)bf16f(A[(size_t)m*K+k]) * (double)bf16f(B[(size_t)n*K+k]);
                float ref = (float)a, npu = bf16f(C[(size_t)m*N+n]);
                cn[(size_t)m*N+n] = npu; cref[(size_t)m*N+n] = ref;
                float e = std::fabs(npu - ref); if (e > me) me = e;
            }
            double c = cos(cn, cref);
            double nn = 0, nr = 0;
            for (size_t i = 0; i < cn.size(); i++) { nn += (double)cn[i]*cn[i]; nr += (double)cref[i]*cref[i]; }
            fprintf(stderr, "[randmat] %-34s M=%d K=%d N=%d : cos=%.5f |npu|/|ref|=%.5f maxabserr=%.4f\n",
                    p.name, M, K, N, c, std::sqrt(nn/nr), me);
            fflush(stderr);
        }
    }
}
/* ── HC 8-row matvec on the vectorized qc kernel (NO new hw_context) ──────
 * Reuses the qc xclbin (M_c=16,K_c=1024,N_c=1024,b_col_maj) for the Hybrid
 * Connection control matvec (M=8, K=HC_DIM=16384, N=24 or 4).  K is tiled into
 * 1024-deep chunks (qc reduces 1024 internally) and partials accumulated in
 * float.  B is [K,N] row-major (hc_fn / hc_head_fn layout); each K-tile is
 * transpose-copied into the [N_c,K_c] host_only BO (bcol wants B[n,k]).  M=8
 * is zero-padded to 16 (one M-tile; padded rows contribute nothing).  ZERO CPU
 * GEMM, ZERO new context (stays at 8/8 — the AMDXDNA safe cap). */
void FSTEngine::npu_gemm_hc(bf16_t* C, const bf16_t* A, const bf16_t* B,
                              int M, int N, int K) {
    constexpr int M_c = 16, K_c = 1024, N_c = 1024;
    auto& krnl = kernel_cache_->get("qc");
    const size_t a_tile_b = (size_t)M_c * K_c * sizeof(bf16_t);
    const size_t b_tile_b = (size_t)N_c * K_c * sizeof(bf16_t);
    const size_t c_tile_b = (size_t)M_c * N_c * sizeof(bf16_t);
    if (bo_mla_atile_.size() < a_tile_b) bo_mla_atile_ = xrt::ext::bo(npu_device_, a_tile_b);
    if (bo_mla_ctile_.size() < c_tile_b) bo_mla_ctile_ = xrt::ext::bo(npu_device_, c_tile_b);
    if (bo_mla_bhost_.size() < b_tile_b)
        bo_mla_bhost_ = xrt::bo(npu_device_, b_tile_b, xrt::bo::flags::host_only, grp_);

    std::vector<float, AlignedAllocator<float>> acc((size_t)M * N, 0.0f);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> a_tile(M_c * K_c, 0);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> c_tile(M_c * N_c);
    for (int kt = 0; kt < K; kt += K_c) {
        const int k_cur = std::min(K_c, K - kt);
        memset(a_tile.data(), 0, a_tile_b);
        for (int m = 0; m < M; m++)
            memcpy(a_tile.data() + (size_t)m * K_c,
                   A + (size_t)m * K + kt,
                   (size_t)k_cur * sizeof(bf16_t));
        memcpy(bo_mla_atile_.map<char*>(), a_tile.data(), a_tile_b);
        npu_sync_to(bo_mla_atile_);
        /* B [K,N] -> bhost [N_c,K_c] (bcol): bhost[n*K_c+k] = B[(kt+k)*N+n]. */
        {auto* b = bo_mla_bhost_.map<bf16_t*>(); memset(b, 0, b_tile_b);
         for (int k = 0; k < k_cur; k++)
             for (int n = 0; n < N; n++)
                 b[(size_t)n * K_c + k] = B[(size_t)(kt + k) * N + n];}
        bo_mla_bhost_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        {auto* c = bo_mla_ctile_.map<bf16_t*>(); memset(c, 0, c_tile_b);}
        npu_sync_to(bo_mla_ctile_);
        auto run = krnl(3,0,0, static_cast<xrt::bo&>(bo_mla_atile_),
                        static_cast<xrt::bo&>(bo_mla_bhost_),
                        static_cast<xrt::bo&>(bo_mla_ctile_),
                        static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
        flush_pending_runs();
        npu_sync_from(bo_mla_ctile_);
        memcpy(c_tile.data(), bo_mla_ctile_.map<char*>(), c_tile_b);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                acc[(size_t)m * N + n] += bf16f(c_tile[(size_t)m * N_c + n]);
    }
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = f2bf(acc[i]);
}

// Unified EW kernel compile-time sizes (must match gen_ew_unified.py)
static constexpr int EW_SILU_N      = 32768;   // Mx(16) * INTER_DIM(2048)  [M=16 std]
static constexpr int EW_SOFTMAX_N    = 1024;    // S_padded
static constexpr int EW_ROPE_ROWS   = 8;
static constexpr int EW_ROPE_COLS    = 512;
static constexpr int EW_RMSNORM_M    = 8;
static constexpr int EW_RMSNORM_N    = 4096;    // hidden dim

void FSTEngine::npu_elementwise(const char* kname, const bf16_t* in, bf16_t* out, int n) {
    int kernel_n = 0;
    if (strcmp(kname, "silu") == 0) kernel_n = EW_SILU_N;
    else if (strcmp(kname, "softmax") == 0) kernel_n = EW_SOFTMAX_N;
    else { fprintf(stderr, "[ew] unknown unary op %s\n", kname); return; }

    const size_t bytes = (size_t)kernel_n * sizeof(bf16_t);
    if (bo_scratch_in_.size() < bytes)  bo_scratch_in_  = xrt::ext::bo(npu_device_, bytes);
    if (bo_scratch_out_.size() < bytes) bo_scratch_out_ = xrt::ext::bo(npu_device_, bytes);

    auto* p_in  = bo_scratch_in_.map<bf16_t*>();
    auto* p_out = bo_scratch_out_.map<bf16_t*>();
    memset(p_in, 0, bytes);
    memset(p_out, 0, bytes);
    int copy_n = std::min(n, kernel_n);
    memcpy(p_in, in, (size_t)copy_n * sizeof(bf16_t));
    npu_sync_to(bo_scratch_in_);
    npu_sync_to(bo_scratch_out_);

    auto& krnl = kernel_cache_->get(kname);
    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_scratch_out_));
    pending_runs_.push_back(std::move(run));
    flush_pending_runs();

    npu_sync_from(bo_scratch_out_);
    memcpy(out, p_out, (size_t)n * sizeof(bf16_t));
}

void FSTEngine::npu_elementwise_bin(const char* kname,
                                     const bf16_t* in_a, const bf16_t* in_b,
                                     bf16_t* out, int n) {
    if (strcmp(kname, "mul") != 0) {
        fprintf(stderr, "[ew] unknown binary op %s\n", kname);
        memcpy(out, in_a, (size_t)n * sizeof(bf16_t));
        return;
    }
    const int kernel_n = EW_SILU_N;  // mul shares the 65536-element tile
    const size_t bytes = (size_t)kernel_n * sizeof(bf16_t);
    if (bo_scratch_in_.size() < bytes)  bo_scratch_in_  = xrt::ext::bo(npu_device_, bytes);
    if (bo_scratch_out_.size() < bytes) bo_scratch_out_ = xrt::ext::bo(npu_device_, bytes);
    if (bo_scratch_big_.size() < bytes) bo_scratch_big_ = xrt::ext::bo(npu_device_, bytes);

    auto* p_a = bo_scratch_in_.map<bf16_t*>();
    auto* p_b = bo_scratch_big_.map<bf16_t*>();
    auto* p_c = bo_scratch_out_.map<bf16_t*>();
    memset(p_a, 0, bytes); memset(p_b, 0, bytes); memset(p_c, 0, bytes);
    int copy_n = std::min(n, kernel_n);
    memcpy(p_a, in_a, (size_t)copy_n * sizeof(bf16_t));
    memcpy(p_b, in_b, (size_t)copy_n * sizeof(bf16_t));
    npu_sync_to(bo_scratch_in_);
    npu_sync_to(bo_scratch_big_);
    npu_sync_to(bo_scratch_out_);

    auto& krnl = kernel_cache_->get("mul");
    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_scratch_big_),
        static_cast<xrt::bo&>(bo_scratch_out_),
        static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
    pending_runs_.push_back(std::move(run));
    flush_pending_runs();

    npu_sync_from(bo_scratch_out_);
    memcpy(out, p_c, (size_t)n * sizeof(bf16_t));
}

void FSTEngine::npu_rope(const char* kname, const bf16_t* in, const bf16_t* lut,
                          bf16_t* out, int rows, int cols) {
    (void)kname;
    // Unified EW rope kernel is compiled for [8, 512].  Chunk over M in 8-row slabs
    // so prefill M>8 (e.g. M=14) gets RoPE applied to ALL rows, not just the first 8.
    // (Prior to this, rows 8..M-1 — including the last prefill position whose logits
    // are sampled — were silently left un-RoPE'd, corrupting positional encoding.)
    const int SLAB = EW_ROPE_ROWS;                       // 8
    int c_copy = std::min(cols, EW_ROPE_COLS);
    const size_t slab_bytes = (size_t)SLAB * EW_ROPE_COLS * sizeof(bf16_t);
    if (bo_scratch_in_.size() < slab_bytes)  bo_scratch_in_  = xrt::ext::bo(npu_device_, slab_bytes);
    if (bo_scratch_out_.size() < slab_bytes) bo_scratch_out_ = xrt::ext::bo(npu_device_, slab_bytes);
    if (bo_scratch_big_.size() < slab_bytes) bo_scratch_big_ = xrt::ext::bo(npu_device_, slab_bytes);

    auto* p_a = bo_scratch_in_.map<bf16_t*>();
    auto* p_l = bo_scratch_big_.map<bf16_t*>();
    auto* p_o = bo_scratch_out_.map<bf16_t*>();
    auto& krnl = kernel_cache_->get("rope");

    for (int m0 = 0; m0 < rows; m0 += SLAB) {
        int mc = std::min(SLAB, rows - m0);              // real rows in this slab
        memset(p_a, 0, slab_bytes); memset(p_l, 0, slab_bytes); memset(p_o, 0, slab_bytes);
        for (int r = 0; r < mc; r++) {
            memcpy(p_a + r * EW_ROPE_COLS, in  + (size_t)(m0 + r) * cols, (size_t)c_copy * sizeof(bf16_t));
            memcpy(p_l + r * EW_ROPE_COLS, lut + (size_t)(m0 + r) * cols, (size_t)c_copy * sizeof(bf16_t));
        }
        npu_sync_to(bo_scratch_in_);
        npu_sync_to(bo_scratch_big_);
        npu_sync_to(bo_scratch_out_);

        auto run = krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_scratch_big_),
            static_cast<xrt::bo&>(bo_scratch_out_),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
        flush_pending_runs();

        npu_sync_from(bo_scratch_out_);
        for (int r = 0; r < mc; r++)
            memcpy(out + (size_t)(m0 + r) * cols, p_o + r * EW_ROPE_COLS, (size_t)c_copy * sizeof(bf16_t));
    }
}

void FSTEngine::npu_softmax_row(bf16_t* scores, int S_padded) {
    // NPU softmax kernel is compiled for 1024 elements.  Pad/zero-fill.
    const size_t bytes = (size_t)EW_SOFTMAX_N * sizeof(bf16_t);
    if (bo_scratch_in_.size() < bytes)  bo_scratch_in_  = xrt::ext::bo(npu_device_, bytes);
    if (bo_scratch_out_.size() < bytes) bo_scratch_out_ = xrt::ext::bo(npu_device_, bytes);

    auto* p_in  = bo_scratch_in_.map<bf16_t*>();
    auto* p_out = bo_scratch_out_.map<bf16_t*>();
    memset(p_in, 0, bytes); memset(p_out, 0, bytes);
    int copy_n = std::min(S_padded, EW_SOFTMAX_N);
    memcpy(p_in, scores, (size_t)copy_n * sizeof(bf16_t));
    npu_sync_to(bo_scratch_in_);
    npu_sync_to(bo_scratch_out_);

    auto& krnl = kernel_cache_->get("softmax");
    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_scratch_out_));
    pending_runs_.push_back(std::move(run));
    flush_pending_runs();

    npu_sync_from(bo_scratch_out_);
    memcpy(scores, p_out, (size_t)S_padded * sizeof(bf16_t));
}

void FSTEngine::npu_rmsnorm_weighted(bf16_t* out, const bf16_t* in,
                                        const bf16_t* weight, int d) {
    /* NPU rmsnorm kernel is compiled for [8, 4096].
     * For d == 4096 (attn/moe/final norms) we run it directly.
     * For d < 4096 (q_norm 1024, kv_norm 512) we zero-pad x and one-pad w
     * to 4096, run the kernel, then scale the output by sqrt(4096/d) to
     * compensate the RMS denominator change.  eps=1e-6 is small enough
     * that the compensation is exact to BF16 precision. */
    const int NPU_N = EW_RMSNORM_N;  // 4096
    const int NPU_M = EW_RMSNORM_M;  // 8
    const size_t row_bytes = (size_t)NPU_N * sizeof(bf16_t);
    const size_t bytes = (size_t)NPU_M * NPU_N * sizeof(bf16_t);
    if (bo_scratch_in_.size() < bytes)  bo_scratch_in_  = xrt::ext::bo(npu_device_, bytes);
    if (bo_scratch_out_.size() < bytes) bo_scratch_out_ = xrt::ext::bo(npu_device_, bytes);
    if (bo_scratch_big_.size() < bytes) bo_scratch_big_ = xrt::ext::bo(npu_device_, bytes);

    auto* p_a = bo_scratch_in_.map<bf16_t*>();
    auto* p_w = bo_scratch_big_.map<bf16_t*>();
    auto* p_o = bo_scratch_out_.map<bf16_t*>();
    memset(p_a, 0, bytes); memset(p_w, 0, bytes); memset(p_o, 0, bytes);
    memcpy(p_a, in, (size_t)d * sizeof(bf16_t));
    for (int r = 0; r < NPU_M; r++) {
        memcpy(p_w + r * NPU_N, weight, (size_t)d * sizeof(bf16_t));
        if (d < NPU_N) {
            for (int i = d; i < NPU_N; i++)
                p_w[r * NPU_N + i] = f2bf(1.0f);  // one-pad weight
        }
    }
    npu_sync_to(bo_scratch_in_);
    npu_sync_to(bo_scratch_big_);
    npu_sync_to(bo_scratch_out_);

    auto& krnl = kernel_cache_->get("rmsnorm");
    auto run = krnl(3, 0, 0,
        static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_scratch_big_),
        static_cast<xrt::bo&>(bo_scratch_out_),
        static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
    pending_runs_.push_back(std::move(run));
    flush_pending_runs();

    npu_sync_from(bo_scratch_out_);
    if (d == NPU_N) {
        memcpy(out, p_o, (size_t)d * sizeof(bf16_t));
    } else {
        /* Zero-padding x from d to NPU_N shrinks the kernel's RMS denominator
         * by sqrt(d/NPU_N), so the kernel output is INFLATED by sqrt(NPU_N/d).
         * Compensate by scaling back down: multiply by sqrt(d/NPU_N).  (The
         * previous sqrt(NPU_N/d) doubled the inflation -> 8x for d=512.) */
        const float scale = sqrtf((float)d / (float)NPU_N);
        for (int i = 0; i < d; i++)
            out[i] = f2bf(bf16f(p_o[i]) * scale);
    }
}

void FSTEngine::npu_router(int* eids, float* wts, const bf16_t* hidden,
                             const float* router_weights, const float* router_bias,
                             int M, int n_experts, int top_k, int hd,
                             int lid, const int* input_ids) {
    const int K = hd, N = n_experts, TOP_K = top_k;
    const int M_PAD = ((M + 15) / 16) * 16;   // M=16 standardization

    // Router weights [K, N] -> bf16
    std::vector<bf16_t, AlignedAllocator<bf16_t>> rw_bf16(K * N);
    for (int i = 0; i < K * N; i++) rw_bf16[i] = f2bf(router_weights[i]);

    // Pad hidden to M_PAD rows
    std::vector<bf16_t, AlignedAllocator<bf16_t>> h_pad(M_PAD * K, 0);
    memcpy(h_pad.data(), hidden, (size_t)M * K * sizeof(bf16_t));

    // NPU router GEMM: h_pad[M_PAD,K] @ rw[K,N] -> logits[M_PAD,N]
    std::vector<bf16_t, AlignedAllocator<bf16_t>> logits_bf16(M_PAD * N);
    /* The router_gemm kernel is specialized M=16 (gen_ew_unified.py:487,
     * TILE_M=16).  Tile along M in 16-row chunks so every row gets correct
     * logits.  M_PAD is always a multiple of 16 (M_PAD=((M+15)/16)*16), so
     * each chunk is exactly 16 — matching the kernel's specialization. */
    for (int m_off = 0; m_off < M_PAD; m_off += 16) {
        npu_gemm("router_gemm", logits_bf16.data() + (size_t)m_off * N,
                 h_pad.data() + (size_t)m_off * K, rw_bf16.data(), 16, N, K);
    }

    /* Top-K + sqrtsoftplus weight on host (selection logic, not GEMM).
     * Reference model.py Gate.forward (569-588):
     *   scores      = sqrt(softplus(raw_logit))     # sqrt(log(1+exp(x)))
     *   original    = scores                        # UNBIASED sqrtsoftplus
     *   scores     += bias                          # bias shifts topk ONLY
     *   indices     = scores.topk(topk)
     *   weights     = original.gather(indices)      # UNBIASED sqrtsoftplus
     *   weights    /= weights.sum()                 # normalize to sum 1
     *   weights    *= route_scale
     * The prior engine code used expf(raw_logit) (a full softmax numerator),
     * which over-concentrates on one expert, AND ignored the bias, selecting
     * the wrong experts.  Both are fixed here. */
    std::vector<float, AlignedAllocator<float>> logit_f((size_t)M * N);
    std::vector<float, AlignedAllocator<float>> score_f((size_t)M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float raw = bf16f(logits_bf16[m * N + n]);
            logit_f[m * N + n] = raw;
            /* sqrtsoftplus = sqrt(log(1+exp(x))); numerically stable for large x. */
            float sp = (raw > 20.0f) ? (raw + 0.0f) : logf(1.0f + expf(raw));
            score_f[m * N + n] = sqrtf(sp);
        }

    /* CPU-ref router logits vs NPU (M>1 audit).  The router_gemm kernel is
     * specialized M=8 (gen_ew_unified.py:487, TILE_M=8, range_(M//m*N//n) with
     * M=8 -> 1 M-tile), so for M_PAD>8 only rows 0..7 are computed; rows
     * 8..M_PAD-1 stay zero (memset).  Confirm by comparing NPU logits to a host
     * float GEMM for m=0,4,8,12.  Read-only diagnostic (pure C++ math). */
    if ((lid == 0 || lid == 26) && getenv("FST_ROUTER_AUDIT")) {
        const int audit_m[4] = {0, 4, 8, 12};
        for (int mi = 0; mi < 4; mi++) {
            int m = audit_m[mi];
            if (m >= M) continue;
            float worst = -1.0f; int wn = 0;
            for (int n = 0; n < N; n++) {
                double acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (double)bf16f(h_pad[(size_t)m * K + k]) *
                           (double)bf16f(rw_bf16[(size_t)k * N + n]);
                float npu = bf16f(logits_bf16[(size_t)m * N + n]);
                float d = std::fabs((float)acc - npu);
                if (d > worst) { worst = d; wn = n; }
            }
            double acc = 0;
            for (int k = 0; k < K; k++)
                acc += (double)bf16f(h_pad[(size_t)m * K + k]) *
                       (double)bf16f(rw_bf16[(size_t)k * N + wn]);
            fprintf(stderr, "[router-cpu] L%d m=%d: worst n=%d cpu=%.4f npu=%.4f (M_PAD=%d)\n",
                    lid, m, wn, (float)acc, bf16f(logits_bf16[(size_t)m * N + wn]), M_PAD);
            fflush(stderr);
        }
    }

    /* Hash routing (model.py 561-582): for lid < n_hash_layers, expert INDICES
     * come from the per-token-id table tid2eid[input_id] (no topk); the WEIGHTS
     * are still the UNBIASED sqrtsoftplus of the logits at those indices,
     * normalized to sum 1 (no bias — hash layers have none).  Falls back to
     * score-based if the sidecar isn't loaded or input_ids is null. */
    const bool hash_layer = tid2eid_loaded_ && lid < config_.n_hash_layers && input_ids != nullptr;

    for (int m = 0; m < M; m++) {
        int idx[64];  /* TOP_K <= 6 */
        if (hash_layer) {
            const int tid = input_ids[m];
            const int* row = tid2eid_.data() + ((size_t)lid * config_.vocab_size + tid) * TOP_K;
            for (int k = 0; k < TOP_K; k++) idx[k] = row[k];
        } else {
            std::vector<int> ord(N);
            for (int i = 0; i < N; i++) ord[i] = i;
            /* topk by (sqrtsoftplus + bias) — bias shifts selection only. */
            std::partial_sort(ord.begin(), ord.begin() + TOP_K, ord.end(),
                [&](int a, int b) {
                    float sa = score_f[m * N + a] + (router_bias ? router_bias[a] : 0.0f);
                    float sb = score_f[m * N + b] + (router_bias ? router_bias[b] : 0.0f);
                    return sa > sb;
                });
            for (int k = 0; k < TOP_K; k++) idx[k] = ord[k];
        }

        /* weights = UNBIASED sqrtsoftplus of selected experts, normalized. */
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) {
            float v = score_f[m * N + idx[k]];
            wts[m * TOP_K + k] = v;
            sum += v;
        }
        float inv = 1.0f / (sum + 1e-12f);
        /* model.py 588: weights /= sum; weights *= route_scale (1.5).
         * FaStar previously stopped at normalize-to-sum-1 (inv only), omitting
         * the *route_scale → routed MoE under-driven 0.667× vs HF.  Applying it
         * makes the routed FFN output LARGER (toward HF), which increases the
         * residual — the opposite of hiding an explosion.  Verified intrinsic:
         * the HF ground-truth reference reproduces the BOS explosion (L42 68813
         * vs FaStar's 38144) precisely because HF uses route_scale=1.5. */
        float rs = config_.route_scale;
        for (int k = 0; k < TOP_K; k++) {
            eids[m * TOP_K + k] = idx[k];
            wts[m * TOP_K + k] = wts[m * TOP_K + k] * inv * rs;
        }
    }

    /* L0 hash-routing audit (Step 1): print input_ids, hash flag, and the
     * selected experts+weights per position to verify tid2eid[input_ids[m]]
     * is indexed correctly for M>1.  Pure C++, read-only. */
    if (lid == 0 && std::getenv("FST_ROUTER_AUDIT")) {
        for (int mi = 0; mi < M; mi++) {
            fprintf(stderr, "[rtr] L0 m=%d tid=%d hash=%d eids=",
                    mi, input_ids ? input_ids[mi] : -1, (int)hash_layer);
            for (int k = 0; k < TOP_K; k++)
                fprintf(stderr, "%d(%.3f) ", eids[mi * TOP_K + k], wts[mi * TOP_K + k]);
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }
}

int FSTEngine::npu_sample_token(const bf16_t* hidden, int M, float temperature, float top_p) {
    const int D = config_.hidden_dim, V = config_.vocab_size;
    std::vector<float> logits(V);
    /* hidden is the 4-stream HC residual [M, N_HC, D]; collapse to plain [M, D]. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> plain((size_t)M * D);
    output_hc_head(M, hidden, plain.data());
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hs(D);
    memcpy(hs.data(), plain.data(), D * sizeof(bf16_t));
    npu_rmsnorm_weighted(hs.data(), hs.data(), model_.final_norm, D);

    const int M_PAD = 16;   // M=16 standardization (lm_head_gemm kernel M=16)
    const int N_PAD = lm_head_n_pad_;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hs_pad(M_PAD * D, 0);
    memcpy(hs_pad.data(), hs.data(), D * sizeof(bf16_t));

    /* NPU LM head GEMM, tiled along N by npu_gemm (64 x 2048-col tiles).
     * bo_lm_head_ is already [D, N_PAD] row-major (transposed + zero-padded). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> bf16_out(M_PAD * N_PAD);
    bool lm_time = std::getenv("FST_LM_HEAD_TIME");
    double lm_t0 = lm_time ? now() : 0.0;
    npu_gemm("lm_head_gemm", bf16_out.data(), hs_pad.data(),
             bo_lm_head_.map<bf16_t*>(), M_PAD, N_PAD, D);
    if (lm_time)
        fprintf(stderr, "[lm-head-time] M=%d N_PAD=%d D=%d  lm_head=%.3f ms\n",
                M, N_PAD, D, (now() - lm_t0) * 1000.0);

    for (int v = 0; v < V; v++) logits[v] = bf16f(bf16_out[v]);

    /* First-decode top-5 logit diagnostic (env-gated, zero production cost).
     * call 0 = prefill last-position logits; call 1 = FIRST DECODE step. */
    if (const char* e = std::getenv("FST_DECODE_LOGITS")) {
        static int s_call = 0;
        int call = s_call++;
        std::vector<std::pair<float, int>> vk;
        vk.reserve(V);
        for (int v = 0; v < V; v++) vk.emplace_back(logits[v], v);
        int k = std::min(5, V);
        std::partial_sort(vk.begin(), vk.begin() + k, vk.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        fprintf(stderr, "[decode-logits] call=%d (%s) temp=%.3f top5:", call,
                call == 0 ? "prefill" : (call == 1 ? "FIRST-DECODE" : "decode"),
                temperature);
        for (int i = 0; i < k; i++)
            fprintf(stderr, " %d(%.5f)", vk[i].second, vk[i].first);
        fprintf(stderr, "\n");
        fflush(stderr);
        /* On the prefill step, dump full logit vector + last-pos pre/post-norm
         * hidden to disk for cosine comparison against the HF reference. */
        if (call == 0) {
            FILE* f = std::fopen("/tmp/eng_prefill_logits.f32", "wb");
            if (f) { std::fwrite(logits.data(), sizeof(float), V, f); std::fclose(f); }
            std::vector<float> pf(D), hf(D);
            for (int i = 0; i < D; i++) {
                pf[i] = bf16f(plain[i]);   // plain[0:D] = sampled (last) position pre-norm
                hf[i] = bf16f(hs[i]);      // post-final-norm
            }
            f = std::fopen("/tmp/eng_prefill_plain.f32", "wb");
            if (f) { std::fwrite(pf.data(), sizeof(float), D, f); std::fclose(f); }
            f = std::fopen("/tmp/eng_prefill_hs.f32", "wb");
            if (f) { std::fwrite(hf.data(), sizeof(float), D, f); std::fclose(f); }
            fprintf(stderr, "[decode-logits] dumped call=0: logits[V=%d] + plain[D=%d] + hs[D=%d] -> /tmp/eng_prefill_*.f32\n", V, D, D);
            fflush(stderr);
        }
    }

    return sample_token(logits.data(), V, temperature, top_p);
}

// ── Weight & KV cache upload ──────────────────────────────────────────────

/* Preload ALL layers' weights into persistent host_only BOs at startup.
 * 43 layers * ~133 MB = ~5.7 GB, fits in 64 GB RAM alongside the 21 GB base.
 * During inference process_layer(lid) just indexes into shared_[lid].bo_*
 * — zero per-layer BO allocation, zero sync. */
void FSTEngine::preload_all_layers() {
    auto make_bo = [&](const void* data, size_t bytes) -> xrt::bo {
        if (bytes == 0) return {};
        xrt::bo bo(npu_device_, bytes, xrt::bo::flags::host_only, grp_);
        memcpy(bo.map<char*>(), data, bytes);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return bo;
    };
    /* Padded BO: allocate [pad_n, k] but only the first [real_n, k] rows hold
     * real data; the rest are zero.  Used for wkv (real N=512, kernel N_c=1024)
     * so the persistent sub-buffer passed to the qck kernel already carries the
     * N-pad the host strided-copy path used to insert per-dispatch. */
    auto make_bo_padded = [&](const void* data, int real_n, int pad_n, int k) -> xrt::bo {
        const size_t row_b = (size_t)k * sizeof(bf16_t);
        xrt::bo bo(npu_device_, (size_t)pad_n * row_b, xrt::bo::flags::host_only, grp_);
        char* p = bo.map<char*>();
        memcpy(p, data, (size_t)real_n * row_b);
        memset(p + (size_t)real_n * row_b, 0, (size_t)(pad_n - real_n) * row_b);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return bo;
    };

    size_t total_bytes = 0;
    for (int l = 0; l < config_.n_layers; l++) {
        auto& w = shared_[l];
        w.bo_wq_a       = make_bo(w.wq_a.data(),       w.wq_a.size()       * sizeof(bf16_t));
        w.bo_wq_b       = make_bo(w.wq_b.data(),       w.wq_b.size()       * sizeof(bf16_t));
        /* wkv: [512,4096] -> padded [1024,4096] (qck kernel N_c=1024). */
        w.bo_wkv        = make_bo_padded(w.wkv.data(), MLA_KV_LORA, 1024, config_.hidden_dim);
        w.bo_wo_a       = make_bo(w.wo_a.data(),       w.wo_a.size()       * sizeof(bf16_t));
        w.bo_wo_b       = make_bo(w.wo_b.data(),       w.wo_b.size()       * sizeof(bf16_t));
        w.bo_shared_gate= make_bo(w.shared_gate.data(),w.shared_gate.size()* sizeof(bf16_t));
        w.bo_shared_up  = make_bo(w.shared_up.data(),  w.shared_up.size()  * sizeof(bf16_t));
        w.bo_shared_down= make_bo(w.shared_down.data(),w.shared_down.size()* sizeof(bf16_t));
        w.bo_w_k_decompress = make_bo(w.w_k_decompress.data(), w.w_k_decompress.size() * sizeof(bf16_t));
        w.bo_w_v_decompress = make_bo(w.w_v_decompress.data(), w.w_v_decompress.size() * sizeof(bf16_t));
        w.bo_w_k_pe     = make_bo(w.w_k_pe.data(),    w.w_k_pe.size()    * sizeof(bf16_t));
        total_bytes += (w.wq_a.size()+w.wq_b.size()+w.wkv.size()+w.wo_a.size()+w.wo_b.size()
                        +w.shared_gate.size()+w.shared_up.size()+w.shared_down.size()
                        +w.w_k_decompress.size()+w.w_v_decompress.size()+w.w_k_pe.size()) * sizeof(bf16_t);
        if ((l + 1) % 10 == 0 || l + 1 == config_.n_layers)
            fprintf(stderr, "[preload] %d/%d layers (%.2f GB)\n",
                    l + 1, config_.n_layers, (double)total_bytes / (1024.0*1024.0*1024.0));
    }
    current_loaded_layer_ = -2;  /* sentinel: all preloaded, no lazy path */
}

void FSTEngine::init_kv_cache_bo() {
    int grp = kernel_cache_->data_group_id();
    for (auto& kv : kv_cache_) {
        size_t kv_bytes = kv.kv_latent.size() * sizeof(bf16_t);
        size_t pe_bytes = kv.k_pe.size() * sizeof(bf16_t);
        kv.kv_latent_bo = xrt::bo(npu_device_, kv_bytes, xrt::bo::flags::host_only, grp);
        memcpy(kv.kv_latent_bo.map<char*>(), kv.kv_latent.data(), kv_bytes);
        kv.kv_latent_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_bytes, 0);
        kv.k_pe_bo = xrt::bo(npu_device_, pe_bytes, xrt::bo::flags::host_only, grp);
        memcpy(kv.k_pe_bo.map<char*>(), kv.k_pe.data(), pe_bytes);
        kv.k_pe_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, pe_bytes, 0);
    }
}

// ── Engine ────────────────────────────────────────────────────────────────

static bool npu_ok = false;

// L0 numerical self-check / audit blocks recompute CPU references and print
// ratios that are themselves buggy (wrong B layout assumptions) — they are
// noisy and misleading in production runs.  Gate them off unless FST_AUDIT=1.
static const bool audit_on = std::getenv("FST_AUDIT") != nullptr;

bool FSTEngine::validate_npu_kernels() {
    try {
        constexpr int M = 1, N = 16, K = 16;
        const size_t a_sz = (size_t)M * K * sizeof(bf16_t);
        const size_t b_sz = (size_t)K * N * sizeof(bf16_t);
        const size_t c_sz = (size_t)M * N * sizeof(bf16_t);

        auto& gk = kernel_cache_->get("gemm");

        xrt::ext::bo bo_a(npu_device_, a_sz);
        xrt::ext::bo bo_b(npu_device_, b_sz);
        xrt::ext::bo bo_c(npu_device_, c_sz);
        xrt::ext::bo d1(npu_device_, 4096), d2(npu_device_, 4096);

        memset(bo_a.map<char*>(), 0, a_sz); npu_sync_to(bo_a);
        memset(bo_b.map<char*>(), 0, b_sz); npu_sync_to(bo_b);
        memset(bo_c.map<char*>(), 0, c_sz); npu_sync_to(bo_c);
        npu_sync_to(d1); npu_sync_to(d2);

        auto run = gk(3, 0, 0,
            static_cast<xrt::bo&>(bo_a), static_cast<xrt::bo&>(bo_b), static_cast<xrt::bo&>(bo_c),
            static_cast<xrt::bo&>(d1), static_cast<xrt::bo&>(d2));

        ert_cmd_state state = run.wait();
        return state == ERT_CMD_STATE_COMPLETED;
    } catch (...) { return false; }
}

FSTEngine::FSTEngine(const std::string& fst_path, size_t cache_mb) {
    try { xrt::ini::set("host_mem_size", "1G"); } catch (...) {}

    /* ── Probe NPU device availability before XRT opens it.
     * If /dev/dri/renderD128 is held by a stale process, XRT will
     * either hang or produce garbage errors.  Fail fast instead. */
    {
        int probe_fd = ::open("/dev/dri/renderD128", O_RDWR);
        if (probe_fd < 0) {
            if (errno == EBUSY) {
                fprintf(stderr,
                    "FATAL: NPU device /dev/dri/renderD128 is busy. "
                    "Kill stale FaStar processes or reboot.\n");
                throw std::runtime_error("NPU device busy");
            }
            /* ENOENT or other: the device might be named differently
             * on this platform.  Let XRT handle it. */
        } else {
            ::close(probe_fd);
        }
    }

    npu_device_ = xrt::device(0);

    /* Resolve the kernel directory (FST_KERNEL_DIR / ./kernels / CWD) once and
     * route every _insts.bin / .xclbin load through it. */
    const std::string kdir = resolve_kernel_dir();
    fprintf(stderr, "[npu] kernel dir: %s\n", kdir.c_str());
    if (access(kpath("fst_expert_gemm_vec.xclbin").c_str(), F_OK) != 0 ||
        access(kpath("fst_expert_gemm_down.xclbin").c_str(), F_OK) != 0 ||
        access(kpath("fst_dequant_v4.xclbin").c_str(), F_OK) != 0)
        throw std::runtime_error("Missing required NPU xclbin files in '" + kdir + "'");

    kernel_cache_ = std::make_unique<AiebuKernelCache>(npu_device_,
                                                       kpath("fst_expert_gemm_vec.xclbin"));
    kernel_cache_->set_base_dir(kdir);

    {
        int efd = ::open(fst_path.c_str(), O_RDONLY);
        if (efd >= 0) {
            auto hb = pread(efd, 128, 0);
            auto* eh = (const FSTH*)hb.data();
            config_.hidden_dim = (int)eh->hd;
            config_.n_layers   = (int)eh->nl;
            config_.n_experts  = (int)eh->ne;
            config_.top_k      = (int)eh->tk;
            config_.expert_block_bytes = (size_t)eh->eb;
            /* Detect Hunyuan-3.0 (HY3): GQA 64 Q / 8 KV @ head_dim 128 — the
             * opposite attention family from DS4's MLA.  Done here (before
             * kernel registration) so the MLA xclbins can be skipped on the
             * HY3 path, freeing hw_contexts.  HY3 constants are unpacked from
             * the reserved r1/r2 slots (no header struct change):
             *   r1 low16  = first_k_dense_replace, r1 high16 = expert_gating_func
             *   r2 high32 = int(yarn_factor*1000),  r2 low32  = int(ew_scale*1e6) */
            if (eh->qh == 64 && eh->kh == 8 && eh->hd2 == 128) {
                config_.arch = ARCH_HY3;
                config_.num_q_heads  = (int)eh->qh;        /* 64 */
                config_.num_kv_heads = (int)eh->kh;        /* 8  */
                config_.head_dim    = (int)eh->hd2;        /* 128 */
                config_.expert_inter_dim = (int)eh->id;   /* 1536 */
                config_.rope_dim    = config_.head_dim;   /* rotate the full head_dim (rotate-half) */
                config_.first_k_dense_replace = (int)(eh->r1 & 0xFFFFu);         /* 1 */
                config_.expert_gating_func    = (int)((eh->r1 >> 16) & 0xFFFFu); /* 2 = sigmoid */
                config_.rope_scaling_factor =
                    (float)((eh->r2 >> 32) & 0xFFFFFFFFu) / 1000.0f;             /* 4.0 YaRN */
                config_.expert_weights_scale =
                    (float)(eh->r2 & 0xFFFFFFFFu) / 1e6f;                      /* 2.826 */
                config_.yarn_orig_ctx = 262144;
                fprintf(stderr, "[arch] HY3 (Hunyuan-3.0): %d layers, GQA %d/%d @ %d, "
                        "%d experts top-%d, dense L0..%d, sigmoid router, ew_scale=%.4f, "
                        "yarn=%.2f\n", config_.n_layers, config_.num_q_heads,
                        config_.num_kv_heads, config_.head_dim, config_.n_experts,
                        config_.top_k, config_.first_k_dense_replace,
                        config_.expert_weights_scale, config_.rope_scaling_factor);
            }
            ::close(efd);
        }
    }

    // ── 5 permanent NPU contexts, zero LRU, zero CPU fallback ────────
    // FST_MC_FFN: register the batched multi-core GEMMs (vec_mc gate/up share one
    // xclbin like gemm/gemm2; down_mc is its own xclbin) instead of the single-core
    // gemm/gemm2/gemm_down.  Same context count (2 FFN-GEMM ctxs).  silu_b/mul_b
    // ride the existing ew_unified xclbin (+0 ctx).  Default keeps the 0.06 baseline.
    const bool mc_ffn = std::getenv("FST_MC_FFN") != nullptr;
    /* Expert BO-cache cap: DS4 fits all 1376 experts in 20 GB; HY3 has 15,360
     * experts (153 GB) and runs under a 35 GB cgroup, so default to 8 GB
     * (≈800 experts, an LRU window — misses re-load from SSD).  Env-overridable. */
    expert_bo_cap_ = (config_.arch == ARCH_HY3) ? (8ULL << 30) : (20ULL << 30);
    if (const char* e = std::getenv("FST_EXPERT_BO_CAP_GB"))
        expert_bo_cap_ = (size_t)std::atol(e) << 30;
    // HY3 dispatch-collapse (FST_HY3_FUSED_FFN): pack ≤8 experts into one 80 MB
    // MXFP4 BO, dequant all 8 in ONE dispatch, then 5 batched MC dispatches
    // (gate/up/silu/mul/down, 8 cores 1 expert/core) = 6 dispatches/layer vs 48.
    // Reuses the proven expert_gemm_mc_op (DS4 MC, cos 0.9988) at HY3 shapes.
    hy3_fused_ffn_ = (config_.arch == ARCH_HY3) && (std::getenv("FST_HY3_FUSED_FFN") != nullptr);
    if (config_.arch == ARCH_HY3) {
        if (hy3_fused_ffn_) {
            // 8-expert batched MC kernels (gate/up share hy3_gemm_vec_mc_8exp;
            // down is its own xclbin).  3 FFN ctxs + ew_unified = 4 total (< 9).
            kernel_cache_->register_kernel("hy3_gemm_mc",      "fst_hy3_gemm_vec_mc_8exp_insts.bin",  "fst_hy3_gemm_vec_mc_8exp.xclbin");
            kernel_cache_->register_kernel("hy3_gemm_down_mc", "fst_hy3_gemm_down_8exp_insts.bin",   "fst_hy3_gemm_down_8exp.xclbin");
        } else {
            // HY3 FFN: expert inter_dim=1536 (DS4 kernels are baked to inter=2048 and
            // CANNOT be reused — N=1536 vs 2048 for gate/up, K=1536 vs 2048 for down).
            // gate/up : A[M,4096] @ B[4096,1536] -> C[M,1536]   (m16k128n32 b_col_maj)
            // down    : A[M,1536] @ B[1536,4096] -> C[M,4096]   (m16k128n64 b_col_maj)
            kernel_cache_->register_kernel("hy3_gemm",      "fst_hy3_gemm_vec_insts.bin",  "fst_hy3_gemm_vec.xclbin");
            kernel_cache_->register_kernel("hy3_gemm2",      "fst_hy3_gemm_vec_insts.bin",  "fst_hy3_gemm_vec.xclbin");
            kernel_cache_->register_kernel("hy3_gemm_down", "fst_hy3_gemm_down_insts.bin", "fst_hy3_gemm_down.xclbin");
        }
    } else if (mc_ffn) {
        kernel_cache_->register_kernel("gemm_mc",     "fst_expert_gemm_vec_mc_insts.bin",  "fst_expert_gemm_vec_mc.xclbin");
        kernel_cache_->register_kernel("gemm_down_mc","fst_expert_gemm_down_mc_insts.bin","fst_expert_gemm_down_mc.xclbin");
    } else {
        // Context 1: expert GEMM (gate/up) — vectorized aie::mmul<4,8,8>
        kernel_cache_->register_kernel("gemm",       "fst_expert_gemm_vec_insts.bin",  "fst_expert_gemm_vec.xclbin");
        kernel_cache_->register_kernel("gemm2",      "fst_expert_gemm_vec_insts.bin",  "fst_expert_gemm_vec.xclbin");
        // Context 2: down GEMM — vectorized aie::mmul<4,8,8>
        kernel_cache_->register_kernel("gemm_down",  "fst_expert_gemm_down_insts.bin", "fst_expert_gemm_down.xclbin");
    }
    // Context 3-6: MLA vectorized GEMM kernels (canonical kernels.mm+zero, 100%
    // aie::mmul).  4 xclbins, 4 hw_contexts — total 4 (FFN/dequant/ew) + 4 MLA = 8
    // (the AMDXDNA hard limit).  The vectorized worker loop `range_(tiles)` is
    // CompileTime-baked, so distinct shapes need distinct xclbins (verified: qc
    // vs kvc ELFs differ).  qk/sv share one (2048,512,512) b_row_maj xclbin (N/K
    // zero-padded to 512, 4x waste on small K/N, 1 shared context); qc/oa/kvc
    // share the (32,4096,1024) b_col_maj xclbin (kvc N-pads 512->1024, 2x waste).
    // NO scalar kernel, NO CPU fallback — MLA is fully vectorized aie::mmul.
    //
    // HY3 (ARCH_HY3) is GQA — it reuses expert_gemm_vec + ew_unified for q/k/v/o
    // and DROPS the whole MLA context set (qc/wqb/ob/qksv/qck), so it stays at
    // ~4 hw_contexts and frees ≥3 of the 9-cap for a later fused-FFN xclbin.
    if (config_.arch == ARCH_DS4) {
    kernel_cache_->register_kernel("qc",   "fst_mla_qc_insts.bin",   "fst_mla_qc.xclbin");
    kernel_cache_->register_kernel("wqb",  "fst_mla_wqb_insts.bin",  "fst_mla_wqb.xclbin");
    kernel_cache_->register_kernel("ob",   "fst_mla_ob_insts.bin",   "fst_mla_ob.xclbin");
    kernel_cache_->register_kernel("qksv", "fst_mla_qksv_insts.bin", "fst_mla_qksv.xclbin");
    // Device-side K fix (Step 1): k_tile=128 -> K_div_k=32 (within NPU2 per-BD
    // <=64 safe zone), full K=4096 streamed on-device in ONE call (no host K-loop,
    // no readback).  Used by the MLA projections via npu_gemm_mla_vec("qck").
    // HC stays on the "qc" xclbin (npu_gemm_hc, K_c=1024) -- separate context.
    kernel_cache_->register_kernel("qck",  "fst_mla_qck_insts.bin",  "fst_mla_qck.xclbin");
    } /* ARCH_DS4 MLA kernels */
    // Context 4: dequant (MXFP4: 17 bytes/block → 32 BF16, emits B[N,K] row-major).
    // HY3 experts are 10,027,008 B (589,824 blocks = 3*196,608) vs DS4's
    // 13,369,344 B (786,432 blocks); the DS4 dequant is baked to 786,432 and would
    // OVERRUN an HY3 expert BO, so HY3 uses its own 589,824-block dequant.
    if (config_.arch == ARCH_HY3) {
        kernel_cache_->register_kernel("hy3_dequant", "fst_hy3_dequant_insts.bin", "fst_hy3_dequant.xclbin");
        if (hy3_fused_ffn_)
            // 8-expert packed dequant (variant, currently UNUSED — the per-slot
            // dequant in process_expert_ffn_hy3_fused avoids its 80 MB pack+sync,
            // which measured 4.5× slower than 8 pipelined per-expert dequants).
            // Kept registered for A/B comparison.
            kernel_cache_->register_kernel("hy3_dequant_8exp", "fst_hy3_dequant_8exp_insts.bin", "fst_hy3_dequant_8exp.xclbin");
    } else
        kernel_cache_->register_kernel("dequant",    "fst_dequant_v4_insts.bin",         "fst_dequant_v4.xclbin");
    // Context 5: ew_unified -- 7 kernels sharing ONE xclbin / ONE hw_context:
    //   rmsnorm, silu, mul, softmax, rope, router_gemm, lm_head_gemm
    kernel_cache_->register_kernel("rmsnorm",      "fst_ew_rmsnorm_insts.bin",      "fst_ew_unified.xclbin");
    kernel_cache_->register_kernel("silu",         "fst_ew_silu_insts.bin",         "fst_ew_unified.xclbin");
    kernel_cache_->register_kernel("mul",          "fst_ew_mul_insts.bin",          "fst_ew_unified.xclbin");
    // Batched silu/mul (FST_MC_FFN): N=E_MC*Mx*INTER_DIM=196608, same ew_unified
    // xclbin, +0 hw_context.  Unused in default mode (harmless extra insts entries).
    kernel_cache_->register_kernel("silu_b",       "fst_ew_silu_b_insts.bin",       "fst_ew_unified.xclbin");
    kernel_cache_->register_kernel("mul_b",        "fst_ew_mul_b_insts.bin",        "fst_ew_unified.xclbin");
    kernel_cache_->register_kernel("softmax",      "fst_ew_softmax_insts.bin",     "fst_ew_unified.xclbin");
    kernel_cache_->register_kernel("rope",         "fst_ew_rope_insts.bin",         "fst_ew_unified.xclbin");
    kernel_cache_->register_kernel("router_gemm",  "fst_ew_router_gemm_insts.bin",  "fst_ew_unified.xclbin");
    kernel_cache_->register_kernel("lm_head_gemm", "fst_ew_lm_head_gemm_insts.bin", "fst_ew_unified.xclbin");

    grp_ = kernel_cache_->data_group_id();
    current_loaded_layer_ = -1;
    const bool randmat = std::getenv("FST_RANDMAT_TEST");
    fprintf(stderr, "[dbg] RANDMAT_TEST env=[%s] randmat=%d grp_=%d\n",
            std::getenv("FST_RANDMAT_TEST") ? "SET" : "null", (int)randmat, grp_); fflush(stderr);
    if (randmat) {
        /* Random-matrix GEMM probe: needs only grp_ + 1MB dummy workspace BOs.
         * npu_gemm / npu_gemm_mla_vec allocate their own A/B/C BOs lazily.
         * Runs BEFORE the 150GB weight load and the empty-embedding print. */
        const size_t MB = 1024 * 1024;
        bo_d1_ = xrt::ext::bo(npu_device_, 1 * MB);
        bo_d2_ = xrt::ext::bo(npu_device_, 1 * MB);
        fprintf(stderr, "[randmat] starting probe (grp_=%d)\n", grp_); fflush(stderr);
        randmat_test();
        std::exit(0);
    }
    if (!randmat) {
    {double ts0=now(); load_shared_weights(fst_path);
     fprintf(stderr,"[time] load_shared_weights %.2fs\n", now()-ts0); fflush(stderr);}

    /* FST_HY3_LOAD_ONLY: stop right after the HY3 weight load (before any
     * NPU scratch-BO alloc / preload / generation) — used to verify the
     * arch detection + TID-based loader end-to-end without the (unwired yet)
     * GQA/MTP inference path.  Prints a per-tensor sanity summary. */
    if (config_.arch == ARCH_HY3 && std::getenv("FST_HY3_LOAD_ONLY")) {
        const auto& w0 = hy3_shared_[0];
        const auto& wl = hy3_shared_[config_.n_layers - 1];
        fprintf(stderr, "[hy3-verify] arch=HY3 layers=%d hidden=%d GQA %d/%d@%d "
                "experts=%d topk=%d dense0..%d inter_dense=%d expert_inter=%d "
                "ew_scale=%.4f yarn=%.2f\n", config_.n_layers, config_.hidden_dim,
                config_.num_q_heads, config_.num_kv_heads, config_.head_dim,
                config_.n_experts, config_.top_k, config_.first_k_dense_replace,
                config_.dense_inter_dim, config_.expert_inter_dim,
                config_.expert_weights_scale, config_.rope_scaling_factor);
        fprintf(stderr, "[hy3-verify] L0   dense_gate=%zu (expect %d) "
                "q_proj=%zu (expect %d) router(empty L0)=%zu\n",
                w0.dense_gate.size(), config_.dense_inter_dim*config_.hidden_dim,
                w0.q_proj.size(), config_.num_q_heads*config_.head_dim*config_.hidden_dim,
                w0.router.size());
        fprintf(stderr, "[hy3-verify] L1   router=%zu (expect %d, transposed) "
                "router_bias=%zu shared_gate=%zu\n",
                hy3_shared_[1].router.size(),
                config_.hidden_dim*config_.n_experts,
                hy3_shared_[1].router_bias.size(),
                hy3_shared_[1].shared_gate.size());
        fprintf(stderr, "[hy3-verify] L80  nextn_eh_proj=%zu (expect %d) "
                "enorm=%zu shared_head_norm=%zu router=%zu\n",
                wl.nextn_eh_proj.size(), config_.hidden_dim*2*config_.hidden_dim,
                wl.nextn_enorm.size(), wl.nextn_shared_head_norm.size(),
                wl.router.size());
        fprintf(stderr, "[hy3-verify] embed=%zu lm_head=%zu final_norm=%zu elems "
                "(expect %d / %d / %d)\n", host_embedding_table_.size(),
                host_lm_head_.size(), model_.final_norm_bytes/sizeof(bf16_t),
                config_.vocab_size*config_.hidden_dim,
                config_.vocab_size*config_.hidden_dim, config_.hidden_dim);
        /* spot-check finite values on L1 q_proj + router + embed row 0 */
        auto fin = [](const bf16_t* v, size_t n){
            int nf=0; float mx=-1e30f,mn=1e30f; for(size_t i=0;i<n;i++){
                float x=bf16f(v[i]); if(std::isnan(x)){nf++;continue;}
                if(x>mx)mx=x; if(x<mn)mn=x;} return std::make_tuple(nf,mn,mx);};
        auto f32fin=[](const float* v, size_t n){
            int nf=0; float mx=-1e30f,mn=1e30f; for(size_t i=0;i<n;i++){
                float x=v[i]; if(std::isnan(x)){nf++;continue;}
                if(x>mx)mx=x; if(x<mn)mn=x;} return std::make_tuple(nf,mn,mx);};
        auto[nf,mn,mx]=fin(hy3_shared_[1].q_proj.data(), 1<<16);
        auto[rnf,rmn,rmx]=f32fin(hy3_shared_[1].router.data(), 1<<16);
        auto[enf,emn,emx]=fin(host_embedding_table_.data(), 4096);
        fprintf(stderr,"[hy3-verify] L1 q_proj[0:65536]: nan=%d mn=%.4f mx=%.4f | "
                "L1 router[0:65536]: nan=%d mn=%.4f mx=%.4f | embed row0: nan=%d "
                "mn=%.4f mx=%.4f\n", nf,mn,mx, rnf,rmn,rmx, enf,emn,emx);
        std::exit(0);
    }

    /* Auto-discover a <model>.fst.hc sidecar holding only the Hybrid
     * Connection weights (TIDs 23-31).  Lets the 150 GB main .fst stay as-is
     * when HC is added retroactively.  Absence is silent: the engine falls
     * back to the (legacy) single-stream residual via has_hc=false. */
    {
        std::string hc_path = fst_path + ".hc";
        struct stat st;
        if (config_.arch == ARCH_DS4 &&
            ::stat(hc_path.c_str(), &st) == 0 && (st.st_mode & S_IFREG)) {
            fprintf(stderr, "[npu] Loading HC sidecar: %s (%lld bytes)\n",
                    hc_path.c_str(), (long long)st.st_size);
            load_hc_weights(hc_path);
        }
    }

    /* Auto-discover a <model>.fst.norm sidecar holding corrected RMSNorm
     * weights + attention sinks (TIDs 1,3,4,14,15,22).  The main .fst baked in
     * ~40x-too-small norms that starve the FFN; this overrides them by TID.
     * Absence is silent: the engine keeps whatever norms the main .fst had.
     * HY3 norms are baked correctly in the .fst (no override needed). */
    {
        std::string norm_path = fst_path + ".norm";
        struct stat st;
        if (config_.arch == ARCH_DS4 &&
            ::stat(norm_path.c_str(), &st) == 0 && (st.st_mode & S_IFREG)) {
            fprintf(stderr, "[npu] Loading norm sidecar: %s (%lld bytes)\n",
                    norm_path.c_str(), (long long)st.st_size);
            load_norm_override(norm_path);
        }
    }

    fprintf(stderr, "[npu] Total active hw_contexts: %zu (limit 8)\n",
            kernel_cache_->active_contexts());

    fprintf(stderr, "[npu] group_id: data=%d inst=%d (extracted from ext::kernel)\n",
            grp_, kernel_cache_->inst_group_id());

    // Preload ALL 43 layers' weights into persistent host_only BOs now.
    // This is ~5.7 GB for the main model; eliminates per-layer BO creation
    // and sync during inference (the 200s/iteration bottleneck).  DS4-only:
    // the HY3 path preloads its GQA BOs in a later phase (load_hy3_shared_weights
    // populates host vectors first; device BOs come with process_gqa).
    if (config_.arch == ARCH_DS4) {
    {double tp0=now(); preload_all_layers();
     fprintf(stderr,"[time] preload %.2fs\n", now()-tp0); fflush(stderr);}
    }
    } /* end !randmat */

    {float mx=-1e30f,mn=1e30f; double s=0; for(size_t i=0;i<host_embedding_table_.size();i++){float v=bf16f(host_embedding_table_[i]); if(std::isnan(v)){continue;} if(v>mx)mx=v; if(v<mn)mn=v; s+=(double)v;} fprintf(stderr,"[emb] after preload: %zu elems, mn=%.4f mx=%.4f sum=%.2f [0:4]=",host_embedding_table_.size(),mn,mx,s); for(int i=0;i<4;i++)fprintf(stderr,"%.4f ",bf16f(host_embedding_table_[i])); fprintf(stderr,"\n"); fflush(stderr);}

    // Pre-allocate scratch BOs for zero-copy dispatch.
    // Dummy/workspace BOs MUST be >= 1 MB: the IRON ext::kernel DMA engines
    // use args 3+ as AIE tile workspace.  A 4 KB workspace caused the NPU to
    // accept EXEC_CMD but never raise the syncobj (deadlock at Layer 0).
    // The 1 MB size matches the proven fst_npu_executor.cpp reference.
    const size_t MB = 1024 * 1024;
    bo_scratch_in_      = xrt::ext::bo(npu_device_, 4 * MB);
    bo_scratch_out_     = xrt::ext::bo(npu_device_, 4 * MB);
    bo_scratch_big_     = xrt::ext::bo(npu_device_, 128 * MB);
    bo_scratch_packed_  = xrt::ext::bo(npu_device_, 64 * MB);
    bo_dequant_out_     = xrt::ext::bo(npu_device_, 64 * MB);  // MXFP4 dequant: 48 MB (3*16MB)
    bo_d1_ = xrt::ext::bo(npu_device_, 1 * MB);
    bo_d2_ = xrt::ext::bo(npu_device_, 1 * MB);
    bo_d3_ = xrt::ext::bo(npu_device_, 1 * MB);
    bo_d4_ = xrt::ext::bo(npu_device_, 1 * MB);

    // Batched multi-core FFN BO: E_MC=6 slots x 3 (gate|up|down) x proj_elems.
    // 6*3*4096*2048*2 = 288 MB; holds all 6 experts' dequanted weights in the 3x-
    // stride layout the multicore GEMM TAPs read (gate@i*3*N*K, up@+N*K, down@+2*N*K).
    if (config_.arch == ARCH_DS4 && std::getenv("FST_MC_FFN")) {
        constexpr int E_MC = 6;
        const size_t batch_bytes = (size_t)E_MC * 3 * config_.hidden_dim * INTER_DIM * sizeof(bf16_t);
        bo_batch_w_ = xrt::ext::bo(npu_device_, batch_bytes);
    }
    // HY3 fused-FFN: same 302 MB bo_batch_w_ (8*3*4096*1536*2 == DS4's
    // 6*3*4096*2048*2 = 301,989,888 B) holds the 8-expert dequant output in the
    // expert-major [E,3,N,K] layout the MC kernels read.  Plus an 80 MB
    // host_only bo_fused_weight_ for the packed MXFP4 input (8*10,027,008 B).
    if (config_.arch == ARCH_HY3 && hy3_fused_ffn_) {
        constexpr int E_MC = 8;
        const size_t batch_bytes = (size_t)E_MC * 3 * config_.hidden_dim * config_.expert_inter_dim * sizeof(bf16_t);
        bo_batch_w_ = xrt::ext::bo(npu_device_, batch_bytes);   // 302 MB
        const size_t expert_pk = 3 * (size_t)(config_.expert_inter_dim / 32) * 17 * config_.hidden_dim;
        bo_fused_weight_ = xrt::bo(npu_device_, (size_t)E_MC * expert_pk,
                                   xrt::bo::flags::host_only, grp_);   // 80 MB host_only
    }

    // 4-set FFN scratch pool for the async expert pipeline.  Each set has its
    // own bo_w (dequant output B[N,K], read by the b_col_maj GEMM) plus the
    // gate/up/silu/mul/down chain intermediates, so expert E+1 can write its
    // set while expert E's down GEMM still reads E's set.  The 1 MB sizing is
    // the IRON workspace-minimum DMA floor (the actual tensors are <=128 KB).
    for (int i = 0; i < 4; ++i) {
        scratch_pool_[i].bo_w    = xrt::ext::bo(npu_device_, 64 * MB);
        scratch_pool_[i].bo_ha   = xrt::ext::bo(npu_device_, 1 * MB);
        scratch_pool_[i].bo_hb   = xrt::ext::bo(npu_device_, 1 * MB);
        scratch_pool_[i].bo_silu = xrt::ext::bo(npu_device_, 1 * MB);
        scratch_pool_[i].bo_mul  = xrt::ext::bo(npu_device_, 1 * MB);
        scratch_pool_[i].bo_down = xrt::ext::bo(npu_device_, 1 * MB);
        scratch_pool_[i].owner_e = -1;
    }
    pending_runs_.reserve(64);

    // MLA BO-to-BO chain intermediates.  max_seq=128, M_PAD=16 (M=16 std).
    // DS4-only: HY3 (GQA) uses expert_gemm_vec + ew_unified, no latent BOs;
    // its KV cache + GQA BOs are allocated in the process_gqa phase.
    if (config_.arch == ARCH_DS4) {
        const int S_MAX = config_.max_seq;   // 128
        bo_mla_h_      = xrt::ext::bo(npu_device_, (size_t)16 * config_.hidden_dim * sizeof(bf16_t));
        bo_mla_qc_     = xrt::ext::bo(npu_device_, (size_t)16 * MLA_Q_LORA * sizeof(bf16_t));
        bo_mla_kv_     = xrt::ext::bo(npu_device_, (size_t)16 * MLA_KV_LORA * sizeof(bf16_t));
        bo_mla_qfull_  = xrt::ext::bo(npu_device_, (size_t)16 * MLA_Q_FLAT * sizeof(bf16_t));
        bo_mla_kpe_    = xrt::ext::bo(npu_device_, (size_t)16 * MLA_N_HEADS * MLA_ROPE_DIM * sizeof(bf16_t));
        bo_mla_scores_ = xrt::ext::bo(npu_device_, (size_t)16 * S_MAX * sizeof(bf16_t));
        bo_mla_ao_     = xrt::ext::bo(npu_device_, (size_t)16 * MLA_Q_FLAT * sizeof(bf16_t));  /* SV out [M_LAT=512, head_dim=512] = [16,32768] */
        bo_mla_oa_     = xrt::ext::bo(npu_device_, (size_t)16 * config_.hidden_dim * sizeof(bf16_t));
        bo_mla_ob_     = xrt::ext::bo(npu_device_, (size_t)16 * config_.hidden_dim * sizeof(bf16_t));
        bo_rope_lut_   = xrt::ext::bo(npu_device_, (size_t)16 * MLA_ROPE_DIM * sizeof(bf16_t));
    }

    // KV cache: DS4 uses the compressed MLA latent (512) + k_pe; HY3's
    // uncompressed GQA KV (8 heads × 128 × 2) lives in a separate structure
    // allocated here (one row per cached position, num_kv*head_dim bf16 each).
    if (config_.arch == ARCH_DS4) {
        kv_cache_.resize(config_.n_layers);
        for (auto& kv : kv_cache_) {
            kv.kv_latent.resize(config_.max_seq * MLA_KV_LORA, 0);
            kv.k_pe.resize(config_.max_seq * MLA_N_HEADS * config_.rope_dim, 0);
        }
        init_kv_cache_bo();
    } else if (config_.arch == ARCH_HY3) {
        hy3_kv_cache_.assign(config_.n_layers, Hy3KVCache{});
        const int kv_row = config_.num_kv_heads * config_.head_dim;  /* 8*128=1024 */
        for (auto& kv : hy3_kv_cache_) {
            kv.k.assign((size_t)config_.max_seq * kv_row, f2bf(0.0f));
            kv.v.assign((size_t)config_.max_seq * kv_row, f2bf(0.0f));
            kv.n = 0;
        }
    }

    /* FST_HY3_ATTN_PROBE: host-only smoke test of process_gqa on layer 1 with a
     * synthetic hidden, before the full HY3 generate path is wired (phase 4).
     * Verifies the GQA math runs end-to-end (rmsnorm → proj GEMM → per-head norm
     * → YaRN RoPE → KV append → causal softmax → o proj → residual) without
     * NaN/crash and prints the output range.  Host-first: no NPU calls.  Set
     * instead of FST_HY3_LOAD_ONLY (the two are mutually exclusive). */
    if (config_.arch == ARCH_HY3 && std::getenv("FST_HY3_ATTN_PROBE")) {
        const int hd = config_.hidden_dim;
        auto synth = [&](bf16_t* h, int off) {
            for (int d = 0; d < hd; d++) h[d] = f2bf(0.01f * (float)((d + off) % 64) - 0.32f);
        };
        /* Case A: M=1 decode-like (single key → softmax identity — exercises
         * rmsnorm/proj/norm/RoPE/append/o_proj/residual). */
        {
            std::vector<bf16_t, AlignedAllocator<bf16_t>> h0(hd);
            synth(h0.data(), 0);
            seq_pos_ = 0;
            double t0 = now();
            process_gqa(1, h0.data(), 1);
            float mn = 1e30f, mx = -1e30f; int nf = 0;
            for (int d = 0; d < hd; d++) { float v = bf16f(h0[d]); if (std::isnan(v)) { nf++; continue; } if (v < mn) mn = v; if (v > mx) mx = v; }
            fprintf(stderr, "[hy3-attn-probe] L1 M=1: %.3fs nan=%d mn=%.4f mx=%.4f "
                    "kv.n=%d (exp 1)\n", now()-t0, nf, mn, mx, hy3_kv_cache_[1].n);
            if (nf) std::exit(1);
        }
        /* Case B: M=4 prefill-like — exercises multi-key causal softmax (each
         * query attends to keys 0..pos).  Fresh cache (reset kv.n). */
        {
            const int Mb = 4;
            std::vector<bf16_t, AlignedAllocator<bf16_t>> hb((size_t)Mb * hd);
            for (int m = 0; m < Mb; m++) synth(hb.data() + (size_t)m*hd, m*7);
            hy3_kv_cache_[1].n = 0;          /* reset; rows already zeroed at alloc */
            seq_pos_ = 0;
            double t0 = now();
            process_gqa(1, hb.data(), Mb);
            int nf = 0; float mn = 1e30f, mx = -1e30f;
            for (size_t i = 0; i < (size_t)Mb*hd; i++) { float v = bf16f(hb[i]); if (std::isnan(v)) { nf++; continue; } if (v < mn) mn = v; if (v > mx) mx = v; }
            fprintf(stderr, "[hy3-attn-probe] L1 M=%d: %.3fs nan=%d mn=%.4f mx=%.4f "
                    "kv.n=%d (exp %d)\n", Mb, now()-t0, nf, mn, mx, hy3_kv_cache_[1].n, Mb);
            if (nf) std::exit(1);
        }
        fflush(stderr);
        std::exit(0);
    }
    /* Compressor state: [2*ratio, comp_width] floats for ratio=4 (two-lane),
     * [ratio, comp_width] for ratio=128.  Zeroed; reset_compressor_state() is
     * called per generation.  DS4-only (V4 KV-compression; HY3 has none). */
    if (config_.arch == ARCH_DS4) {
        cmp_state_.assign(config_.n_layers, CompressorState{});
        for (int l = 0; l < config_.n_layers; l++) {
            int ratio = (l < (int)config_.compress_ratios.size()) ? config_.compress_ratios[l] : 0;
            if (ratio == 0) continue;
            int cw = (ratio == 4) ? 1024 : 512;
            int rows = (ratio == 4) ? 2 * ratio : ratio;
            cmp_state_[l].state_kv.assign((size_t)rows * cw, 0.0f);
            cmp_state_[l].state_sc.assign((size_t)rows * cw, 0.0f);
        }
    }

    pager_ = std::make_unique<ExpertPager>(fst_path, cache_mb * 1024ULL * 1024ULL);

    // Skip fork-based NPU validation — the fork can leave the NPU in a
    // bad state when the child process exits without properly releasing
     // hw_contexts and BOs.
    npu_ok = true;

    /* FST_HY3_FFN_PROBE: host-only smoke test of process_ffn_hy3 — L0 (dense
     * SwiGLU) and L1 (MoE: sigmoid router + 8 routed MXFP4 experts via the
     * pager + always-on shared expert).  Verifies the FFN math + the expert
     * bank MXFP4 dequant inverse run without NaN/crash and prints the range.
     * Host-first: the only SSD I/O is the 8 L1 expert blocks the pager loads
     * on demand.  Runs after the pager is created (ATTN_PROBE exits earlier;
     * the two probes don't co-exist). */
    if (config_.arch == ARCH_HY3 && std::getenv("FST_HY3_FFN_PROBE")) {
        const int hd = config_.hidden_dim;
        auto synth = [&](bf16_t* h, int off) {
            for (int d = 0; d < hd; d++) h[d] = f2bf(0.01f * (float)((d + off) % 64) - 0.32f);
        };
        auto range = [&](const bf16_t* h, int n, const char* tag, double dt) -> int {
            int nf = 0; float mn = 1e30f, mx = -1e30f;
            for (int i = 0; i < n; i++) { float v = bf16f(h[i]); if (std::isnan(v)) { nf++; continue; } if (v < mn) mn = v; if (v > mx) mx = v; }
            fprintf(stderr, "[hy3-ffn-probe] %s: %.3fs nan=%d mn=%.4f mx=%.4f\n", tag, dt, nf, mn, mx);
            return nf;
        };
        /* L0 dense: lid=0 < first_k_dense_replace(1) → dense SwiGLU, no router. */
        {
            std::vector<bf16_t, AlignedAllocator<bf16_t>> h0(hd);
            synth(h0.data(), 0);
            double t0 = now();
            process_ffn_hy3(0, h0.data(), 1);
            if (range(h0.data(), hd, "L0 dense M=1", now()-t0)) std::exit(1);
        }
        /* L1 MoE: router → 8 routed experts (SSD-loaded by pager) + shared. */
        {
            std::vector<bf16_t, AlignedAllocator<bf16_t>> h1(hd);
            synth(h1.data(), 11);
            double t0 = now();
            process_ffn_hy3(1, h1.data(), 1);
            if (range(h1.data(), hd, "L1 MoE  M=1", now()-t0)) std::exit(1);
        }
        fflush(stderr);
        std::exit(0);
    }

    /* FST_HY3_NEXTN_PROBE: host-only smoke test of the NextN MTP head (blk.80).
     * Runs forward_nextn_draft (n_max=3, p_min=0 → all 3) on a synthetic trunk
     * post-output_norm hidden + a token id, then re-runs step 0 to capture the
     * post-norm hidden range.  Verifies eh_proj + blk.80 GQA/MoE + shared_head_norm
     * + shared lm_head run without NaN/crash and produce finite draft tokens. */
    if (config_.arch == ARCH_HY3 && std::getenv("FST_HY3_NEXTN_PROBE")) {
        const int hd = config_.hidden_dim;
        const int V  = config_.vocab_size;
        std::vector<bf16_t, AlignedAllocator<bf16_t>> h_prev(hd);
        for (int d = 0; d < hd; d++) h_prev[d] = f2bf(0.01f * (float)(d % 64) - 0.32f);
        const int last_id = 1;
        double t0 = now();
        DraftResult dr = forward_nextn_draft(h_prev.data(), last_id, 3, 0.0f);
        double dt = now() - t0;
        /* Re-run step 0 (clean MTP KV) to capture the post-norm hidden range. */
        std::vector<bf16_t, AlignedAllocator<bf16_t>> hpost(hd);
        std::vector<float> logits(V);
        hy3_kv_cache_[config_.n_layers - 1].n = 0;
        int d0 = forward_nextn_step(last_id, h_prev.data(), 0, hpost.data(), logits.data());
        int nf = 0; float mn = 1e30f, mx = -1e30f;
        for (int d = 0; d < hd; d++) { float v = bf16f(hpost[d]); if (std::isnan(v)) { nf++; continue; } if (v < mn) mn = v; if (v > mx) mx = v; }
        int lf = 0; for (int v = 0; v < V; v++) if (std::isnan(logits[v])) lf++;
        fprintf(stderr, "[hy3-nextn-probe] draft tokens:");
        for (int t : dr.tokens) fprintf(stderr, " %d", t);
        fprintf(stderr, "  conf:");
        for (float c : dr.confidence) fprintf(stderr, " %.3f", c);
        fprintf(stderr, "\n[hy3-nextn-probe] step0 draft=%d h_post nan=%d mn=%.4f mx=%.4f "
                "logits_nan=%d V=%d  %.3fs\n", d0, nf, mn, mx, lf, V, dt);
        if (nf || lf || dr.tokens.empty()) std::exit(1);
        fflush(stderr);
        std::exit(0);
    }
}

FSTEngine::~FSTEngine() {
    kernel_cache_.reset();
    if (pager_) pager_->shutdown();
}

void FSTEngine::init_device() {}

void FSTEngine::load_shared_weights(const std::string& fst_path) {
    int fd=::open(fst_path.c_str(),O_RDONLY);auto hb=pread(fd,128,0);auto*h=(const FSTH*)hb.data();
    rope_freq_base_=(h->rf>0.0f)?h->rf:10000.0f;

    config_.hidden_dim = (int)h->hd;
    config_.n_layers = (int)h->nl;
    config_.n_experts = (int)h->ne;
    config_.top_k = (int)h->tk;
    config_.vocab_size = (int)h->vs;
    config_.expert_block_bytes = (size_t)h->eb;

    /* Arch branch: HY3 (Tencent Hunyuan-3.0, GQA) loads via a TID-based loader
     * and returns here — the DS4 (MLA) shape-keyed map below does NOT apply
     * (it collides on HY3's k_proj/v_proj "1024x4096x1" and attn/ffn_norm
     * "4096x1x1").  The ctor pre-read already set config_.arch; re-derive from
     * this header so the branch is self-contained, then hand off. */
    if (h->qh == 64 && h->kh == 8 && h->hd2 == 128) {
        config_.arch = ARCH_HY3;
        config_.num_q_heads  = (int)h->qh;        /* 64 */
        config_.num_kv_heads = (int)h->kh;        /* 8  */
        config_.head_dim    = (int)h->hd2;        /* 128 */
        config_.expert_inter_dim = (int)h->id;   /* 1536 */
        config_.rope_dim    = config_.head_dim;
        config_.first_k_dense_replace = (int)(h->r1 & 0xFFFFu);
        config_.expert_gating_func    = (int)((h->r1 >> 16) & 0xFFFFu);
        config_.rope_scaling_factor   = (float)((h->r2 >> 32) & 0xFFFFFFFFu) / 1000.0f;
        config_.expert_weights_scale  = (float)(h->r2 & 0xFFFFFFFFu) / 1e6f;
        config_.yarn_orig_ctx = 262144;
        ::close(fd);
        load_hy3_shared_weights(fst_path);
        return;
    }

    /* DeepSeek-V4 FFN/router params: hardcoded to the HF config.json values
     * (swiglu_limit=10, n_hash_layers=3, scoring_func=sqrtsoftplus) so no
     * .fst reconversion is required.  FST_SWIGLU_LIMIT overrides (−1 disables
     * the clamp — for diagnosing the residual explosion). */
    if (const char* e = std::getenv("FST_SWIGLU_LIMIT")) {
        float v = std::strtof(e, nullptr);
        if (v < 0)      config_.swiglu_limit = 0;   /* -1 disables (clamp guard is > 0) */
        else            config_.swiglu_limit = v;
    }
    /* route_scale (routed_scaling_factor=1.5 in HF config.json, model.py 588).
     * Default 1.5 matches HF; FST_ROUTE_SCALE overrides (set 1.0 to reproduce
     * the prior under-driven behavior for comparison). */
    if (const char* e = std::getenv("FST_ROUTE_SCALE")) {
        config_.route_scale = std::strtof(e, nullptr);
    }

    /* Hash-routing sidecar: <model>.fst.tid2eid = [n_hash_layers, vocab, top_k]
     * int32, extracted from HF `layers.{0..2}.ffn.gate.tid2eid`.  Optional — if
     * absent, hash layers L0..L2 fall back to score-based routing. */
    {
        std::string tid_path = fst_path + ".tid2eid";
        int tfd = ::open(tid_path.c_str(), O_RDONLY);
        if (tfd >= 0) {
            const size_t want = (size_t)config_.n_hash_layers * config_.vocab_size * config_.top_k * sizeof(int);
            tid2eid_.resize(config_.n_hash_layers * config_.vocab_size * config_.top_k);
            ssize_t got = ::pread(tfd, tid2eid_.data(), want, 0);
            ::close(tfd);
            if (got == (ssize_t)want) {
                tid2eid_loaded_ = true;
                fprintf(stderr, "[hash] loaded tid2eid sidecar: %zu bytes (%d layers, vocab=%d, topk=%d)\n",
                        want, config_.n_hash_layers, config_.vocab_size, config_.top_k);
            } else {
                tid2eid_.clear();
                fprintf(stderr, "[hash] tid2eid sidecar short read (%zd/%zu) — hash routing disabled\n", got, want);
            }
        }
    }

    shared_.resize(config_.n_layers);
    size_t ne=(size_t)h->sdc;auto db=pread(fd,ne*64,(off_t)h->sdo);
    struct T{int l;uint64_t o,sz;int q;};
    std::unordered_map<std::string,std::vector<T>> sm;
    for(size_t i=0;i<ne;i++){auto*e=(const FSTE*)(db.data()+i*64);char k[64];
        snprintf(k,sizeof(k),"%llux%llux%llu",(unsigned long long)e->s0,(unsigned long long)e->s1,(unsigned long long)e->s2);
        sm[k].push_back({(int)e->lid,e->off,e->sz,(int)e->qt});}
    /* V4 compress_ratios global (TID 40, F32 [n_layers]) — load before the
     * per-layer loop so each layer knows its ratio.  Falls back to the ds4.c
     * default pattern (0/0/4/128/...) if the tensor is absent. */
    {
        config_.compress_ratios.assign(config_.n_layers, 0);
        bool found = false;
        for (size_t i=0;i<ne;i++){auto*e=(const FSTE*)(db.data()+i*64);
            if (e->tid != 40) continue;
            auto raw = pread(fd, (size_t)e->sz, (off_t)e->off);
            int n = (int)(raw.size() / sizeof(float));
            if (n > config_.n_layers) n = config_.n_layers;
            if (n > 0) {
                /* TID 40 is F32 [n_layers]; convert float→int (the bit pattern
                 * is NOT the ratio — 4.0f as bits is 0x40800000). */
                std::vector<float> frat(n);
                memcpy(frat.data(), raw.data(), (size_t)n * sizeof(float));
                for (int il = 0; il < n; il++)
                    config_.compress_ratios[il] = (int)frat[il];
                found = true;
            }
            break;
        }
        if (!found) {
            for (int il=0; il<config_.n_layers; il++)
                config_.compress_ratios[il] = (il<2 || il>=41) ? 0 : (il%2==0 ? 4 : 128);
        }
        fprintf(stderr, "[cmp] compress_ratios: n=%d vals=[", config_.n_layers);
        for (int il=0; il<config_.n_layers && il<43; il++) fprintf(stderr, "%d,", config_.compress_ratios[il]);
        fprintf(stderr, "]\n"); fflush(stderr);
    }
    auto pop=[&](const char*k,int lid,bool f32,std::vector<float>&fv,std::vector<bf16_t, AlignedAllocator<bf16_t>>&bv)->bool{
        auto it=sm.find(k);if(it==sm.end()||it->second.empty())return false;
        size_t bi=0;for(size_t j=0;j<it->second.size();j++)if(it->second[j].l==lid||it->second[j].l==0xFFFF){bi=j;break;}
        auto&ti=it->second[bi];auto raw=pread(fd,(size_t)ti.sz,(off_t)ti.o);
        size_t nel=(size_t)ti.sz/2;
        if(ti.q==1){size_t s0=1,s1=1;sscanf(k,"%zux%zu",&s0,&s1);nel=s0*(s1?s1:1);size_t nb=(nel+31)/32;
            auto*sc=(const uint16_t*)raw.data();auto*v8=(const int8_t*)(raw.data()+nb*2);std::vector<float> fq(nel);
            for(size_t b=0;b<nb;b++){float sf=fp16f(sc[b]);if(fabsf(sf)<1e-12f)sf=1.f;
                for(int j=0;j<32;j++){size_t ix=b*32+j;if(ix<nel)fq[ix]=(float)v8[ix]*sf;}}
            if(f32){fv=std::move(fq);}else{bv.resize(nel);for(size_t x=0;x<nel;x++)bv[x]=f2bf(fq[x]);}}
        else{bv.resize(nel);memcpy(bv.data(),raw.data(),ti.sz);if(f32){fv.resize(nel);for(size_t x=0;x<nel;x++)fv[x]=bf16f(bv[x]);bv.clear();}}
        it->second.erase(it->second.begin()+bi);return true;};
    auto pb=[&](const char*k,int l,std::vector<bf16_t, AlignedAllocator<bf16_t>>&v){std::vector<float>d;pop(k,l,false,d,v);};
    auto pf=[&](const char*k,int l,std::vector<float>&v){std::vector<bf16_t, AlignedAllocator<bf16_t>>d;pop(k,l,true,v,d);};
    /* Per-layer TID loader (raw bytes) — for F32 tensors like attn_sinks that
     * the bf16-oriented pop() helper cannot read.  Matches tid AND lid. */
    auto pop_tid_layer=[&](uint32_t target_tid,int lid)->std::vector<uint8_t>{
        for(size_t i=0;i<ne;i++){auto*e=(const FSTE*)(db.data()+i*64);
            if(e->tid!=target_tid||(int)e->lid!=lid)continue;
            return pread(fd,(size_t)e->sz,(off_t)e->off);}
        return {};};
    for(int l=0;l<config_.n_layers;l++){auto&w=shared_[l];
        pb("8192x4096x1",l,w.wo_a);  /* [N,K] native — b_col_maj reads directly */
        pb("4096x8192x1",l,w.wo_b);  /* [N,K] native — b_col_maj reads directly */
        pb("1024x4096x1",l,w.wq_a);  /* [N,K] native — b_col_maj reads directly */
        pb("32768x1024x1",l,w.wq_b); /* [N,K] native — b_col_maj reads directly */
        pb("512x4096x1",l,w.wkv);    /* [N,K] native — b_col_maj (kvc N-pads 512->1024 in dispatch) */
        pb("4096x1x1",l,w.attn_norm);pb("4096x1x1",l,w.moe_norm);
        pb("1024x1x1",l,w.q_norm);pb("512x1x1",l,w.kv_norm);
        pb("2048x4096x1",l,w.shared_gate);  /* [N,K] native — b_col_maj reads directly (NO transpose: converter _maybe_dequant_q8 already stores [out=N,in=K], matching the routed sc.bo_w layout the same gemm kernels read).  A transpose here made the kernel compute A@W instead of A@Wᵀ → shared expert output ~12x too small (L0 |shared|mx 0.16 vs HF 1.93). */
        pb("2048x4096x1",l,w.shared_up);
        pb("4096x2048x1",l,w.shared_down);
        {pf("256x4096x1",l,w.router); transpose_f32(w.router,256,4096);}
        pb("512x1024x1",l,w.w_k_decompress); transpose_bf16(w.w_k_decompress,512,1024);
        pb("512x1024x1",l,w.w_v_decompress); transpose_bf16(w.w_v_decompress,512,1024);
        pb("4096x4096x1",l,w.w_k_pe);  transpose_bf16(w.w_k_pe,4096,4096);
        if (l == 0) {
            fprintf(stderr, "[audit] L0 wo_a: size=%zu => [%d x %d] (rows x cols, after transpose)\n",
                    w.wo_a.size(), (int)(w.wo_a.size() / 8192), 8192);
            fprintf(stderr, "[audit] L0 wo_b: size=%zu => [%d x %d]\n",
                    w.wo_b.size(), 8192, (int)(w.wo_b.size() / 8192));
            fprintf(stderr, "[audit] L0 heads input to O-proj: [M_PAD x %d] (n_heads*head_dim, h-outer d-inner)\n",
                    MLA_N_HEADS * MLA_HEAD_DIM);
        }
        /* Attention sink: TID 22, [n_heads] F32 per-layer.  Applied as a per-head
         * logit in the softmax denominator (no value vector).  Falls back to 0
         * (no sink) if the tensor is absent. */
        {
            auto sink_raw = pop_tid_layer(22, l);
            const size_t need = (size_t)MLA_N_HEADS * sizeof(float);
            w.attn_sinks.assign(MLA_N_HEADS, 0.0f);
            if (sink_raw.size() >= need)
                memcpy(w.attn_sinks.data(), sink_raw.data(), need);
        }
        /* Router bias (TID 10): per-expert topk correction bias, n_experts F32
         * (or BF16 — converter stores either).  Added to sqrtsoftplus score
         * BEFORE topk but does NOT enter the routing weight (model.py 578-588).
         * Absent for hash-routed layers L0..L2 (falls back to 0 → no-op). */
        {
            auto bias_raw = pop_tid_layer(10, l);
            w.router_bias.assign(config_.n_experts, 0.0f);
            if (bias_raw.size() == config_.n_experts * sizeof(float))
                memcpy(w.router_bias.data(), bias_raw.data(), bias_raw.size());
            else if (bias_raw.size() == config_.n_experts * sizeof(uint16_t)) {
                auto* b16 = (const uint16_t*)bias_raw.data();
                for (size_t i = 0; i < (size_t)config_.n_experts; i++)
                    w.router_bias[i] = fp16f(b16[i]);
            }
        }
        /* V4 KV compressor weights (TIDs 32-35), only for ratio>0 layers.
         *   TID 32 CMP_APE   : F32 [ratio, comp_width]
         *   TID 33 CMP_WKV   : BF16 [comp_width, 4096]  (native [N,K], NO transpose)
         *   TID 34 CMP_GATE  : BF16 [comp_width, 4096]  (native [N,K], NO transpose)
         *   TID 35 CMP_NORM  : BF16 [512]
         * comp_width = (ratio==4)?1024:512.  ratio=0 layers load nothing. */
        {
            w.cmp_ratio = (l < (int)config_.compress_ratios.size()) ? config_.compress_ratios[l] : 0;
            w.cmp_comp_width = (w.cmp_ratio == 4) ? 1024 : (w.cmp_ratio == 128 ? 512 : 0);
            if (w.cmp_ratio != 0) {
                auto ape_raw = pop_tid_layer(32, l);
                if (ape_raw.size() >= (size_t)w.cmp_ratio * w.cmp_comp_width * sizeof(float)) {
                    w.cmp_ape.resize((size_t)w.cmp_ratio * w.cmp_comp_width);
                    memcpy(w.cmp_ape.data(), ape_raw.data(), ape_raw.size() < w.cmp_ape.size()*sizeof(float) ? ape_raw.size() : w.cmp_ape.size()*sizeof(float));
                }
                auto wkv_raw = pop_tid_layer(33, l);
                if (wkv_raw.size() >= (size_t)w.cmp_comp_width * 4096 * sizeof(bf16_t)) {
                    w.cmp_wkv.resize((size_t)w.cmp_comp_width * 4096);
                    memcpy(w.cmp_wkv.data(), wkv_raw.data(), w.cmp_wkv.size() * sizeof(bf16_t));
                }
                auto wgate_raw = pop_tid_layer(34, l);
                if (wgate_raw.size() >= (size_t)w.cmp_comp_width * 4096 * sizeof(bf16_t)) {
                    w.cmp_wgate.resize((size_t)w.cmp_comp_width * 4096);
                    memcpy(w.cmp_wgate.data(), wgate_raw.data(), w.cmp_wgate.size() * sizeof(bf16_t));
                }
                auto norm_raw = pop_tid_layer(35, l);
                if (norm_raw.size() >= (size_t)512 * sizeof(bf16_t)) {
                    w.cmp_norm.resize(512);
                    memcpy(w.cmp_norm.data(), norm_raw.data(), 512 * sizeof(bf16_t));
                }
                if (l == 2 || l == 40)
                    fprintf(stderr, "[cmp] L%d ratio=%d cw=%d ape=%zu wkv=%zu norm=%zu\n",
                            l, w.cmp_ratio, w.cmp_comp_width,
                            w.cmp_ape.size(), w.cmp_wkv.size(), w.cmp_norm.size());
            }
        }
        /* HC (Hybrid Connection) per-layer weights, read by TID.  The fn
         * matrices (attn & ffn) share the 16384x24 dim-key, so a dim-key
         * lookup would collide — TID-based raw read is unambiguous.  The
         * checkpoint stores fn as BF16 [HC_MIX_DIM, HC_DIM] = [24, 16384];
         * transpose to [HC_DIM, HC_MIX_DIM] so npu_gemm_hc (A@B, B=[K,N], transposed internally)
         * computes mix = flat @ fn.  scale is F32 [3]; base is F32 [24]. */
        {
            const size_t fn_n = (size_t)HC_DIM * HC_MIX_DIM;
            auto a_fn = pop_tid_layer(23, l);
            if (a_fn.size() >= fn_n * sizeof(bf16_t)) {
                w.hc_attn_fn.resize(fn_n);
                memcpy(w.hc_attn_fn.data(), a_fn.data(), fn_n * sizeof(bf16_t));
                transpose_bf16(w.hc_attn_fn, HC_MIX_DIM, HC_DIM);  /* [24,16384]->[16384,24] */
            }
            auto a_sc = pop_tid_layer(24, l);
            if (a_sc.size() >= 3 * sizeof(float)) { w.hc_attn_scale.resize(3); memcpy(w.hc_attn_scale.data(), a_sc.data(), 3 * sizeof(float)); }
            auto a_bs = pop_tid_layer(25, l);
            if (a_bs.size() >= HC_MIX_DIM * sizeof(float)) { w.hc_attn_base.resize(HC_MIX_DIM); memcpy(w.hc_attn_base.data(), a_bs.data(), HC_MIX_DIM * sizeof(float)); }
            auto f_fn = pop_tid_layer(26, l);
            if (f_fn.size() >= fn_n * sizeof(bf16_t)) {
                w.hc_ffn_fn.resize(fn_n);
                memcpy(w.hc_ffn_fn.data(), f_fn.data(), fn_n * sizeof(bf16_t));
                transpose_bf16(w.hc_ffn_fn, HC_MIX_DIM, HC_DIM);
            }
            auto f_sc = pop_tid_layer(27, l);
            if (f_sc.size() >= 3 * sizeof(float)) { w.hc_ffn_scale.resize(3); memcpy(w.hc_ffn_scale.data(), f_sc.data(), 3 * sizeof(float)); }
            auto f_bs = pop_tid_layer(28, l);
            if (f_bs.size() >= HC_MIX_DIM * sizeof(float)) { w.hc_ffn_base.resize(HC_MIX_DIM); memcpy(w.hc_ffn_base.data(), f_bs.data(), HC_MIX_DIM * sizeof(float)); }
        }
        if(w.attn_norm.empty())w.attn_norm.assign(config_.hidden_dim,f2bf(1.f));
        if(w.moe_norm.empty())w.moe_norm.assign(config_.hidden_dim,f2bf(1.f));}

    auto pop_tid=[&](uint32_t target_tid)->std::vector<uint8_t>{
        for(size_t i=0;i<ne;i++){auto*e=(const FSTE*)(db.data()+i*64);
            if(e->tid!=target_tid)continue;return pread(fd,(size_t)e->sz,(off_t)e->off);}
        return {};};
    {
        auto emb_raw=pop_tid(0);
        if(!emb_raw.empty()){
            model_.embedding_bytes=emb_raw.size();
            model_.embedding_table=(bf16_t*)malloc(emb_raw.size());
            memcpy(model_.embedding_table,emb_raw.data(),emb_raw.size());
            host_embedding_table_.resize(emb_raw.size()/sizeof(bf16_t));
            memcpy(host_embedding_table_.data(),emb_raw.data(),emb_raw.size());
            if (config_.hidden_dim > 0)
                config_.vocab_size = (int)(emb_raw.size() / (config_.hidden_dim * sizeof(bf16_t)));
            {float mx=-1e30f,mn=1e30f; double s=0; for(size_t i=0;i<host_embedding_table_.size();i++){float v=bf16f(host_embedding_table_[i]); if(std::isnan(v)){continue;} if(v>mx)mx=v; if(v<mn)mn=v; s+=(double)v;} fprintf(stderr,"[emb] after load: %zu elems, mn=%.4f mx=%.4f sum=%.2f [0:4]=",host_embedding_table_.size(),mn,mx,s); for(int i=0;i<4;i++)fprintf(stderr,"%.4f ",bf16f(host_embedding_table_[i])); fprintf(stderr,"\n"); fflush(stderr);}
        }else{
            throw std::runtime_error("FATAL: Embedding (TID=0) missing from .fst file");
        }
    }
    {
        auto lm_raw=pop_tid(2);
        if(!lm_raw.empty()){
            model_.lm_head_bytes=lm_raw.size();
            model_.lm_head=(bf16_t*)malloc(lm_raw.size());
            memcpy(model_.lm_head,lm_raw.data(),lm_raw.size());
            host_lm_head_.resize(lm_raw.size()/sizeof(bf16_t));
            memcpy(host_lm_head_.data(),lm_raw.data(),lm_raw.size());
        }else{
            throw std::runtime_error("FATAL: LM_Head (TID=2) missing from .fst file");
        }
    }
    {
        const int V = config_.vocab_size, D = config_.hidden_dim;
        const int N_PAD = 131072;  /* vocab padded to a multiple of the 2048 N-tile */
        const size_t lm_bytes = (size_t)N_PAD * D * sizeof(bf16_t);
        bo_lm_head_ = xrt::bo(npu_device_, lm_bytes, xrt::bo::flags::host_only,
                                kernel_cache_->data_group_id());
        auto* dst = bo_lm_head_.map<bf16_t*>();
        memset(dst, 0, lm_bytes);
        /* host_lm_head_ is [V, D] row-major; store transposed as [D, V] then
         * zero-pad the remaining (N_PAD - V) columns so every 2048-col N-tile
         * the lm_head_gemm kernel reads is fully in-bounds. */
        for (int v = 0; v < V; v++)
            for (int d = 0; d < D; d++)
                dst[d * N_PAD + v] = host_lm_head_[v * D + d];
        bo_lm_head_.sync(XCL_BO_SYNC_BO_TO_DEVICE, lm_bytes, 0);
        lm_head_n_pad_ = N_PAD;
    }
    {
        auto fn_raw=pop_tid(1);
        if(!fn_raw.empty()){
            model_.final_norm_bytes=fn_raw.size();
            model_.final_norm=(bf16_t*)malloc(fn_raw.size());
            memcpy(model_.final_norm,fn_raw.data(),fn_raw.size());
        }else{
            throw std::runtime_error("FATAL: Final_Norm (TID=1) missing from .fst file");
        }
    }
    /* HC global output-collapse weights: hc_head_fn=[HC_DIM, N_HC] BF16 (TID 29),
     * hc_head_scale=[1] F32 (TID 30), hc_head_base=[4] F32 (TID 31).  The
     * checkpoint stores fn as [N_HC, HC_DIM] = [4, 16384]; transpose to
     * [HC_DIM, N_HC] for the npu_gemm_hc matvec.  Missing tensors => the .fst
     * predates HC; output_hc_head then collapses to stream 0. */
    {
        const size_t fn_n = (size_t)HC_DIM * N_HC;
        auto hfn = pop_tid(29);
        if (hfn.size() >= fn_n * sizeof(bf16_t)) {
            model_.hc_head_fn.resize(fn_n);
            memcpy(model_.hc_head_fn.data(), hfn.data(), fn_n * sizeof(bf16_t));
            transpose_bf16(model_.hc_head_fn, N_HC, HC_DIM);  /* [4,16384]->[16384,4] */
        }
        auto hsc = pop_tid(30);
        if (hsc.size() >= sizeof(float)) { model_.hc_head_scale.resize(1); memcpy(model_.hc_head_scale.data(), hsc.data(), sizeof(float)); }
        auto hbs = pop_tid(31);
        if (hbs.size() >= N_HC * sizeof(float)) { model_.hc_head_base.resize(N_HC); memcpy(model_.hc_head_base.data(), hbs.data(), N_HC * sizeof(float)); }
    }
    ::close(fd);
}

/* ── HY3 (Hunyuan-3.0) TID-based shared-bank loader ──────────────────────
 * The DS4 loader keys tensors by "s0xs1xs2" shape, which collides on HY3
 * (k_proj & v_proj are both "1024x4096x1"; attn_norm & ffn_norm both
 * "4096x1x1").  HY3 instead reads each shared-bank tensor by (tid,lid) and
 * interprets the entry qtype directly.  HY3's shared bank is BF16 (qt=2) +
 * F32 router/bias (qt=3) — NO Q8_0 — so a raw memcpy suffices (no dequant).
 * Only the router is transposed [n_experts,hidden]→[hidden,n_experts] to
 * match router_gemm's [K,N] B layout (the DS4 convention).  Globals (embed/
 * lm_head/output_norm, lid=0xFFFF) are stored [vocab,hidden]/[hidden] and
 * copied straight into model_ + host_embedding_table_/host_lm_head_. */
void FSTEngine::load_hy3_shared_weights(const std::string& fst_path) {
    /* HY3 shared-bank TIDs (must match scripts/fst_converter.py). */
    constexpr uint32_t T_EMBED=0, T_OUTPUT_NORM=1, T_LM_HEAD=2,
        T_INPUT_NORM=3, T_POST_ATTN_NORM=4, T_Q_PROJ=5, T_K_PROJ=6,
        T_V_PROJ=7, T_O_PROJ=8, T_ROUTER=9, T_ROUTER_BIAS=10,
        T_SHARED_GATE=11, T_SHARED_UP=12, T_SHARED_DOWN=13,
        T_HY3_Q_NORM=50, T_HY3_K_NORM=51, T_HY3_DENSE_GATE=52,
        T_HY3_DENSE_UP=53, T_HY3_DENSE_DOWN=54, T_HY3_NEXTN_EH_PROJ=55,
        T_HY3_NEXTN_ENORM=56, T_HY3_NEXTN_HNORM=57, T_HY3_NEXTN_SHN=58;
    constexpr int GLOBAL = 0xFFFF;

    int fd = ::open(fst_path.c_str(), O_RDONLY);
    if (fd < 0) { ::perror("open fst"); throw std::runtime_error("HY3 open fst"); }
    auto hb = pread(fd, 128, 0);
    auto* h = (const FSTH*)hb.data();
    const size_t ne = (size_t)h->sdc;
    auto db = pread(fd, ne * 64, (off_t)h->sdo);

    /* Locate one directory entry by (tid,lid); return raw bytes + shape. */
    struct Ent { std::vector<uint8_t> raw; uint64_t s0=0, s1=0; bool found=false; };
    auto get = [&](uint32_t tid, int lid) -> Ent {
        for (size_t i = 0; i < ne; i++) {
            auto* e = (const FSTE*)(db.data() + i * 64);
            if (e->tid != tid || (int)e->lid != lid) continue;
            return { pread(fd, (size_t)e->sz, (off_t)e->off), e->s0, e->s1, true };
        }
        return {};
    };
    auto to_bf16 = [&](const Ent& en, std::vector<bf16_t, AlignedAllocator<bf16_t>>& v) {
        if (!en.found) return false;
        v.resize(en.raw.size() / sizeof(bf16_t));
        std::memcpy(v.data(), en.raw.data(), en.raw.size());
        return true;
    };
    auto to_f32 = [&](const Ent& en, std::vector<float>& v) {
        if (!en.found) return false;
        v.resize(en.raw.size() / sizeof(float));
        std::memcpy(v.data(), en.raw.data(), en.raw.size());
        return true;
    };

    hy3_shared_.assign(config_.n_layers, Hy3LayerWeights{});
    const int last = config_.n_layers - 1;   /* 80 */
    for (int l = 0; l < config_.n_layers; l++) {
        auto& w = hy3_shared_[l];
        /* every block (0..80): GQA attention + per-head Q/K norm + ffn_norm */
        to_bf16(get(T_INPUT_NORM,    l), w.attn_norm);   /* [4096] */
        to_bf16(get(T_Q_PROJ,        l), w.q_proj);      /* [8192,4096] */
        to_bf16(get(T_K_PROJ,        l), w.k_proj);      /* [1024,4096] */
        to_bf16(get(T_V_PROJ,        l), w.v_proj);      /* [1024,4096] */
        to_bf16(get(T_O_PROJ,        l), w.o_proj);      /* [4096,8192] */
        to_bf16(get(T_HY3_Q_NORM,    l), w.q_norm);      /* [128] */
        to_bf16(get(T_HY3_K_NORM,    l), w.k_norm);      /* [128] */
        to_bf16(get(T_POST_ATTN_NORM,l), w.ffn_norm);    /* [4096] */

        if (l < config_.first_k_dense_replace) {
            /* L0: dense SwiGLU FFN (inter from the stored gate shape). */
            auto dg = get(T_HY3_DENSE_GATE, l);
            to_bf16(dg,                       w.dense_gate); /* [13312,4096] */
            to_bf16(get(T_HY3_DENSE_UP,   l), w.dense_up);
            to_bf16(get(T_HY3_DENSE_DOWN, l), w.dense_down);/* [4096,13312] */
            if (dg.found && config_.dense_inter_dim == 0)
                config_.dense_inter_dim = (int)dg.s0;     /* 13312 */
        }
        if (l >= config_.first_k_dense_replace) {
            /* L1..L80 (incl. the MTP block): sigmoid router + bias + shared.
             * Router transposed [n_experts,hidden]→[hidden,n_experts] for router_gemm. */
            auto rr = get(T_ROUTER, l);                 /* F32 [192,4096] */
            if (to_f32(rr, w.router))
                transpose_f32(w.router, (int)rr.s0, (int)rr.s1);  /* →[4096,192] */
            to_f32(get(T_ROUTER_BIAS, l), w.router_bias);         /* F32 [192] */
            to_bf16(get(T_SHARED_GATE, l), w.shared_gate);        /* [1536,4096] */
            to_bf16(get(T_SHARED_UP,   l), w.shared_up);
            to_bf16(get(T_SHARED_DOWN, l), w.shared_down);       /* [4096,1536] */
        }
        if (l == last) {
            /* L80: NextN MTP head tensors (blk.80 also carries the MoE tensors above). */
            to_bf16(get(T_HY3_NEXTN_EH_PROJ, l), w.nextn_eh_proj);  /* [4096,8192] */
            to_bf16(get(T_HY3_NEXTN_ENORM,   l), w.nextn_enorm);   /* [4096] */
            to_bf16(get(T_HY3_NEXTN_HNORM,   l), w.nextn_hnorm);   /* [4096] */
            to_bf16(get(T_HY3_NEXTN_SHN,     l), w.nextn_shared_head_norm);
        }

        if ((l + 1) % 10 == 0 || l + 1 == config_.n_layers) {
            fprintf(stderr, "[hy3-load] L%-3d q=%zu k=%zu v=%zu o=%zu qn=%zu "
                    "router=%zu rbias=%zu sg=%zu dense_g=%zu eh=%zu\n",
                    l, w.q_proj.size(), w.k_proj.size(), w.v_proj.size(),
                    w.o_proj.size(), w.q_norm.size(), w.router.size(),
                    w.router_bias.size(), w.shared_gate.size(),
                    w.dense_gate.size(), w.nextn_eh_proj.size());
            fflush(stderr);
        }
    }

    /* globals (lid=0xFFFF): embed [vocab,hidden], lm_head [vocab,hidden],
     * output_norm [hidden] — all BF16, direct memcpy (engine indexes embed
     * by tid*hidden and transposes lm_head to padded [hidden,N_PAD] later). */
    {
        auto e = get(T_EMBED, GLOBAL);
        if (e.found) {
            model_.embedding_bytes = e.raw.size();
            model_.embedding_table = (bf16_t*)std::malloc(e.raw.size());
            std::memcpy(model_.embedding_table, e.raw.data(), e.raw.size());
            host_embedding_table_.resize(e.raw.size() / sizeof(bf16_t));
            std::memcpy(host_embedding_table_.data(), e.raw.data(), e.raw.size());
        }
    }
    {
        auto fn = get(T_OUTPUT_NORM, GLOBAL);
        if (fn.found) {
            model_.final_norm_bytes = fn.raw.size();
            model_.final_norm = (bf16_t*)std::malloc(fn.raw.size());
            std::memcpy(model_.final_norm, fn.raw.data(), fn.raw.size());
        }
    }
    {
        auto lm = get(T_LM_HEAD, GLOBAL);
        if (lm.found) {
            model_.lm_head_bytes = lm.raw.size();
            model_.lm_head = (bf16_t*)std::malloc(lm.raw.size());
            std::memcpy(model_.lm_head, lm.raw.data(), lm.raw.size());
            host_lm_head_.resize(lm.raw.size() / sizeof(bf16_t));
            std::memcpy(host_lm_head_.data(), lm.raw.data(), lm.raw.size());
        }
    }
    fprintf(stderr, "[hy3-load] done: %d layers, dense_inter_dim=%d, "
            "embed=%zu lm_head=%zu final_norm=%zu elems\n",
            config_.n_layers, config_.dense_inter_dim,
            host_embedding_table_.size(), host_lm_head_.size(),
            model_.final_norm_bytes / sizeof(bf16_t));
    fflush(stderr);
    ::close(fd);
}

void FSTEngine::load_hc_weights(const std::string& hc_path) {
    /* Read ONLY the HC TIDs (23-31) from a <model>.fst.hc sidecar and populate
     * shared_[l].hc_* / model_.hc_*.  The sidecar is a normal .fst (header +
     * shared dir + shared data, empty expert bank), so we reuse FSTH/FSTE.
     * Mirrors the inline HC reads in load_shared_weights. */
    int fd = ::open(hc_path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[npu] HC sidecar open failed: %s\n", hc_path.c_str()); return; }
    auto hb = pread(fd, 128, 0); auto* h = (const FSTH*)hb.data();
    size_t ne = (size_t)h->sdc; auto db = pread(fd, ne * 64, (off_t)h->sdo);

    auto pop_tid_layer = [&](uint32_t target_tid, int lid) -> std::vector<uint8_t> {
        for (size_t i = 0; i < ne; i++) { auto* e = (const FSTE*)(db.data() + i * 64);
            if (e->tid != target_tid || (int)e->lid != lid) continue;
            return pread(fd, (size_t)e->sz, (off_t)e->off); }
        return {}; };
    auto pop_tid = [&](uint32_t target_tid) -> std::vector<uint8_t> {
        for (size_t i = 0; i < ne; i++) { auto* e = (const FSTE*)(db.data() + i * 64);
            if (e->tid != target_tid) continue;
            return pread(fd, (size_t)e->sz, (off_t)e->off); }
        return {}; };

    int n_layers = config_.n_layers;
    const size_t fn_n = (size_t)HC_DIM * HC_MIX_DIM;
    for (int l = 0; l < n_layers; l++) {
        auto& w = shared_[l];
        auto a_fn = pop_tid_layer(23, l);
        if (a_fn.size() >= fn_n * sizeof(bf16_t)) {
            w.hc_attn_fn.resize(fn_n);
            memcpy(w.hc_attn_fn.data(), a_fn.data(), fn_n * sizeof(bf16_t));
            transpose_bf16(w.hc_attn_fn, HC_MIX_DIM, HC_DIM);  /* [24,16384]->[16384,24] */
        }
        auto a_sc = pop_tid_layer(24, l);
        if (a_sc.size() >= 3 * sizeof(float)) { w.hc_attn_scale.resize(3); memcpy(w.hc_attn_scale.data(), a_sc.data(), 3 * sizeof(float)); }
        auto a_bs = pop_tid_layer(25, l);
        if (a_bs.size() >= HC_MIX_DIM * sizeof(float)) { w.hc_attn_base.resize(HC_MIX_DIM); memcpy(w.hc_attn_base.data(), a_bs.data(), HC_MIX_DIM * sizeof(float)); }
        auto f_fn = pop_tid_layer(26, l);
        if (f_fn.size() >= fn_n * sizeof(bf16_t)) {
            w.hc_ffn_fn.resize(fn_n);
            memcpy(w.hc_ffn_fn.data(), f_fn.data(), fn_n * sizeof(bf16_t));
            transpose_bf16(w.hc_ffn_fn, HC_MIX_DIM, HC_DIM);
        }
        auto f_sc = pop_tid_layer(27, l);
        if (f_sc.size() >= 3 * sizeof(float)) { w.hc_ffn_scale.resize(3); memcpy(w.hc_ffn_scale.data(), f_sc.data(), 3 * sizeof(float)); }
        auto f_bs = pop_tid_layer(28, l);
        if (f_bs.size() >= HC_MIX_DIM * sizeof(float)) { w.hc_ffn_base.resize(HC_MIX_DIM); memcpy(w.hc_ffn_base.data(), f_bs.data(), HC_MIX_DIM * sizeof(float)); }
    }
    /* Global HC output-collapse. */
    {
        const size_t hfn_n = (size_t)HC_DIM * N_HC;
        auto hfn = pop_tid(29);
        if (hfn.size() >= hfn_n * sizeof(bf16_t)) {
            model_.hc_head_fn.resize(hfn_n);
            memcpy(model_.hc_head_fn.data(), hfn.data(), hfn_n * sizeof(bf16_t));
            transpose_bf16(model_.hc_head_fn, N_HC, HC_DIM);  /* [4,16384]->[16384,4] */
        }
        auto hsc = pop_tid(30);
        if (hsc.size() >= sizeof(float)) { model_.hc_head_scale.resize(1); memcpy(model_.hc_head_scale.data(), hsc.data(), sizeof(float)); }
        auto hbs = pop_tid(31);
        if (hbs.size() >= N_HC * sizeof(float)) { model_.hc_head_base.resize(N_HC); memcpy(model_.hc_head_base.data(), hbs.data(), N_HC * sizeof(float)); }
    }
    ::close(fd);

    /* Sanity: count layers that actually got HC weights. */
    int n_hc = 0;
    for (int l = 0; l < n_layers; l++) if (!shared_[l].hc_attn_fn.empty()) n_hc++;
    fprintf(stderr, "[npu] HC loaded: %d/%d layers, head_fn=%zu\n",
            n_hc, n_layers, model_.hc_head_fn.size());
}

void FSTEngine::load_norm_override(const std::string& norm_path) {
    /* Read ONLY the norm TIDs (1,3,4,14,15,22) from a <model>.fst.norm sidecar
     * and OVERRIDE the values loaded from the main .fst.  The main .fst was
     * written by an early converter that baked in ~40x-too-small norm weights
     * (attn_norm ~0.03 instead of ~1.24, moe_norm ~0.02 instead of ~1.61),
     * which starves the FFN sublayer (ffn_out ≈ 0 → garbage output).  Reading
     * by TID is unambiguous and bypasses the dim-key "4096x1x1" collision that
     * swapped attn_norm/moe_norm in load_shared_weights.
     *
     * Storage: norm weights are BF16 [hidden_dim]; attn_sinks are F32 [n_heads].
     * q_norm is [1024], kv_norm is [512]; the rest are [4096].  final_norm
     * (TID 1) is global, stored in model_.final_norm (malloc'd BF16). */
    int fd = ::open(norm_path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[npu] norm sidecar open failed: %s\n", norm_path.c_str()); return; }
    auto hb = pread(fd, 128, 0); auto* h = (const FSTH*)hb.data();
    size_t ne = (size_t)h->sdc; auto db = pread(fd, ne * 64, (off_t)h->sdo);

    auto pop_tid_layer = [&](uint32_t target_tid, int lid) -> std::vector<uint8_t> {
        for (size_t i = 0; i < ne; i++) { auto* e = (const FSTE*)(db.data() + i * 64);
            if (e->tid != target_tid || (int)e->lid != lid) continue;
            return pread(fd, (size_t)e->sz, (off_t)e->off); }
        return {}; };
    auto pop_tid = [&](uint32_t target_tid) -> std::vector<uint8_t> {
        for (size_t i = 0; i < ne; i++) { auto* e = (const FSTE*)(db.data() + i * 64);
            if (e->tid != target_tid) continue;
            return pread(fd, (size_t)e->sz, (off_t)e->off); }
        return {}; };

    const int D = config_.hidden_dim;
    const int n_layers = config_.n_layers;
    int n_overridden = 0;
    for (int l = 0; l < n_layers; l++) {
        auto& w = shared_[l];
        auto load_bf16_vec = [&](uint32_t tid, std::vector<bf16_t, AlignedAllocator<bf16_t>>& dst, int expect_n) -> bool {
            auto raw = pop_tid_layer(tid, l);
            const size_t need = (size_t)expect_n * sizeof(bf16_t);
            if (raw.size() < need) return false;
            dst.resize(expect_n);
            memcpy(dst.data(), raw.data(), need);
            return true;
        };
        bool a = load_bf16_vec(3,  w.attn_norm, D);          /* TID 3  input norm   [4096] */
        bool m = load_bf16_vec(4,  w.moe_norm,  D);          /* TID 4  post-attn norm[4096] */
        bool q = load_bf16_vec(14, w.q_norm,    1024);       /* TID 14 q norm        [1024] */
        bool k = load_bf16_vec(15, w.kv_norm,   512);        /* TID 15 kv norm       [512]  */
        /* TID 22 attn_sink: F32 [n_heads]. */
        auto sink_raw = pop_tid_layer(22, l);
        const size_t sink_need = (size_t)MLA_N_HEADS * sizeof(float);
        if (sink_raw.size() >= sink_need) {
            w.attn_sinks.resize(MLA_N_HEADS);
            memcpy(w.attn_sinks.data(), sink_raw.data(), sink_need);
        }
        if (a && m) n_overridden++;
    }
    /* Global final norm (TID 1): BF16 [hidden_dim], replaces model_.final_norm. */
    auto fn_raw = pop_tid(1);
    if (fn_raw.size() >= (size_t)D * sizeof(bf16_t)) {
        if (model_.final_norm) { free(model_.final_norm); model_.final_norm = nullptr; }
        model_.final_norm_bytes = (size_t)D * sizeof(bf16_t);
        model_.final_norm = (bf16_t*)malloc(model_.final_norm_bytes);
        memcpy(model_.final_norm, fn_raw.data(), model_.final_norm_bytes);
    }
    ::close(fd);

    /* Sanity report: print a sample norm magnitude so the override is
     * verifiable at load time (good moe_norm ≈ 1.6, bad ≈ 0.03). */
    auto mean_abs = [](const std::vector<bf16_t, AlignedAllocator<bf16_t>>& v) -> float {
        if (v.empty()) return 0.0f;
        double s = 0.0;
        for (auto x : v) s += std::fabs(bf16f(x));
        return (float)(s / v.size());
    };
    if (!shared_.empty()) {
        fprintf(stderr, "[npu] norm override: %d/%d layers, "
                "L0 attn_norm|ma=%.4f moe_norm|ma=%.4f q_norm|ma=%.4f kv_norm|ma=%.4f "
                "final_norm|ma=%.4f\n",
                n_overridden, n_layers,
                mean_abs(shared_[0].attn_norm), mean_abs(shared_[0].moe_norm),
                mean_abs(shared_[0].q_norm), mean_abs(shared_[0].kv_norm),
                model_.final_norm ? mean_abs(std::vector<bf16_t, AlignedAllocator<bf16_t>>(
                    model_.final_norm, model_.final_norm + D)) : 0.0f);
        /* Sink sanity: print L0 attn_sinks[0..7] so NaN/huge sinks are visible
         * at load time (softmax NaN originates here if these are non-finite). */
        if (!shared_[0].attn_sinks.empty()) {
            fprintf(stderr, "[npu] L0 attn_sinks[0:8]:");
            for (int i = 0; i < 8 && i < (int)shared_[0].attn_sinks.size(); i++)
                fprintf(stderr, " %g", shared_[0].attn_sinks[i]);
            int nan = 0;
            for (auto sv : shared_[0].attn_sinks) if (!std::isfinite(sv)) nan++;
            fprintf(stderr, "  (n=%zu nan=%d)\n", shared_[0].attn_sinks.size(), nan);
        } else {
            fprintf(stderr, "[npu] L0 attn_sinks: EMPTY (sink=0 fallback)\n");
        }
    }
}

void FSTEngine::build_rope_lut(bf16_t* lut, int M, int lid, bool inverse){
    const int rope_dim=config_.rope_dim;
    const float sin_sign = inverse ? -1.0f : 1.0f;
    /* V4 Flash: compress_ratio per layer (config_.compress_ratios, 43 entries).
     * ratio==0 layers (L0,L1) use rope_freq_base_=10000 with plain extrapolation;
     * ratio!=0 layers (L2..L42, ratios 4/128) use compress_rope_freq_base=160000
     * + YaRN.  Verified vs HF config.json (compress_ratios[0..42]) and HF safetensors
     * (L40/L41/L42 all carry compressor.* weights → they ARE compressed).  model.py
     * Attention.__init__ keys on args.compress_ratios[layer_id] (model.py 484-487). */
    const bool compressed = (lid < (int)config_.compress_ratios.size())
                            ? (config_.compress_ratios[lid] != 0)
                            : (lid >= 2);
    const float freq_base = compressed ? 160000.0f : rope_freq_base_;
    const float theta_scale = std::pow(freq_base, -2.0f / (float)rope_dim);
    if(!compressed){
        for(int m=0;m<M;m++){
            float theta_extrap=(float)(seq_pos_+m);
            for(int d=0;d<rope_dim;d+=2){
                lut[m*rope_dim+d]    =f2bf(cosf(theta_extrap));
                lut[m*rope_dim+d+1]  =f2bf(sin_sign*sinf(theta_extrap));
                theta_extrap*=theta_scale;
            }
        }
        return;
    }
    /* YaRN (compressed layers).  freq_scale=1/16, ext_factor=1, n_ctx_orig=65536,
     * beta_fast=32, beta_slow=1 (ds4.c defaults).  mscale cancels to 1.0, so the
     * LUT entries are just cos/sin of the ramp-mixed theta. */
    const float freq_scale=1.0f/16.0f;
    const float ext_factor=1.0f;
    const float n_ctx_orig=65536.0f;
    const float beta_fast=32.0f, beta_slow=1.0f;
    const float n_rot=(float)rope_dim;
    auto corr_dim=[&](float beta){
        return (float)rope_dim * logf(n_ctx_orig/(beta*2.0f*(float)M_PI)) / (2.0f*logf(freq_base));
    };
    float corr0=std::max(0.0f, floorf(corr_dim(beta_fast)));
    float corr1=std::min((float)(rope_dim-1), ceilf(corr_dim(beta_slow)));
    auto ramp=[&](int i0){
        float y=((float)(i0/2)-corr0)/std::max(0.001f, corr1-corr0);
        return 1.0f - std::min(1.0f, std::max(0.0f, y));
    };
    for(int m=0;m<M;m++){
        float theta_extrap=(float)(seq_pos_+m);
        for(int d=0;d<rope_dim;d+=2){
            float ramp_mix=ramp(d)*ext_factor;
            float theta_interp=freq_scale*theta_extrap;
            float theta=theta_interp*(1.0f-ramp_mix)+theta_extrap*ramp_mix;
            lut[m*rope_dim+d]    =f2bf(cosf(theta));
            lut[m*rope_dim+d+1]  =f2bf(sin_sign*sinf(theta));
            theta_extrap*=theta_scale;
        }
    }
}

void FSTEngine::rope_qpe_all_heads(bf16_t* heads, int M, int lid, bool inverse) {
    /* Apply RoPE to the rope_dim tail of ALL n_heads (ds4.c model.py:505 does
     * `apply_rotary_emb(q[..., -rd:], freqs_cis)` on q shaped [M, n_heads,
     * head_dim], rotating every head's tail).  The prior head-0-only code
     * gathered just offset m*n_heads*head_dim+nope_dim (head 0's tail), leaving
     * heads 1..63 un-RoPE'd.  RoPE is norm-preserving per (x0,x1) pair, so the
     * un-RoPE'd heads kept the right MAGNITUDE but the wrong DIRECTION ->
     * orthogonal residual / incoherent tokens.  `heads` is row-major
     * [M*n_heads, head_dim]; row r=m*n_heads+h, tail at r*head_dim+nope_dim. */
    const int rope_dim = config_.rope_dim;
    const int head_dim = MLA_HEAD_DIM;
    const int n_heads = MLA_N_HEADS;
    const int nope_dim = head_dim - rope_dim;
    const int rows = M * n_heads;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> pe((size_t)rows * rope_dim);
    for (int r = 0; r < rows; r++)
        memcpy(pe.data() + (size_t)r * rope_dim,
               heads + (size_t)r * head_dim + nope_dim,
               rope_dim * sizeof(bf16_t));
    /* Per-position LUT [M, rope_dim] (position seq_pos_+m), broadcast across
     * heads: row (m*n_heads+h) reuses position m's freqs. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> lut_m((size_t)M * rope_dim);
    build_rope_lut(lut_m.data(), M, lid, inverse);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> lut((size_t)rows * rope_dim);
    for (int m = 0; m < M; m++)
        for (int hh = 0; hh < n_heads; hh++)
            memcpy(lut.data() + ((size_t)m * n_heads + hh) * rope_dim,
                   lut_m.data() + (size_t)m * rope_dim,
                   rope_dim * sizeof(bf16_t));
    npu_rope("rope", pe.data(), lut.data(), pe.data(), rows, rope_dim);
    for (int r = 0; r < rows; r++)
        memcpy(heads + (size_t)r * head_dim + nope_dim,
               pe.data() + (size_t)r * rope_dim, rope_dim * sizeof(bf16_t));
}

void FSTEngine::apply_rope(bf16_t* q,bf16_t* k,int M,int S){
    const int dim_per_head=MLA_KV_DECOMP/MLA_N_HEADS;
    const float inv_freq=1.0f/std::pow(rope_freq_base_,2.0f/dim_per_head);
    for(int head=0;head<MLA_N_HEADS;head++){
        for(int d=0;d<dim_per_head;d+=2){
            float theta=inv_freq*(float)d;
            for(int m=0;m<M;m++){
                float angle=(float)(seq_pos_+m)*theta;
                float c=cosf(angle),s=sinf(angle);
                int idx=head*dim_per_head+d;
                float q0=bf16f(q[m*MLA_KV_DECOMP+idx]);float q1=bf16f(q[m*MLA_KV_DECOMP+idx+1]);
                q[m*MLA_KV_DECOMP+idx]=f2bf(q0*c-q1*s);
                q[m*MLA_KV_DECOMP+idx+1]=f2bf(q0*s+q1*c);
            }
            for(int s_pos=0;s_pos<S;s_pos++){
                float angle=(float)s_pos*theta;
                float c=cosf(angle),s=sinf(angle);
                int idx=head*dim_per_head+d;
                float k0=bf16f(k[s_pos*MLA_KV_DECOMP+idx]);float k1=bf16f(k[s_pos*MLA_KV_DECOMP+idx+1]);
                k[s_pos*MLA_KV_DECOMP+idx]=f2bf(k0*c-k1*s);
                k[s_pos*MLA_KV_DECOMP+idx+1]=f2bf(k0*s+k1*c);
            }
        }
    }
}

void FSTEngine::apply_rope_pe(bf16_t* pe,int npos,int pos_start){
    const int rope_dim=config_.rope_dim;
    const float inv_freq=1.0f/std::pow(rope_freq_base_,2.0f/rope_dim);
    for(int head=0;head<MLA_N_HEADS;head++){
        for(int d=0;d<rope_dim;d+=2){
            float theta=inv_freq*(float)d;
            for(int i=0;i<npos;i++){
                float angle=(float)(pos_start+i)*theta;
                float c=cosf(angle),s=sinf(angle);
                int idx=i*MLA_N_HEADS*rope_dim+head*rope_dim+d;
                float v0=bf16f(pe[idx]);float v1=bf16f(pe[idx+1]);
                pe[idx]=f2bf(v0*c-v1*s);
                pe[idx+1]=f2bf(v0*s+v1*c);
            }
        }
    }
}

std::vector<bf16_t, AlignedAllocator<bf16_t>> FSTEngine::forward_embeddings(const std::vector<int>& token_ids){
    const int D=config_.hidden_dim;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> out(token_ids.size()*D);
    for(size_t t=0;t<token_ids.size();t++){
        int tid=token_ids[t];
        assert(tid >= 0 && tid < config_.vocab_size);
        memcpy(out.data()+t*D,host_embedding_table_.data()+(size_t)tid*D,D*sizeof(bf16_t));}
    {float mx=-1e30f,mn=1e30f; for(size_t i=0;i<out.size();i++){float v=bf16f(out[i]); if(std::isnan(v))continue; if(v>mx)mx=v; if(v<mn)mn=v;} fprintf(stderr,"[emb] forward_embeddings: %zu tok, out mn=%.4f mx=%.4f [0:4]=",token_ids.size(),mn,mx); for(int i=0;i<4;i++)fprintf(stderr,"%.4f ",bf16f(out[i])); fprintf(stderr," src[0:4]="); for(int i=0;i<4;i++)fprintf(stderr,"%.4f ",bf16f(host_embedding_table_[(size_t)token_ids[0]*D+i])); fprintf(stderr,"\n"); fflush(stderr);}
    return out;
}

void FSTEngine::apply_final_norm_and_lm_head(bf16_t* hs,int M,float* logits){
    const int D=config_.hidden_dim, V=config_.vocab_size;
    const int M_PAD = ((M + 15) / 16) * 16;   // M=16 standardization
    const int N_PAD = lm_head_n_pad_;

    /* hs is the 4-stream HC residual [M, N_HC, D].  The 4 streams MUST be
     * collapsed to a plain [M, D] vector via output_hc_head BEFORE final_norm
     * + lm_head — exactly what ds4.c output_hc_head_one does.  Using only the
     * first stream (treating hs as plain [M, D]) produces garbage logits: the
     * greedy argmax lands on a high-ID token (129012) because 3/4 of the
     * residual signal is dropped.  This is the real L42 "garbage" root cause
     * downstream of the (correct) O-proj. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> plain((size_t)M * D);
    output_hc_head(M, hs, plain.data());

    if (M == 1) {
        float mx=-1e30f,mn=1e30f; int nan=0;
        for(int i=0;i<D;i++){float v=bf16f(plain[i]);if(std::isnan(v)||std::isinf(v))nan++;if(v>mx&&v<1e30f)mx=v;if(v<mn&&v>-1e30f)mn=v;}
        fprintf(stderr,"[dbg] hidden(plain): min=%.4f max=%.4f nan=%d [0:5]: ",mn,mx,nan);
        for(int i=0;i<5;i++) fprintf(stderr,"%.4f ",bf16f(plain[i]));
        fprintf(stderr,"\n");
    }

    for (int m = 0; m < M; m++)
        npu_rmsnorm_weighted(plain.data() + m * D, plain.data() + m * D, model_.final_norm, D);

    /* Pad hidden to M_PAD rows, zero-fill extra rows */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hs_pad(M_PAD * D, 0);
    memcpy(hs_pad.data(), plain.data(), (size_t)M * D * sizeof(bf16_t));

    /* NPU LM head GEMM, tiled along N by npu_gemm (64 x 2048-col tiles).
     * bo_lm_head_ is already [D, N_PAD] row-major (transposed + zero-padded). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> bf16_out(M_PAD * N_PAD);
    bool lm_time = std::getenv("FST_LM_HEAD_TIME");
    double lm_t0 = lm_time ? now() : 0.0;
    npu_gemm("lm_head_gemm", bf16_out.data(), hs_pad.data(),
             bo_lm_head_.map<bf16_t*>(), M_PAD, N_PAD, D);
    if (lm_time)
        fprintf(stderr, "[lm-head-time] M=%d N_PAD=%d D=%d  lm_head=%.3f ms\n",
                M, N_PAD, D, (now() - lm_t0) * 1000.0);

    /* Slice: read only M rows, V columns from M_PAD x N_PAD output */
    for (int m = 0; m < M; m++)
        for (int v = 0; v < V; v++)
            logits[m * V + v] = bf16f(bf16_out[m * N_PAD + v]);

    /* ── CPU-vs-NPU diagnostic (FST_AUDIT, M==1 only) ──────────────────────
     * Recompute the ENTIRE final-logits path from the SAME 4-stream HC
     * residual `hs` on the CPU, using the raw BF16 weights, and compare the
     * argmax against the NPU-computed `logits`.  This decisively localizes the
     * garbage-argmax bug:
     *   - MATCH  (both garbage) -> bug is UPSTREAM in `hs` (HC residual
     *     direction drift via hc_pre/hc_post/process_layer).
     *   - DIFFER -> bug is in the NPU final-logits pipeline (output_hc_head's
     *     npu_gemm_hc matvec, npu_rmsnorm_weighted, or the lm_head npu_gemm).  */
    if (M == 1 && std::getenv("FST_AUDIT") && !model_.hc_head_fn.empty()) {
        const int hd = D;
        std::vector<float> flat(HC_DIM);
        double ss = 0;
        for (int i = 0; i < HC_DIM; i++) { float v = bf16f(hs[i]); flat[i] = v; ss += (double)v * v; }
        float inv = 1.0f / sqrtf((float)(ss / (double)HC_DIM) + 1e-6f);
        for (int i = 0; i < HC_DIM; i++) flat[i] *= inv;
        float pre[4] = {0, 0, 0, 0};
        for (int h = 0; h < N_HC; h++) {
            double acc = 0;
            for (int i = 0; i < HC_DIM; i++) acc += (double)flat[i] * bf16f(model_.hc_head_fn[(size_t)i * N_HC + h]);
            pre[h] = (float)acc;
        }
        float scale = model_.hc_head_scale.empty() ? 1.0f : model_.hc_head_scale[0];
        const float* base = model_.hc_head_base.empty() ? nullptr : model_.hc_head_base.data();
        float w[4];
        for (int h = 0; h < N_HC; h++) {
            float z = pre[h] * scale + (base ? base[h] : 0.0f);
            w[h] = 1.0f / (1.0f + expf(-z)) + HC_EPS;
        }
        std::vector<float> embd(hd);
        for (int d = 0; d < hd; d++) {
            float acc = 0;
            for (int h = 0; h < N_HC; h++) acc += w[h] * bf16f(hs[(size_t)h * hd + d]);
            embd[d] = acc;
        }
        std::vector<float> norm(hd);
        double sn = 0;
        for (int d = 0; d < hd; d++) sn += (double)embd[d] * embd[d];
        float inorm = 1.0f / sqrtf((float)(sn / (double)hd) + 1e-6f);
        for (int d = 0; d < hd; d++) norm[d] = embd[d] * inorm * bf16f(model_.final_norm[d]);
        int cpu_argmax = 0; float cpu_max = -1e30f;
        for (int v = 0; v < V; v++) {
            double acc = 0;
            const bf16_t* row = host_lm_head_.data() + (size_t)v * D;
            for (int d = 0; d < hd; d++) acc += (double)norm[d] * bf16f(row[d]);
            float l = (float)acc;
            if (l > cpu_max) { cpu_max = l; cpu_argmax = v; }
        }
        int npu_argmax = 0; float npu_max = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > npu_max) { npu_max = logits[v]; npu_argmax = v; }
        fprintf(stderr, "[CPU-vs-NPU] cpu_argmax=%d (%.3f)  npu_argmax=%d (%.3f)  %s\n",
                cpu_argmax, cpu_max, npu_argmax, npu_max,
                (cpu_argmax == npu_argmax) ? "MATCH(upstream-bug)" : "DIFFER(NPU-pipeline-bug)");
        fflush(stderr);

        /* ── Per-step localization ───────────────────────────────────────
         * (a) NPU normed hidden (plain[], in-place after npu_rmsnorm_weighted)
         *     vs CPU `norm`.  Big diff => bug in output_hc_head or rmsnorm.
         * (b) Feed the EXACT NPU normed plain (plain[], already bf16) through
         *     a CPU lm_head matvec, and compare its argmax to the NPU logits
         *     argmax (which used the SAME plain[] as input).  Same bf16 input
         *     on both sides -> any diff is purely the NPU lm_head GEMM.  */
        double md_plain = 0, mx_plain = 0; int np_plain = 0;
        for (int d = 0; d < hd; d++) {
            float a = bf16f(plain[d]), b = norm[d];
            double dd = (double)a - (double)b;
            if (dd < 0) dd = -dd;
            md_plain += dd; if (dd > mx_plain) mx_plain = dd; np_plain++;
        }
        fprintf(stderr, "[step-a] NPU-normed-plain vs CPU-norm: md=%.4f mx=%.4f (over %d)\n",
                md_plain / np_plain, mx_plain, np_plain);
        /* (b) CPU lm_head matvec fed with the NPU normed plain (bf16->float) */
        std::vector<float> npu_norm_f(hd);
        for (int d = 0; d < hd; d++) npu_norm_f[d] = bf16f(plain[d]);
        int cpu_lm_argmax = 0; float cpu_lm_max = -1e30f;
        for (int v = 0; v < V; v++) {
            double acc = 0;
            const bf16_t* row = host_lm_head_.data() + (size_t)v * D;
            for (int d = 0; d < hd; d++) acc += (double)npu_norm_f[d] * bf16f(row[d]);
            float l = (float)acc;
            if (l > cpu_lm_max) { cpu_lm_max = l; cpu_lm_argmax = v; }
        }
        fprintf(stderr, "[step-b] CPU-lm_head(NPU-plain) argmax=%d (%.3f)  vs NPU-lm_head argmax=%d (%.3f) -> %s\n",
                cpu_lm_argmax, cpu_lm_max, npu_argmax, npu_max,
                (cpu_lm_argmax == npu_argmax) ? "MATCH(lm_head-ok)" : "DIFFER(lm_head-broken)");
        fflush(stderr);

        /* (c) single-tile fast-path N=2048: copy bo_lm_head_[:, 0:2048] into a
         *   host vector and call the kernel ONCE (N<=2048 -> single-tile path,
         *   the same path router_gemm uses successfully).  Compare to the CPU
         *   reference restricted to the first 2048 columns.
         *   MATCH => the N=2048 kernel is fine, bug is the multi-tile host path.
         *   DIFFER => the N=2048 kernel specialization itself is broken.  */
        std::vector<bf16_t, AlignedAllocator<bf16_t>> b_first((size_t)D * 2048);
        const bf16_t* lm_map = bo_lm_head_.map<bf16_t*>();
        for (int k = 0; k < D; k++)
            memcpy(b_first.data() + (size_t)k * 2048, lm_map + (size_t)k * N_PAD, 2048 * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> c_fast((size_t)M_PAD * 2048, 0);
        npu_gemm("lm_head_gemm", c_fast.data(), hs_pad.data(), b_first.data(), M_PAD, 2048, D);
        int fast_argmax = 0; float fast_max = bf16f(c_fast[0]);
        for (int v = 1; v < 2048; v++) if (bf16f(c_fast[v]) > fast_max) { fast_max = bf16f(c_fast[v]); fast_argmax = v; }
        int cpu_argmax_2048 = 0; float cpu_max_2048 = -1e30f;
        for (int v = 0; v < 2048; v++) {
            double acc = 0;
            const bf16_t* row = host_lm_head_.data() + (size_t)v * D;
            for (int d = 0; d < hd; d++) acc += (double)npu_norm_f[d] * bf16f(row[d]);
            float l = (float)acc;
            if (l > cpu_max_2048) { cpu_max_2048 = l; cpu_argmax_2048 = v; }
        }
        fprintf(stderr, "[step-c] single-tile N=2048: fast_argmax=%d (%.3f) vs cpu_argmax_2048=%d (%.3f) -> %s\n",
                fast_argmax, fast_max, cpu_argmax_2048, cpu_max_2048,
                (fast_argmax == cpu_argmax_2048) ? "MATCH(kernel-ok->multi-tile-bug)" : "DIFFER(kernel-broken)");
        fflush(stderr);
    }
}

// ── Hybrid Connection (HC) 4-stream residual ─────────────────────────────
// Mirrors ds4.c hc_split_sinkhorn_one / hc_pre_from_state_one / hc_post_one /
// output_hc_head_one.  The HC fn matvec ([HC_DIM, HC_MIX_DIM] @ [HC_DIM]) is a
// real GEMM and runs on the NPU via the proven kvc kernel (8-row M-chunks, the
// only shape that kernel accepts).  The rms_norm_no_weight over HC_DIM=16384,
// the 24-element Sinkhorn split, the 4-stream weighted sum, and hc_post are
// small control/elementwise ops that the reference also runs on host; a fused
// NPU HC kernel is deferred to Phase 4 (the rmsnorm needs a 16384-wide kernel,
// which the current 4096-wide EW kernel cannot tile without a new compile).
static void hc_split_sinkhorn(float* split, const float* mix, const float* scale,
                               const float* base, int n_hc, int iters, float eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < n_hc; i++) {
        float z = mix[i] * pre_scale + base[i];
        split[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    for (int i = 0; i < n_hc; i++) {
        int off = n_hc + i;
        float z = mix[off] * post_scale + base[off];
        split[off] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = -1e30f;
        for (int src = 0; src < n_hc; src++) {
            int idx = src + dst * n_hc;
            float v = mix[2 * n_hc + idx] * comb_scale + base[2 * n_hc + idx];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            int idx = src + dst * n_hc;
            c[idx] = expf(c[idx] - row_max);
            row_sum += c[idx];
        }
        float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++)
            c[src + dst * n_hc] = c[src + dst * n_hc] * inv + eps;
    }
    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
        float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }
    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];
            float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
            float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }
    for (int i = 0; i < n_hc * n_hc; i++) split[2 * n_hc + i] = c[i];
}

/* rms_norm_no_weight over HC_DIM (one RMS denominator across all 4 streams).
 * f32 scratch; writes f32 normalized flat. */
static inline void hc_rmsnorm_no_weight(float* out, const bf16_t* x, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) { float v = bf16f(x[i]); out[i] = v; ss += (double)v * v; }
    float inv_rms = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] *= inv_rms;
}

void FSTEngine::hc_pre(int M, const bf16_t* residual_hc, const bf16_t* hc_fn,
                       const float* hc_scale, const float* hc_base,
                       bf16_t* cur, float* post, float* comb) {
    const int hd = config_.hidden_dim;
    /* flat[M, HC_DIM] = rms_norm_no_weight(residual_hc[m]) (host; 16384-wide). */
    std::vector<float, AlignedAllocator<float>> flat((size_t)M * HC_DIM);
    for (int m = 0; m < M; m++)
        hc_rmsnorm_no_weight(flat.data() + (size_t)m * HC_DIM,
                             residual_hc + (size_t)m * N_HC * hd, HC_DIM, HC_EPS);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> flat_bf((size_t)M * HC_DIM);
    for (size_t i = 0; i < (size_t)M * HC_DIM; i++) flat_bf[i] = f2bf(flat[i]);

    /* mix[M, HC_MIX_DIM] = flat @ hc_fn  on the kvc kernel (8-row M-chunks). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> mix_bf((size_t)M * HC_MIX_DIM, 0);
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        std::vector<bf16_t, AlignedAllocator<bf16_t>> a_chunk(8 * HC_DIM, 0);
        memcpy(a_chunk.data(), flat_bf.data() + (size_t)m0 * HC_DIM,
               (size_t)mc * HC_DIM * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> c_chunk(8 * HC_MIX_DIM, 0);
        npu_gemm_hc(c_chunk.data(), a_chunk.data(), hc_fn, 8, HC_MIX_DIM, HC_DIM);
        memcpy(mix_bf.data() + (size_t)m0 * HC_MIX_DIM, c_chunk.data(),
               (size_t)mc * HC_MIX_DIM * sizeof(bf16_t));
    }

    /* Per-token Sinkhorn split -> pre/post/comb; cur = weighted_sum(streams, pre). */
    for (int m = 0; m < M; m++) {
        float mix_f[HC_MIX_DIM];
        for (int i = 0; i < HC_MIX_DIM; i++) mix_f[i] = bf16f(mix_bf[(size_t)m * HC_MIX_DIM + i]);
        float split[HC_MIX_DIM];
        hc_split_sinkhorn(split, mix_f, hc_scale, hc_base, N_HC, HC_SINKHORN_ITER, HC_EPS);
        const float* pre = split;
        memcpy(post + (size_t)m * N_HC, split + N_HC, N_HC * sizeof(float));
        memcpy(comb + (size_t)m * N_HC * N_HC, split + 2 * N_HC, N_HC * N_HC * sizeof(float));
        const bf16_t* r = residual_hc + (size_t)m * N_HC * hd;
        bf16_t* o = cur + (size_t)m * hd;
        for (int d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (int h = 0; h < N_HC; h++) acc += pre[h] * bf16f(r[(size_t)h * hd + d]);
            o[d] = f2bf(acc);
        }
    }
}

void FSTEngine::hc_post(int M, const bf16_t* block_out, const bf16_t* residual_hc,
                        const float* post, const float* comb, bf16_t* out_hc) {
    const int hd = config_.hidden_dim;
    for (int m = 0; m < M; m++) {
        const bf16_t* bo = block_out + (size_t)m * hd;
        const bf16_t* r = residual_hc + (size_t)m * N_HC * hd;
        const float* p = post + (size_t)m * N_HC;
        const float* c = comb + (size_t)m * N_HC * N_HC;
        for (int dst = 0; dst < N_HC; dst++) {
            float gp = p[dst];
            bf16_t* o = out_hc + (size_t)m * N_HC * hd + (size_t)dst * hd;
            for (int d = 0; d < hd; d++) {
                float acc = bf16f(bo[d]) * gp;
                for (int src = 0; src < N_HC; src++)
                    acc += c[dst + src * N_HC] * bf16f(r[(size_t)src * hd + d]);
                o[d] = f2bf(acc);
            }
        }
    }
}

void FSTEngine::output_hc_head(int M, const bf16_t* residual_hc, bf16_t* out_plain) {
    const int hd = config_.hidden_dim;
    if (model_.hc_head_fn.empty()) {
        for (int m = 0; m < M; m++)
            memcpy(out_plain + (size_t)m * hd, residual_hc + (size_t)m * N_HC * hd,
                   (size_t)hd * sizeof(bf16_t));
        return;
    }
    std::vector<float, AlignedAllocator<float>> flat((size_t)M * HC_DIM);
    for (int m = 0; m < M; m++)
        hc_rmsnorm_no_weight(flat.data() + (size_t)m * HC_DIM,
                             residual_hc + (size_t)m * N_HC * hd, HC_DIM, HC_EPS);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> flat_bf((size_t)M * HC_DIM);
    for (size_t i = 0; i < (size_t)M * HC_DIM; i++) flat_bf[i] = f2bf(flat[i]);

    /* pre[M, N_HC] = flat @ hc_head_fn[HC_DIM, N_HC] on kvc (8-row chunks). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> pre_bf((size_t)M * N_HC, 0);
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        std::vector<bf16_t, AlignedAllocator<bf16_t>> a_chunk(8 * HC_DIM, 0);
        memcpy(a_chunk.data(), flat_bf.data() + (size_t)m0 * HC_DIM,
               (size_t)mc * HC_DIM * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> c_chunk(8 * N_HC, 0);
        npu_gemm_hc(c_chunk.data(), a_chunk.data(), model_.hc_head_fn.data(), 8, N_HC, HC_DIM);
        memcpy(pre_bf.data() + (size_t)m0 * N_HC, c_chunk.data(),
               (size_t)mc * N_HC * sizeof(bf16_t));
    }
    const float scale = model_.hc_head_scale.empty() ? 1.0f : model_.hc_head_scale[0];
    const float* base = model_.hc_head_base.empty() ? nullptr : model_.hc_head_base.data();
    for (int m = 0; m < M; m++) {
        float w[N_HC];
        for (int h = 0; h < N_HC; h++) {
            float z = bf16f(pre_bf[(size_t)m * N_HC + h]) * scale + (base ? base[h] : 0.0f);
            w[h] = 1.0f / (1.0f + expf(-z)) + HC_EPS;
        }
        const bf16_t* r = residual_hc + (size_t)m * N_HC * hd;
        bf16_t* o = out_plain + (size_t)m * hd;
        for (int d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (int h = 0; h < N_HC; h++) acc += w[h] * bf16f(r[(size_t)h * hd + d]);
            o[d] = f2bf(acc);
        }
    }
}

/* Capture the HC-mean of the 4-stream residual h[M, N_HC, hd] for one of the
 * DSpark target layers (the last 3 main layers, 40/41/42).  HF:
 * main_hiddens.append(h.mean(dim=2))  -> [M, hd]; the draft's main_hidden is
 * the concat of the 3 target layers' HC-means -> [M, 3*hd=12288].  Stored in
 * target_hc_[lid - (n_layers-3)] = [M, hd]; the SD loop selects the accepted
 * token's row and concats.  NPU math is not needed — this is a reduction over
 * the already-computed HC residual (not a model op); doing it host-side keeps
 * it out of the dispatch-count budget. */
void FSTEngine::capture_target_hc(int lid, const bf16_t* h, int M) {
    const int n_layers = config_.n_layers;
    const int hd = config_.hidden_dim;
    int idx = lid - (n_layers - 3);
    if (idx < 0 || idx >= 3) return;
    auto& dst = target_hc_[idx];
    dst.assign((size_t)M * hd, f2bf(0.0f));
    for (int m = 0; m < M; m++) {
        const bf16_t* r = h + (size_t)m * N_HC * hd;
        bf16_t* o = dst.data() + (size_t)m * hd;
        for (int d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (int s = 0; s < N_HC; s++) acc += bf16f(r[(size_t)s * hd + d]);
            o[d] = f2bf(acc / (float)N_HC);
        }
    }
}

/* main_x = main_norm(main_proj(main_hidden)) for the DSpark draft (HF
 * forward_embed).  main_hidden is [count, 12288] (concat of the 3 target
 * layers' HC-means); main_x is [count, hd].  main_proj is [N=hd, K=12288]
 * native b_col_maj (loaded stage-0-only); N-tiled by the qck MLA kernel.
 * main_norm (RMSNorm) is applied per row.  M padded to 16 (AIE2P SIMD). */
std::vector<bf16_t, AlignedAllocator<bf16_t>>
FSTEngine::compute_draft_main_x(int count, const bf16_t* main_hidden) {
    const int hd = draft_cfg_.hidden_dim;
    const int K = 3 * hd;          /* 12288 */
    const int M_PAD = ((count + 15) / 16) * 16;
    auto& w0 = draft_shared_[0];
    /* main_proj GEMM: C[M_PAD, hd] = A[M_PAD, K] @ B[hd, K] (bcol), N-tiled. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> a_pad((size_t)M_PAD * K, 0);
    memcpy(a_pad.data(), main_hidden, (size_t)count * K * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> proj((size_t)M_PAD * hd, 0);
    if (!w0.main_proj.empty()) {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> tile((size_t)M_PAD * 2048);
        for (int n_off = 0; n_off < hd; n_off += 2048) {
            npu_gemm_mla_vec("qck", tile.data(), a_pad.data(),
                             w0.main_proj.data() + (size_t)n_off * K,
                             M_PAD, 2048, K);
            for (int m = 0; m < M_PAD; m++)
                memcpy(proj.data() + (size_t)m * hd + n_off,
                       tile.data() + (size_t)m * 2048, 2048 * sizeof(bf16_t));
        }
    }
    /* main_norm per row (RMSNorm). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> out((size_t)count * hd);
    if (w0.main_norm.empty()) {
        memcpy(out.data(), proj.data(), (size_t)count * hd * sizeof(bf16_t));
    } else {
        for (int m = 0; m < count; m++)
            npu_rmsnorm_weighted(out.data() + m * hd,
                                 proj.data() + m * hd, w0.main_norm.data(), hd);
    }
    return out;
}

/* Fill the DSpark sliding-window KV cache from main_x.  For each draft layer
 * l and each position p in [0, count): main_kv = kv_norm_l(wkv_l(main_x[p]))
 * with rotary @ (base_pos+p) applied to the last rope_dim, written to
 * draft_kv_cache_[l].kv_latent[(base_pos+p) % win].  HF: the window holds the
 * MAIN model's KV (from main_x), one slot per token; the per-block candidate
 * KV is recomputed each forward (not stored).  Filling the accepted positions
 * after each verify keeps the window gap-free across SD iterations (the anchor
 * alone would leave accepted-draft slots empty).  No attention is computed
 * here — this is pure KV staging. */
void FSTEngine::fill_draft_window(int count, const bf16_t* main_x, int base_pos) {
    const int hd = draft_cfg_.hidden_dim;
    const int rope_dim = MLA_ROPE_DIM;
    const int nope_dim = MLA_HEAD_DIM - rope_dim;     /* 448 */
    const int win = draft_cfg_.window_size;          /* 128 */
    const int M_PAD = ((count + 15) / 16) * 16;
    const float theta_scale = std::pow(rope_freq_base_, -2.0f / rope_dim);

    for (int l = 0; l < draft_cfg_.n_layers; l++) {
        auto& w = draft_shared_[l];
        auto& kvc = draft_kv_cache_[l];
        if (kvc.kv_latent.empty()) continue;
        /* wkv GEMM: main_kv_b[M_PAD, MLA_KV_LORA] = main_x[M_PAD, hd] @ wkv[KV,hd]. */
        std::vector<bf16_t, AlignedAllocator<bf16_t>> x_pad((size_t)M_PAD * hd, 0);
        memcpy(x_pad.data(), main_x, (size_t)count * hd * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> kvb((size_t)M_PAD * MLA_KV_LORA, 0);
        npu_gemm_mla_vec("qck", kvb.data(), x_pad.data(), w.wkv.data(),
                         M_PAD, MLA_KV_LORA, hd);
        for (int p = 0; p < count; p++) {
            std::vector<bf16_t, AlignedAllocator<bf16_t>> mkv(MLA_KV_LORA);
            if (!w.kv_norm.empty())
                npu_rmsnorm_weighted(mkv.data(), kvb.data() + (size_t)p * MLA_KV_LORA,
                                     w.kv_norm.data(), MLA_KV_LORA);
            else
                memcpy(mkv.data(), kvb.data() + (size_t)p * MLA_KV_LORA,
                       MLA_KV_LORA * sizeof(bf16_t));
            /* rotary @ (base_pos+p) on the rope half. */
            std::vector<bf16_t, AlignedAllocator<bf16_t>> pe(rope_dim);
            memcpy(pe.data(), mkv.data() + nope_dim, rope_dim * sizeof(bf16_t));
            std::vector<bf16_t, AlignedAllocator<bf16_t>> lut(rope_dim);
            float th = (float)(base_pos + p);
            for (int d = 0; d < rope_dim; d += 2) {
                lut[d] = f2bf(cosf(th)); lut[d + 1] = f2bf(sinf(th)); th *= theta_scale;
            }
            npu_rope("rope", pe.data(), lut.data(), pe.data(), 1, rope_dim);
            memcpy(mkv.data() + nope_dim, pe.data(), rope_dim * sizeof(bf16_t));
            int slot = (base_pos + p) % win;
            memcpy(kvc.kv_latent.data() + (size_t)slot * MLA_KV_LORA,
                   mkv.data(), MLA_KV_LORA * sizeof(bf16_t));
        }
    }
    flush_pending_runs();
}

/* process_layer: h is the 4-stream HC residual [M, N_HC, hd].  Each sublayer
 * is wrapped by hc_pre (streams -> sublayer input + post/comb gates) and
 * hc_post (gated block output + combined old streams -> new streams).  The
 * sublayers themselves (process_mla, process_expert_ffn, process_shared_expert)
 * still operate on a single [M, hd] stream.  Falls back to a plain single-
 * stream add when HC weights are absent (pre-HC .fst). */
void FSTEngine::process_layer(int lid, bf16_t* h, int M, const int* input_ids) {
    /* ARCH_HY3: plain-residual GQA block (attention + FFN).  Fully separate
     * from the DS4 4-stream HC path below — shared_[lid] is empty for HY3, so
     * branch out before any DS4 state is touched.  FFN (process_ffn_hy3) is a
     * no-op stub until phase 4; attention is wired + smoke-tested here. */
    if (config_.arch == ARCH_HY3) {
        process_gqa(lid, h, M);
        process_ffn_hy3(lid, h, M);
        return;
    }

    auto& w = shared_[lid];
    const int hd = config_.hidden_dim, tk = config_.top_k;
    const int hc_stride = N_HC * hd;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> an(M * hd), mn(M * hd), cur(M * hd);
    std::vector<float> post_a(M * N_HC), comb_a(M * N_HC * N_HC);
    std::vector<float> post_f(M * N_HC), comb_f(M * N_HC * N_HC);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> new_hc((size_t)M * hc_stride);
    const bool has_hc = !w.hc_attn_fn.empty();

    /* Per-layer HC recurrence decomposition audit (FST_AUDIT).  Localizes the
     * residual growth: for each hc_post it prints max|res|, max|block|, the max
     * post gate over tokens, the max comb ROW sum (application direction:
     * out[dst]=sum_src comb[dst,src]*res[src] -> row dst sum; >1 means amplifying),
     * and the realized max|post⊙block| and max|comb@res| vs max|out|. */
    const bool hc_audit = std::getenv("FST_AUDIT") &&
        (lid == 0 || lid == 1 || lid == 18 || lid == 19 || lid == 20 ||
         lid == 21 || lid == 22 || lid == 25 || lid == 26 || lid == 27 ||
         lid == 28 || lid == 29 || lid == 30);
    auto audit_recurrence = [&](const char* tag, const bf16_t* block_out,
                                const bf16_t* residual_hc, const float* post,
                                const float* comb, const bf16_t* out_hc) {
        if (!hc_audit) return;
        float mx_res=0, mx_block=0, mx_post=0, mx_combrow=0, mx_postblock=0, mx_combres=0, mx_out=0;
        int worst_m = 0, worst_block_m = 0; float worst_res = 0;
        for (int m = 0; m < M; m++) {
            const bf16_t* bo = block_out + (size_t)m * hd;
            const bf16_t* r = residual_hc + (size_t)m * N_HC * hd;
            const bf16_t* o = out_hc + (size_t)m * N_HC * hd;
            const float* p = post + (size_t)m * N_HC;
            const float* c = comb + (size_t)m * N_HC * N_HC;
            for (int h = 0; h < N_HC; h++) if (p[h] > mx_post) mx_post = p[h];
            for (int dst = 0; dst < N_HC; dst++) {
                float rsum = 0;
                for (int src = 0; src < N_HC; src++) rsum += c[dst + src * N_HC];
                if (rsum > mx_combrow) mx_combrow = rsum;
            }
            float m_block = 0, m_res = 0;
            for (int d = 0; d < hd; d++) {
                float b = std::fabs(bf16f(bo[d])); if (b > mx_block) mx_block = b;
                if (b > m_block) m_block = b;
                for (int dst = 0; dst < N_HC; dst++) {
                    float pb = std::fabs(p[dst] * bf16f(bo[d]));
                    float cr = 0; for (int src = 0; src < N_HC; src++) cr += c[dst + src*N_HC] * bf16f(r[src*hd+d]);
                    float ov = bf16f(o[dst*hd+d]);
                    if (pb > mx_postblock) mx_postblock = pb;
                    if (std::fabs(cr) > mx_combres) mx_combres = std::fabs(cr);
                    if (std::fabs(ov) > mx_out) mx_out = std::fabs(ov);
                }
                for (int h = 0; h < N_HC; h++) {
                    float rv = std::fabs(bf16f(r[h*hd+d]));
                    if (rv > mx_res) { mx_res = rv; worst_m = m; }
                    if (rv > m_res) m_res = rv;
                }
            }
            if (m_block > 0 && m_res > 0 && (m_block > 50.0f || m_res > 200.0f))
                fprintf(stderr, "[hc-rec]   L%d %s m=%d: |res|=%.2f |block|=%.3f\n",
                        lid, tag, m, m_res, m_block);
        }
        fprintf(stderr, "[hc-rec] L%d %s: |res|=%.2f |block|=%.3f postmax=%.4f combrow=%.4f "
                        "|post⊙b|=%.3f |comb@r|=%.3f |out|=%.3f (worst_res_m=%d worst_block_m=%d)\n",
                lid, tag, mx_res, mx_block, mx_post, mx_combrow,
                mx_postblock, mx_combres, mx_out, worst_m, worst_block_m);
        fflush(stderr);
    };

    if (lid == 0 || lid == 1 || lid % 10 == 9 || lid == 42) {
        float mx=-1e30f,mn_v=1e30f;
        for(int i=0;i<M*hc_stride;i++){float v=bf16f(h[i]);if(v>mx&&v<1e30f)mx=v;if(v<mn_v&&v>-1e30f)mn_v=v;}
        fprintf(stderr,"[dbg] L%d INPUT hc: min=%.2f max=%.2f h[0:4]=",lid,mn_v,mx);
        for(int i=0;i<4;i++){float v=bf16f(h[i]); fprintf(stderr,"%g ",v);}
        fprintf(stderr," (raw16=");
        for(int i=0;i<4;i++){uint16_t r; memcpy(&r,&h[i],2); fprintf(stderr,"0x%04x ", (unsigned)r);}
        fprintf(stderr,")\n"); fflush(stderr);
    }
    /* Baseline: dump the L0 INPUT (embedding, last prefill pos m13) before any
     * layer runs, so the per-layer cosine starts at 1.0 (engine & HF share
     * embed.weight).  Gated by FST_LAYER_DUMP. */
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_LAYER_DUMP")) {
        const int mlast = M - 1;
        std::vector<float> tmp((size_t)N_HC * hd);
        for (size_t i = 0; i < (size_t)N_HC * hd; i++)
            tmp[i] = bf16f(h[(size_t)mlast * hc_stride + i]);
        FILE* f = std::fopen("/tmp/eng_L00_in.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), tmp.size(), f); std::fclose(f); }
    }

    /* ── Attention sublayer ── */
    if (has_hc)
        hc_pre(M, h, w.hc_attn_fn.data(), w.hc_attn_scale.data(), w.hc_attn_base.data(),
               cur.data(), post_a.data(), comb_a.data());
    else
        for (int m = 0; m < M; m++) memcpy(cur.data() + m * hd, h + m * hc_stride, (size_t)hd * sizeof(bf16_t));
    if (lid == 0 && has_hc) {
        fprintf(stderr, "[dbg] L0 attn HC gates (post/comb) for m0:\n  post=");
        for (int h = 0; h < N_HC; h++) fprintf(stderr, "%.4f ", post_a[h]);
        fprintf(stderr, "\n  comb=");
        for (int i = 0; i < N_HC * N_HC; i++) fprintf(stderr, "%.4f ", comb_a[i]);
        fprintf(stderr, "\n  hc_attn_scale=%.4f %.4f %.4f  base[0..7]=",
                w.hc_attn_scale[0], w.hc_attn_scale[1], w.hc_attn_scale[2]);
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", w.hc_attn_base[i]);
        fprintf(stderr, "\n"); fflush(stderr);
    }
    for (int m = 0; m < M; m++)
        npu_rmsnorm_weighted(an.data() + m * hd, cur.data() + m * hd, w.attn_norm.data(), hd);
    process_mla(lid, an.data(), M);   /* an = attn_out */
    /* Dump MLA attention output (an, pre-hc_post) for L0 m13, to compare with HF
     * attn_out and isolate whether the stream-0 residual 0.912 cos comes from the
     * MLA output itself or the hc_post gate mixing. Gated by FST_ROUTED_DUMP. */
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_ROUTED_DUMP")) {
        const int m = M - 1;
        std::vector<float> tmp(hd);
        for (int d = 0; d < hd; d++) tmp[d] = bf16f(an[(size_t)m*hd + d]);
        FILE* f = std::fopen("/tmp/eng_L00_an.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
        float amx = 0;
        for (float v : tmp) amx = std::max(amx, std::fabs(v));
        fprintf(stderr,"[an-dump] L0 m=%d: an(mla out) -> /tmp/eng_L00_an.f32 |mx=%.4f\n", m, amx);
        fflush(stderr);
    }
    if (lid == 0 || lid == 1 || lid % 10 == 9 || lid == 42) {
        float mx=-1e30f,mn_v=1e30f;
        for(int i=0;i<M*hd;i++){float v=bf16f(an[i]);if(v>mx)mx=v;if(v<mn_v)mn_v=v;}
        fprintf(stderr,"[dbg] L%d attn_out: min=%.3f max=%.3f\n",lid,mn_v,mx);
    }

    if (has_hc) {
        hc_post(M, an.data(), h, post_a.data(), comb_a.data(), new_hc.data());
        audit_recurrence("attn", an.data(), h, post_a.data(), comb_a.data(), new_hc.data());
        memcpy(h, new_hc.data(), (size_t)M * hc_stride * sizeof(bf16_t));
    } else {
        for (int i = 0; i < M * hd; i++) h[i] = f2bf(bf16f(h[i]) + bf16f(an[i]));
    }
    /* Attention-only residual (after attn hc_post, BEFORE FFN) for m13 — to
     * isolate attn vs ffn as the L0 divergence source. Gated by FST_LAYER_DUMP. */
    if (is_prefill_ && M > 1 && std::getenv("FST_LAYER_DUMP")) {
        char path[128];
        std::snprintf(path, sizeof(path), "/tmp/eng_L%02d_attn.f32", lid);
        FILE* f = std::fopen(path, "wb");
        if (f) {
            const int mlast = M - 1;
            std::vector<float> tmp((size_t)N_HC * hd);
            for (size_t i = 0; i < (size_t)N_HC * hd; i++)
                tmp[i] = bf16f(h[(size_t)mlast * hc_stride + i]);
            std::fwrite(tmp.data(), sizeof(float), tmp.size(), f);
            std::fclose(f);
        }
    }

    /* ── FFN sublayer ── */
    if (has_hc)
        hc_pre(M, h, w.hc_ffn_fn.data(), w.hc_ffn_scale.data(), w.hc_ffn_base.data(),
               cur.data(), post_f.data(), comb_f.data());
    else
        for (int m = 0; m < M; m++) memcpy(cur.data() + m * hd, h + m * hc_stride, (size_t)hd * sizeof(bf16_t));
    for (int m = 0; m < M; m++)
        npu_rmsnorm_weighted(mn.data() + m * hd, cur.data() + m * hd, w.moe_norm.data(), hd);
    /* Dump engine's pre-rmsnorm FFN input (cur = hc_ffn_pre output) for L0 m13, to
     * isolate whether the mn-vs-HF-xf 0.93 direction error arises in the attention
     * residual (h), the hc_pre split, or the moe_norm. Gated by FST_ROUTED_DUMP. */
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_ROUTED_DUMP")) {
        const int m = M - 1;
        std::vector<float> tmp(hd);
        for (int d = 0; d < hd; d++) tmp[d] = bf16f(cur[(size_t)m*hd + d]);
        FILE* f = std::fopen("/tmp/eng_L00_cur.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
        float cmx = 0;
        for (float v : tmp) cmx = std::max(cmx, std::fabs(v));
        fprintf(stderr,"[cur-dump] L0 m=%d: cur(pre-norm) -> /tmp/eng_L00_cur.f32 |mx=%.4f\n", m, cmx);
        fflush(stderr);
    }
    if (lid == 0 || lid == 1) {
        float cmx=-1e30f,cmn=1e30f;
        for(int i=0;i<M*hd;i++){float v=bf16f(cur[i]);if(v>cmx)cmx=v;if(v<cmn)cmn=v;}
        float wmx=-1e30f,wmn=1e30f;
        for(int i=0;i<hd;i++){float v=bf16f(w.moe_norm[i]);if(v>wmx)wmx=v;if(v<wmn)wmn=v;}
        fprintf(stderr,"[dbg] L%d ffn cur(in): min=%.4f max=%.4f | moe_norm: min=%.4f max=%.4f\n",lid,cmn,cmx,wmn,wmx); fflush(stderr);
    }

    std::vector<int> eids(M * tk);
    std::vector<float> ewts(M * tk);
    if (lid == 0 || lid == 1) {
        float mx=-1e30f,mn_v=1e30f;
        for(int i=0;i<M*hd;i++){float v=bf16f(mn[i]);if(v>mx)mx=v;if(v<mn_v)mn_v=v;}
        fprintf(stderr,"[dbg] L%d ffn_in(mn): min=%.4f max=%.4f\n",lid,mn_v,mx); fflush(stderr);
    }
    /* Per-token FFN-input audit at the explosion layer: res/cur/mn max AND rms.
     * If mn scales with |res| (rms>>1 for peaky cur), moe_norm is failing to
     * normalize; if mn is bounded (~weight rms), the spike is FFN weight alignment. */
    std::vector<float> mn_snap;
    if (audit_on && (lid == 0 || lid == 2 || lid == 3 || lid == 9 || lid == 13 || lid == 26)) {
        mn_snap.assign((size_t)M * hd, 0.f);
        for (int m = 0; m < M; m++) {
            const bf16_t* r = h + (size_t)m * hc_stride;
            const bf16_t* c = cur.data() + m * hd;
            const bf16_t* n = mn.data() + m * hd;
            float rmx=0, rmssum=0, cmx=0, cmssum=0, nmx=0, nmssum=0;
            for (int d = 0; d < hd; d++) {
                float rv=std::fabs(bf16f(r[d])), cv=std::fabs(bf16f(c[d])), nv=std::fabs(bf16f(n[d]));
                if (rv>rmx) rmx=rv; rmssum += bf16f(r[d])*bf16f(r[d]);
                if (cv>cmx) cmx=cv; cmssum += bf16f(c[d])*bf16f(c[d]);
                if (nv>nmx) nmx=nv; nmssum += bf16f(n[d])*bf16f(n[d]);
                mn_snap[(size_t)m*hd + d] = bf16f(n[d]);
            }
            fprintf(stderr,"[ffn-in] L%d m=%d: |res|mx=%.2f rms=%.3f | cur|mx=%.3f rms=%.3f | mn|mx=%.3f rms=%.3f\n",
                    lid, m, rmx, sqrtf(rmssum/hd), cmx, sqrtf(cmssum/hd), nmx, sqrtf(nmssum/hd));
            /* Step-1 RMSNorm audit (m=0, L0/L26): print mean(x^2), eps, moe_norm rms,
             * and compare the NPU mn output to a host numpy-ref recomputing the EXACT
             * kernel formula y = cur/sqrt(mean(cur^2)+1e-6)*moe_norm.  ratio~1.0 =>
             * the NPU RMSNorm is correct (and mn rms == rms(moe_norm) by construction). */
            if (m == 0 && (lid == 0 || lid == 26)) {
                const bf16_t* g = w.moe_norm.data();
                double ss = 0, grms = 0;
                for (int d = 0; d < hd; d++) { ss += (double)bf16f(c[d])*bf16f(c[d]); grms += (double)bf16f(g[d])*bf16f(g[d]); }
                float mean_sq = (float)(ss / hd);
                float inv = 1.0f / sqrtf(mean_sq + 1e-6f);
                double rs = 0; float ref_mx = 0;
                for (int d = 0; d < hd; d++) { float rv = bf16f(c[d]) * bf16f(g[d]) * inv; if (std::fabs(rv) > ref_mx) ref_mx = std::fabs(rv); rs += (double)rv*rv; }
                float ref_rms = sqrtf((float)(rs / hd));
                float npu_rms = sqrtf(nmssum / hd);
                fprintf(stderr,"[rmsn-audit] L%d m=0: mean(x^2)=%.6f eps=1e-6 | moe_norm rms=%.4f | NPU mn rms=%.4f mx=%.4f | numpy ref rms=%.4f mx=%.4f | ratio=%.4f\n",
                        lid, mean_sq, sqrtf((float)(grms/hd)), npu_rms, nmx, ref_rms, ref_mx, npu_rms / (ref_rms > 1e-6f ? ref_rms : 1e-6f));
                fflush(stderr);
            }
        }
        fflush(stderr);
    }
    /* Dump the engine's FFN input (mn = moe_normed, bf16) for L0 m13 so a CPU ref
     * can recompute per-expert FFN with the EXACT engine activation + exact weights
     * + fp32 accumulation, to isolate NPU-GEMM precision vs bf16-activation vs a
     * real dispatch bug in the routed per-expert 0.80 cos. */
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_ROUTED_DUMP")) {
        const int m = M - 1;
        std::vector<float> tmp(hd);
        for (int d = 0; d < hd; d++) tmp[d] = bf16f(mn[(size_t)m*hd + d]);
        FILE* f = std::fopen("/tmp/eng_L00_mn.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
        fprintf(stderr,"[mn-dump] L0 m=%d: mn -> /tmp/eng_L00_mn.f32 |mx=%.4f\n",
                m, *std::max_element(tmp.begin(), tmp.end(),
                    [](float a, float b){ return std::fabs(a) < std::fabs(b); }));
        fflush(stderr);
    }
    npu_router(eids.data(), ewts.data(), mn.data(), w.router.data(),
               w.router_bias.empty() ? nullptr : w.router_bias.data(),
               M, config_.n_experts, tk, hd, lid, input_ids);
    pager_->predict_and_prefetch(lid, eids.data(), tk);
    /* ds4.c 7855/7863/7873: routed MoE and shared FFN BOTH take the moe_normed
     * input (`norm`) and write to SEPARATE buffers; ffn_out = moe + shared.
     * Previously process_expert_ffn overwrote mn with the routed output and
     * process_shared_expert then ran on the routed output — i.e. it computed
     * shared(routed) instead of shared(moe_normed), amplifying large routed
     * outputs into huge shared outputs and exploding the HC residual (L26
     * ±1632).  Now both read mn (moe_normed, read-only) and write to separate
     * output buffers; the caller sums routed + shared into mn. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> routed_out((size_t)M * hd);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> shared_out((size_t)M * hd);
    /* L0 prefill: dump FFN input (moe_norm output, = HF xf) for m=last. */
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_LAYER_DUMP")) {
        const int m = M - 1;
        std::vector<float> tmp(hd);
        for (int d = 0; d < hd; d++) tmp[d] = bf16f(mn[(size_t)m*hd + d]);
        FILE* f = std::fopen("/tmp/eng_L00_xf13.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
    }
    process_expert_ffn(lid, mn.data(), routed_out.data(), M, eids.data(), tk, ewts.data());
    process_shared_expert(lid, mn.data(), shared_out.data(), M);
    flush_pending_runs();
    for (int i = 0; i < M * hd; i++)
        mn[i] = f2bf(bf16f(routed_out[i]) + bf16f(shared_out[i]));
    /* L0 prefill M>1: dump routed/shared/ffn_out for m=last to compare with HF
     * (isolates FFN pre-hc_post from the hc expansion). Pure C++ diagnostic. */
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_LAYER_DUMP")) {
        const int m = M - 1;
        std::vector<float> tmp(hd);
        for (int d = 0; d < hd; d++) tmp[d] = bf16f(mn[(size_t)m*hd + d]);
        FILE* f = std::fopen("/tmp/eng_L00_ffn.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
        for (int d = 0; d < hd; d++) tmp[d] = bf16f(routed_out[(size_t)m*hd + d]);
        f = std::fopen("/tmp/eng_L00_routed.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
        for (int d = 0; d < hd; d++) tmp[d] = bf16f(shared_out[(size_t)m*hd + d]);
        f = std::fopen("/tmp/eng_L00_shared.f32", "wb");
        if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
        float rmx=0,smx=0,fmx=0;
        for (int d=0; d<hd; d++){
            float r=std::fabs(bf16f(routed_out[(size_t)m*hd+d]));
            float s=std::fabs(bf16f(shared_out[(size_t)m*hd+d]));
            float ff=std::fabs(bf16f(mn[(size_t)m*hd+d]));
            if(r>rmx)rmx=r; if(s>smx)smx=s; if(ff>fmx)fmx=ff;
        }
        fprintf(stderr,"[eng-split] L0 m=%d: |routed|mx=%.4f |shared|mx=%.4f |ffn|mx=%.4f\n",
                m, rmx, smx, fmx); fflush(stderr);
    }
    if (audit_on && (lid == 0 || lid == 2 || lid == 3 || lid == 9 || lid == 13 || lid == 26)) {
        for (int m = 0; m < M; m++) {
            const bf16_t* ro = routed_out.data() + (size_t)m * hd;
            const bf16_t* so = shared_out.data() + (size_t)m * hd;
            const bf16_t* fo = mn.data() + (size_t)m * hd;
            float rmx=0, smx=0, fmx=0;
            for (int d = 0; d < hd; d++) {
                if (std::fabs(bf16f(ro[d]))>rmx) rmx=std::fabs(bf16f(ro[d]));
                if (std::fabs(bf16f(so[d]))>smx) smx=std::fabs(bf16f(so[d]));
                if (std::fabs(bf16f(fo[d]))>fmx) fmx=std::fabs(bf16f(fo[d]));
            }
            fprintf(stderr,"[split] L%d m=%d: |routed|mx=%.3f |shared|mx=%.3f |ffn_out|mx=%.3f\n",
                    lid, m, rmx, smx, fmx);
        }
        fflush(stderr);
    }
    if (audit_on && (lid == 0 || lid == 2 || lid == 3 || lid == 9 || lid == 13 || lid == 26)) {
        for (int m = 0; m < M; m++) {
            const bf16_t* o = mn.data() + m * hd;
            const float* nin = mn_snap.data() + (size_t)m * hd;
            float omx=0, omssum=0, nimx=0, nimssum=0;
            for (int d = 0; d < hd; d++) {
                float v=bf16f(o[d]); if (std::fabs(v)>omx) omx=std::fabs(v); omssum += v*v;
                if (nin[d]>nimx) nimx=nin[d]; nimssum += nin[d]*nin[d];
            }
            fprintf(stderr,"[ffn-out] L%d m=%d: |ffn_out|mx=%.3f rms=%.3f (mn_in|mx=%.3f rms=%.3f ratio=%.1f)\n",
                    lid, m, omx, sqrtf(omssum/hd), nimx, sqrtf(nimssum/hd),
                    omx / (nimx>1e-6f ? nimx : 1e-6f));
        }
        fflush(stderr);
    }
    if (lid == 0 || lid == 1 || lid % 10 == 9 || lid == 42) {
        float mx=-1e30f,mn_v=1e30f;
        for(int i=0;i<M*hd;i++){float v=bf16f(mn[i]);if(v>mx)mx=v;if(v<mn_v)mn_v=v;}
        fprintf(stderr,"[dbg] L%d ffn_out: min=%.3f max=%.3f\n",lid,mn_v,mx);
    }

    if (has_hc) {
        hc_post(M, mn.data(), h, post_f.data(), comb_f.data(), new_hc.data());
        audit_recurrence("ffn", mn.data(), h, post_f.data(), comb_f.data(), new_hc.data());
        memcpy(h, new_hc.data(), (size_t)M * hc_stride * sizeof(bf16_t));
    } else {
        for (int i = 0; i < M * hd; i++) h[i] = f2bf(bf16f(h[i]) + bf16f(mn[i]));
    }

    /* DSpark: capture the HC-mean of the last 3 main layers (40/41/42) for the
     * draft's main_hidden (concat -> [3*hd=12288]).  HF: main_hiddens.append(
     * h.mean(dim=2)) at target_layer_ids.  Done after the full layer (attn+ffn)
     * so h is the post-layer residual. */
    if (lid >= config_.n_layers - 3)
        capture_target_hc(lid, h, M);

    /* Per-layer residual dump (prefill only, last position m13) for cosine-vs-HF
     * localization: h[mlast] is the 4-stream HC residual [N_HC*hd] AFTER layer lid.
     * Gated by FST_LAYER_DUMP.  Decode (M=1) is skipped so prefill dumps survive. */
    if (is_prefill_ && M > 1 && std::getenv("FST_LAYER_DUMP")) {
        char path[128];
        std::snprintf(path, sizeof(path), "/tmp/eng_L%02d_h.f32", lid);
        FILE* f = std::fopen(path, "wb");
        if (f) {
            const int mlast = M - 1;
            std::vector<float> tmp((size_t)N_HC * hd);
            for (size_t i = 0; i < (size_t)N_HC * hd; i++)
                tmp[i] = bf16f(h[(size_t)mlast * hc_stride + i]);
            std::fwrite(tmp.data(), sizeof(float), tmp.size(), f);
            std::fclose(f);
        }
    }

    if (lid == 0 || lid == 1 || lid % 10 == 9 || lid == 42) {
        float mx=-1e30f,mn_v=1e30f;
        for(int i=0;i<M*hc_stride;i++){float v=bf16f(h[i]);if(v>mx&&v<1e30f)mx=v;if(v<mn_v&&v>-1e30f)mn_v=v;}
        fprintf(stderr,"[dbg] L%d OUTPUT hc: min=%.2f max=%.2f\n",lid,mn_v,mx);
    }
}

// ── Prefill tiling ──────────────────────────────────────────────────────────
// The GEMM kernels are compiled for M=16.  For prefill with M<=16, process
// all tokens in a single chunk — no tiling needed.  For M>16 (long prompts),
// split into 16-token chunks.
void FSTEngine::prefill_tiled(bf16_t* hidden, int M, const int* input_ids,
                               int pos_start, bool reset_cmp) {
    const int CHUNK = 16;   // M=16 standardization (kernels compiled at M=16)
    const int hd = config_.hidden_dim;
    const int hc_stride = N_HC * hd;   /* hidden is the 4-stream HC residual */
    if (reset_cmp) reset_compressor_state();   /* fresh V4 compressor state per generation */
    double t_layer0 = now();
    double t_start = t_layer0;
    for (int l = 0; l < config_.n_layers; l++) {
        int offset = 0;
        seq_pos_ = pos_start;          /* each layer starts at the append position */
        while (offset < M) {
            int m = std::min(CHUNK, M - offset);
            process_layer(l, hidden + (size_t)offset * hc_stride, m,
                          input_ids ? input_ids + offset : nullptr);
            seq_pos_ += m;
            offset  += m;
        }
        double t_layer1 = now();
        fprintf(stderr, "[time] L%d %.2fs (cum %.2fs)\n", l,
                t_layer1 - t_layer0, t_layer1 - t_start); fflush(stderr);
        t_layer0 = t_layer1;
    }
    /* Leave seq_pos_ at pos_start so callers' `seq_pos_ += M` accounting
     * (decode starts at position pos_start+M) is unchanged from the non-tiled
     * path (pos_start=0 reproduces the original behaviour exactly). */
    seq_pos_ = pos_start;
}

/* ── Interactive multi-turn API ───────────────────────────────────────────
 * Thin wrappers around the proven generate() prefill section + decode-loop
 * body, so the numerics are byte-identical to the CLI path.  The only
 * difference from generate() is that prefill_user() APPENDS at the current
 * seq_pos_ (and skips the compressor reset when seq_pos_>0), letting a
 * multi-turn chat reuse the KV cache across turns without re-prefilling. */
void FSTEngine::reset_session() {
    reset_compressor_state();
    /* Zero the pre-sized KV buffers (host vector + device BO) so no stale rows
     * from a prior session leak into the next.  Attention only reads ws<S, so
     * this is defensive, but cheap (max_seq*512 bf16/layer). */
    for (int l = 0; l < config_.n_layers; l++) {
        auto& kvc = kv_cache_[l];
        std::fill(kvc.kv_latent.begin(), kvc.kv_latent.end(), f2bf(0.0f));
        std::fill(kvc.k_pe.begin(),     kvc.k_pe.end(),     f2bf(0.0f));
        size_t kv_bytes = kvc.kv_latent.size() * sizeof(bf16_t);
        size_t pe_bytes = kvc.k_pe.size() * sizeof(bf16_t);
        std::fill(kvc.kv_latent_bo.map<bf16_t*>(),
                  kvc.kv_latent_bo.map<bf16_t*>() + kvc.kv_latent.size(), f2bf(0.0f));
        kvc.kv_latent_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_bytes, 0);
        std::fill(kvc.k_pe_bo.map<bf16_t*>(),
                  kvc.k_pe_bo.map<bf16_t*>() + kvc.k_pe.size(), f2bf(0.0f));
        kvc.k_pe_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, pe_bytes, 0);
    }
    seq_pos_ = 0;
    is_prefill_ = true;
}

int FSTEngine::prefill_user(const std::vector<int>& token_ids, float temperature, float top_p) {
    int M = (int)token_ids.size();
    if (M <= 0) return -1;

    std::vector<bf16_t, AlignedAllocator<bf16_t>> hidden = forward_embeddings(token_ids);
    /* Expand to the 4-stream HC residual [M, N_HC, hd] (each stream starts as a
     * copy of the plain embedding — ds4.c hc_from_plain_embedding). */
    {
        const int hd = config_.hidden_dim;
        std::vector<bf16_t, AlignedAllocator<bf16_t>> hc((size_t)M * N_HC * hd);
        for (int m = 0; m < M; m++)
            for (int s = 0; s < N_HC; s++)
                memcpy(hc.data() + ((size_t)m * N_HC + s) * hd,
                       hidden.data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
        hidden = std::move(hc);
    }

    const int pos_start = seq_pos_;
    is_prefill_ = true;
    /* First turn (pos_start==0) resets the compressor; subsequent turns APPEND
     * with the running compressor state preserved. */
    prefill_tiled(hidden.data(), M, token_ids.data(), pos_start, /*reset_cmp=*/ pos_start == 0);

    int tid = npu_sample_token(hidden.data() + (size_t)(M - 1) * N_HC * config_.hidden_dim,
                               1, temperature, top_p);
    fprintf(stderr, "[chat] prefill-last=%d (pos_start=%d, M=%d)\n", tid, pos_start, M);
    fflush(stderr);
    seq_pos_ += M;
    return tid;
}

int FSTEngine::decode_step(int prev_tid, float temperature, float top_p) {
    const int hd = config_.hidden_dim;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> dec_hidden((size_t)N_HC * hd, f2bf(0.0f));
    if (prev_tid >= 0 && prev_tid < config_.vocab_size)
        for (int s = 0; s < N_HC; s++)
            memcpy(dec_hidden.data() + (size_t)s * hd,
                   host_embedding_table_.data() + (size_t)prev_tid * hd,
                   (size_t)hd * sizeof(bf16_t));

    for (int l = 0; l < config_.n_layers; l++)
        process_layer(l, dec_hidden.data(), 1, &prev_tid);

    int tid = npu_sample_token(dec_hidden.data(), 1, temperature, top_p);
    seq_pos_ += 1;
    if (seq_pos_ >= config_.max_seq - 1) seq_pos_ = 0;
    return tid;
}

/* ── V4 KV compressor helpers ──────────────────────────────────────────────
 * E4M3 round-trip quant LUT (ds4.c dsv4_e4m3fn_value_cpu / _dequant_cpu).
 * 127 positive grid values 0..448; nearest-neighbour with the ds4.c tie-break.
 * Used to quantize the [0,448) nope part of each compressed KV row in 64-dim
 * blocks (ds4.c dsv4_fp8_kv_quantize_row_inplace_cpu). */
static float g_e4m3_pos[127];
static bool g_e4m3_init = false;
static void init_e4m3_grid() {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    for (int i = 0; i < 127; i++) {
        int exp = (i >> 3) & 0x0f, mant = i & 0x07;
        g_e4m3_pos[i] = (exp == 0) ? (float)mant * 0.001953125f
                                  : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
    }
    g_e4m3_init = true;
}
static inline float e4m3_roundtrip(float v) {
    if (!g_e4m3_init) init_e4m3_grid();
    const float sign = v < 0.0f ? -1.0f : 1.0f;
    float ax = std::min(std::fabs(v), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (g_e4m3_pos[mid] <= ax) lo = mid; else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = std::fabs(ax - g_e4m3_pos[best]);
        float nd = std::fabs(ax - g_e4m3_pos[best + 1]);
        if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
    }
    return sign * g_e4m3_pos[best];
}

/* RoPE-tail YaRN for a single 512-dim compressed row (ds4.c
 * rope_tail_layer_inplace on out_comp with n_head=1, head_dim=512, n_rot=64).
 * Rotates the last 64 dims (offset n_nope=448) at absolute position `pos` for
 * layer `lid`.  Same YaRN params as build_rope_lut (freq_base=160000,
 * freq_scale=1/16, ext_factor=1, n_ctx_orig=65536, beta_fast=32, beta_slow=1);
 * the mscale magnitude factor cancels to 1.0. */
void FSTEngine::rope_tail_yarn(float* x512, int lid, int pos) {
    const int n_rot = config_.rope_dim;        /* 64 */
    const int head_dim = MLA_HEAD_DIM;         /* 512 */
    const int n_nope = head_dim - n_rot;        /* 448 */
    const bool compressed = (lid < (int)config_.compress_ratios.size())
                            ? (config_.compress_ratios[lid] != 0)
                            : (lid >= 2);
    const float freq_base = compressed ? 160000.0f : rope_freq_base_;
    const float theta_scale = std::pow(freq_base, -2.0f / (float)n_rot);
    float* tail = x512 + n_nope;
    if (!compressed) {
        float theta = (float)pos;
        for (int i = 0; i < n_rot; i += 2) {
            float c = cosf(theta), s = sinf(theta);
            float x0 = tail[i], x1 = tail[i + 1];
            tail[i]     = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
            theta *= theta_scale;
        }
        return;
    }
    const float freq_scale = 1.0f / 16.0f;
    const float ext_factor = 1.0f;
    const float n_ctx_orig = 65536.0f;
    const float beta_fast = 32.0f, beta_slow = 1.0f;
    auto corr_dim = [&](float beta) {
        return (float)n_rot * logf(n_ctx_orig / (beta * 2.0f * (float)M_PI)) / (2.0f * logf(freq_base));
    };
    float corr0 = std::max(0.0f, floorf(corr_dim(beta_fast)));
    float corr1 = std::min((float)(n_rot - 1), ceilf(corr_dim(beta_slow)));
    auto ramp = [&](int i0) {
        float y = ((float)(i0 / 2) - corr0) / std::max(0.001f, corr1 - corr0);
        return 1.0f - std::min(1.0f, std::max(0.0f, y));
    };
    float theta_extrap = (float)pos;
    for (int i = 0; i < n_rot; i += 2) {
        float ramp_mix = ramp(i) * ext_factor;
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        float c = cosf(theta), s = sinf(theta);
        float x0 = tail[i], x1 = tail[i + 1];
        tail[i]     = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
        theta_extrap *= theta_scale;
    }
}

/* Stream one token's attn-normed hidden through the per-layer V4 KV
 * compressor.  On a ratio boundary ((pos+1)%ratio==0) emit a 512-dim
 * compressed KV row into cmp_state_[lid].cache.  NPU projects (qc kernel,
 * M=1 padded to 32); host-float control ops match ds4.c
 * compressor_decode_one + compressor_pool_decode_state exactly.  Only
 * ratio=4 emits for S<128. */
void FSTEngine::reset_compressor_state() {
    for (int l = 0; l < config_.n_layers; l++) {
        auto& st = cmp_state_[l];
        std::fill(st.state_kv.begin(), st.state_kv.end(), 0.0f);
        std::fill(st.state_sc.begin(), st.state_sc.end(), 0.0f);
        st.cache.clear();
        st.n_comp = 0;
    }
}

void FSTEngine::compress_token(int lid, const bf16_t* hidden_normed, int pos) {
    auto& w = shared_[lid];
    if (w.cmp_ratio == 0 || w.cmp_wkv.empty() || w.cmp_wgate.empty() ||
        w.cmp_ape.empty() || w.cmp_norm.empty())
        return;
    const int ratio = w.cmp_ratio;
    const int cw = w.cmp_comp_width;          /* 1024 (ratio 4) / 512 (ratio 128) */
    const int head_dim = MLA_HEAD_DIM;        /* 512 */
    const int coff = (ratio == 4) ? 2 : 1;
    const int width = coff * head_dim;        /* == cw */
    const int pos_mod = pos % ratio;
    const int row = (ratio == 4) ? ratio + pos_mod : pos_mod;
    const bool should_compress = (((pos + 1) % ratio) == 0);

    auto& st = cmp_state_[lid];
    /* Project hidden[4096] -> kv_cur[width] (wkv) and sc_cur[width] (wgate) on
     * the NPU (qc kernel, M=1, N=width<=1024=N_c, K=4096=K_c, bcol).  Two
     * dispatches.  ZERO CPU GEMM. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kv_b(width, 0), sc_b(width, 0);
    npu_gemm_mla_vec("qck", kv_b.data(), hidden_normed, w.cmp_wkv.data(), 1, width, 4096);
    npu_gemm_mla_vec("qck", sc_b.data(), hidden_normed, w.cmp_wgate.data(), 1, width, 4096);
    std::vector<float, AlignedAllocator<float>> kv_cur(width), sc_cur(width);
    for (int j = 0; j < width; j++) { kv_cur[j] = bf16f(kv_b[j]); sc_cur[j] = bf16f(sc_b[j]); }

    /* APE add (ds4.c: sc_cur[j] += ape[j, pos_mod]).  ape is [ratio, cw] F32. */
    for (int j = 0; j < width; j++)
        sc_cur[j] += w.cmp_ape[(size_t)pos_mod * cw + j];

    /* State write into the secondary lane (ratio=4) or single lane (ratio=128). */
    memcpy(st.state_kv.data() + (size_t)row * width, kv_cur.data(), (size_t)width * sizeof(float));
    memcpy(st.state_sc.data() + (size_t)row * width, sc_cur.data(), (size_t)width * sizeof(float));

    if (!should_compress)
        return;

    /* Pool: per-dim softmax over the window rows (two-lane for ratio=4).
     * primary: state[r, j]            (j in [0,512))
     * secondary: state[ratio+r, 512+j] (j in [0,512))   [ratio=4 only] */
    std::vector<float, AlignedAllocator<float>> pooled(head_dim, 0.0f);
    for (int j = 0; j < head_dim; j++) {
        float max_score = -1e30f;
        if (ratio == 4) {
            for (int r = 0; r < ratio; r++) {
                float sp = st.state_sc[(size_t)r * width + j];
                float sc2 = st.state_sc[(size_t)(ratio + r) * width + head_dim + j];
                if (sp > max_score) max_score = sp;
                if (sc2 > max_score) max_score = sc2;
            }
        } else {
            for (int r = 0; r < ratio; r++) {
                float s = st.state_sc[(size_t)r * width + j];
                if (s > max_score) max_score = s;
            }
        }
        if (max_score <= -0.5e30f) { pooled[j] = 0.0f; continue; }
        float denom = 0.0f, sum = 0.0f;
        if (ratio == 4) {
            for (int r = 0; r < ratio; r++) {
                float wp = expf(st.state_sc[(size_t)r * width + j] - max_score);
                float wc = expf(st.state_sc[(size_t)(ratio + r) * width + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * st.state_kv[(size_t)r * width + j];
                sum += wc * st.state_kv[(size_t)(ratio + r) * width + head_dim + j];
            }
        } else {
            for (int r = 0; r < ratio; r++) {
                float ww = expf(st.state_sc[(size_t)r * width + j] - max_score);
                denom += ww;
                sum += ww * st.state_kv[(size_t)r * width + j];
            }
        }
        pooled[j] = denom > 0.0f ? sum / denom : 0.0f;
    }

    /* RMS-norm with cmp_norm[512] (ds4.c DS4_RMS_EPS=1e-6). */
    double ss = 0.0;
    for (int i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + 1e-6f);
    std::vector<float, AlignedAllocator<float>> out(head_dim);
    for (int i = 0; i < head_dim; i++)
        out[i] = pooled[i] * rms * bf16f(w.cmp_norm[i]);

    /* RoPE tail (last 64 dims) at comp_pos = pos+1-ratio. */
    const int comp_pos = pos + 1 - ratio;
    rope_tail_yarn(out.data(), lid, comp_pos);

    /* E4M3 round-trip quant on [0,448) in 64-blocks (last 64 rope dims left as-is). */
    const int n_nope = head_dim - config_.rope_dim;   /* 448 */
    for (int off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (int i = 0; i < 64; i++) { float av = std::fabs(out[off + i]); if (av > amax) amax = av; }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (int i = 0; i < 64; i++) {
            float v = out[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            out[off + i] = e4m3_roundtrip(v) * scale;
        }
    }

    /* Push the 512-dim compressed row (BF16) into the cache. */
    st.cache.resize((size_t)(st.n_comp + 1) * head_dim);
    for (int i = 0; i < head_dim; i++)
        st.cache[(size_t)st.n_comp * head_dim + i] = f2bf(out[i]);

    /* One-shot per-layer audit of every compressor stage. */
    if (false && (lid == 2 || lid == 28) && pos < 14) {
        auto mm = [](const float* p, int n, const char* /*t*/){ float mn=1e30f,mx=-1e30f; for(int i=0;i<n;i++){float v=p[i]; if(v<mn)mn=v; if(v>mx)mx=v;} return std::pair<float,float>{mn,mx}; };
        float pmax = 0.0f; for (int i = 0; i < head_dim; i++) pmax = std::max(pmax, std::fabs(pooled[i]));
        auto k = mm(kv_cur.data(), width, "");
        auto s = mm(sc_cur.data(), width, "");
        auto o = mm(out.data(), head_dim, "");
        fprintf(stderr, "[cmp-audit] L%d pos=%d row=%d: kv[%g,%g] sc[%g,%g] pool_max=%g pooled[%g,%g] "
                "rms*norm_out[%g,%g] nope[%g,%g] rope[%g,%g] norm_w[%g,%g]\n",
                lid, pos, row, k.first, k.second, s.first, s.second, pmax,
                /*pooled mm*/ mm(pooled.data(), head_dim, "").first, mm(pooled.data(), head_dim, "").second,
                o.first, o.second,
                /*nope*/ mm(out.data(), 448, "").first, mm(out.data(), 448, "").second,
                /*rope tail*/ mm(out.data() + 448, 64, "").first, mm(out.data() + 448, 64, "").second,
                /*norm weight*/ (float)bf16f(w.cmp_norm[0]), (float)bf16f(w.cmp_norm[256]));
        fflush(stderr);
    }

    st.n_comp++;

    /* Window rotation (ratio=4 only, ds4.c 8739): lane0 := old lane1, then
     * lane1 := new lane0.  Both lanes become the just-processed window so the
     * next window's tokens overwrite the secondary lane as they arrive. */
    if (ratio == 4) {
        for (int r = 0; r < ratio; r++) {
            memcpy(st.state_kv.data() + (size_t)r * width,
                   st.state_kv.data() + (size_t)(ratio + r) * width,
                   (size_t)width * sizeof(float));
            memcpy(st.state_sc.data() + (size_t)r * width,
                   st.state_sc.data() + (size_t)(ratio + r) * width,
                   (size_t)width * sizeof(float));
        }
        for (int r = 0; r < ratio; r++) {
            memcpy(st.state_kv.data() + (size_t)(ratio + r) * width,
                   st.state_kv.data() + (size_t)r * width,
                   (size_t)width * sizeof(float));
            memcpy(st.state_sc.data() + (size_t)(ratio + r) * width,
                   st.state_sc.data() + (size_t)r * width,
                   (size_t)width * sizeof(float));
        }
    }
}

void FSTEngine::process_expert_ffn_hy3(int lid, const bf16_t* h, float* acc, int M,
                                       const int* eids, const float* wts) {
    /* Routed-expert FFN.  Default = NPU BO-to-BO chain (mirrors DS4
     * process_expert_ffn, inter=1536): get_expert_bo (LRU host_only BO) →
     * hy3_dequant (MXFP4→BF16 B[N,K]) → hy3_gemm/hy3_gemm2 (gate/up) →
     * silu/mul (ew_unified) → hy3_gemm_down → readback → router-weighted acc.
     * FST_HY3_HOST_FFN selects the verified host-first path (host dequant + 3
     * fp32 GEMMs/expert) for A/B correctness comparison.  Expert block =
     * gate[1536,4096]+up+down[4096,1536] dense MXFP4, 10,027,008 B total. */
    const int hd = config_.hidden_dim, inter = config_.expert_inter_dim, tk = config_.top_k;
    const int total = M * tk;

    if (std::getenv("FST_HY3_HOST_FFN")) {
        const size_t proj_bytes = (size_t)(inter / 32) * 17 * hd;   /* 3,342,336 */
        const size_t proj_elems = (size_t)inter * hd;
        const size_t down_elems = (size_t)hd * inter;
        std::vector<float> hf((size_t)M * hd);
        for (size_t i = 0; i < (size_t)M * hd; i++) hf[i] = bf16f(h[i]);
        std::vector<float> gate_f(proj_elems), up_f(proj_elems), down_f(down_elems);
        std::vector<float> eout((size_t)M * hd, 0.0f);
        for (int j = 0; j < total; j++) {
            int e = eids[j]; float w = wts[j]; int m = j / tk;
            const Expert& exp = pager_->get(lid, e);
            const uint8_t* pk = exp.packed_weights.data();
            dequant_mxfp4_dense(gate_f.data(), pk,                inter, hd);
            dequant_mxfp4_dense(up_f.data(),   pk + proj_bytes,    inter, hd);
            dequant_mxfp4_dense(down_f.data(), pk + 2 * proj_bytes, hd, inter);
            swiglu_f32_host(hf.data(), gate_f.data(), up_f.data(), down_f.data(), M, hd, inter, eout.data());
            for (int d = 0; d < hd; d++) acc[(size_t)m * hd + d] += w * eout[(size_t)m * hd + d];
        }
        return;
    }

    /* FST_HY3_FUSED_FFN (ctor guard): dispatch-collapsed batched path — pack ≤8
     * experts into one 80 MB BO, ONE dequant + 5 batched MC dispatches (6/layer
     * vs 48 in the per-op path below).  Guard unset falls through to that path. */
    if (hy3_fused_ffn_) {
        process_expert_ffn_hy3_fused(lid, h, acc, M, eids, wts);
        return;
    }

    /* ── NPU BO-to-BO routed-expert FFN ── */
    const int Mx = 16;                       // kernel M-tile (gate/up/down compiled M=16)
    const int rep = Mx / M;                  // M=1 -> 16, M=16 -> 1
    const size_t proj_elems = (size_t)hd * inter;
    const size_t proj_sz    = proj_elems * sizeof(bf16_t);          // 12,582,912 B
    const size_t full_b     = (size_t)hd * inter * sizeof(bf16_t); // down B [N=hd,K=inter]
    const size_t in_sz      = (size_t)Mx * hd * sizeof(bf16_t);

    auto& deq_krnl       = kernel_cache_->get("hy3_dequant");
    auto& gemm_krnl      = kernel_cache_->get("hy3_gemm");       // gate (M=16,K=4096,N=1536)
    auto& gemm2_krnl     = kernel_cache_->get("hy3_gemm2");      // up   (M=16,K=4096,N=1536)
    auto& gemm_down_krnl = kernel_cache_->get("hy3_gemm_down"); // down (M=16,K=1536,N=4096)

    if (bo_scratch_in_.size() < in_sz) bo_scratch_in_ = xrt::ext::bo(npu_device_, in_sz);
    {
        char* hp = bo_scratch_in_.map<char*>();
        for (int r = 0; r < rep; r++)
            std::memcpy(hp + (size_t)r * M * hd * sizeof(bf16_t), h,
                        (size_t)M * hd * sizeof(bf16_t));
        int filled = rep * M;
        if (filled < Mx)
            std::memset(hp + (size_t)filled * hd * sizeof(bf16_t), 0,
                        (size_t)(Mx - filled) * hd * sizeof(bf16_t));
        npu_sync_to(bo_scratch_in_);
    }

    std::vector<bf16_t, AlignedAllocator<bf16_t>> down_out(Mx * hd);

    /* drain_set: read back set s's down output (after flush) and accumulate the
     * owner expert's router-weighted contribution into acc. */
    int drained_count = 0;
    auto drain_set = [&](int s) {
        int e = scratch_pool_[s].owner_e;
        if (e < 0) return;
        const size_t c_sz = (size_t)Mx * hd * sizeof(bf16_t);
        npu_sync_from(scratch_pool_[s].bo_down);
        std::memcpy(down_out.data(), scratch_pool_[s].bo_down.map<char*>(), c_sz);
        scratch_pool_[s].owner_e = -1;

        /* FST_HY3_FFN_AUDIT: on lid 0 / first drained expert, localize the NPU
         * divergence vs the host path.  (1) dequant cos: NPU sc.bo_w gate slice
         * vs host dequant_mxfp4_dense of the same packed expert — isolates the
         * new 589824-block dequant kernel.  (2) GEMM cos: NPU down_out[0:hd] vs a
         * host fp32 FFN computed on the NPU-dequanted weights (sc.bo_w) —
         * isolates the new N=1536 gate/up/down GEMM chain. */
        if (lid == 1 && drained_count == 0 && std::getenv("FST_HY3_FFN_AUDIT")) {
            ++drained_count;
            npu_sync_from(scratch_pool_[s].bo_w);
            const bf16_t* Wnpu = scratch_pool_[s].bo_w.map<bf16_t*>();
            const Expert& exp = pager_->get(lid, e);
            const uint8_t* pk = exp.packed_weights.data();
            const size_t paud = (size_t)(inter / 32) * 17 * hd;
            std::vector<float> gate_ref((size_t)inter * hd);
            dequant_mxfp4_dense(gate_ref.data(), pk, inter, hd);
            float dmn = 1e30f, dmx = -1e30f, dd = 0; double dot = 0, nn = 0, nr = 0;
            for (size_t i = 0; i < (size_t)inter * hd; i++) {
                float a = bf16f(Wnpu[i]), b = gate_ref[i];
                if (a < dmn) dmn = a; if (a > dmx) dmx = a;
                float df = std::fabs(a - b); if (df > dd) dd = df;
                dot += (double)a * b; nn += (double)a * a; nr += (double)b * b;
            }
            double dcos = (nn > 0 && nr > 0) ? dot / (std::sqrt(nn) * std::sqrt(nr)) : 0;
            fprintf(stderr, "[hy3-audit] L0 e=%d DEQUANT gate: npu mn=%.4f mx=%.4f "
                    "maxdiff=%.4f cos=%.5f |npu|/|ref|=%.5f\n", e, dmn, dmx, dd, dcos,
                    (nr > 0) ? std::sqrt(nn / nr) : 0); fflush(stderr);
            const bf16_t* Wg = Wnpu;
            const bf16_t* Wu = Wnpu + (size_t)inter * hd;
            const bf16_t* Wd = Wnpu + 2 * (size_t)inter * hd;
            std::vector<float> act(inter);
            for (int n = 0; n < inter; n++) {
                double g = 0, u = 0;
                for (int k = 0; k < hd; k++) {
                    g += (double)bf16f(h[k]) * (double)bf16f(Wg[(size_t)n * hd + k]);
                    u += (double)bf16f(h[k]) * (double)bf16f(Wu[(size_t)n * hd + k]);
                }
                float gf = (float)g, uf = (float)u;
                act[n] = (gf / (1.0f + expf(-gf))) * uf;
            }
            float gmn = 1e30f, gmx = -1e30f, gmd = 0; double gdot = 0, gnn = 0, gnr = 0;
            for (int d = 0; d < hd; d++) {
                double a = 0;
                for (int k = 0; k < inter; k++)
                    a += (double)act[k] * (double)bf16f(Wd[(size_t)d * inter + k]);
                float ref = (float)a, npu = bf16f(down_out[(size_t)0 * hd + d]);
                if (npu < gmn) gmn = npu; if (npu > gmx) gmx = npu;
                float df = std::fabs(npu - ref); if (df > gmd) gmd = df;
                gdot += (double)npu * ref; gnn += (double)npu * npu; gnr += (double)ref * ref;
            }
            double gcos = (gnn > 0 && gnr > 0) ? gdot / (std::sqrt(gnn) * std::sqrt(gnr)) : 0;
            fprintf(stderr, "[hy3-audit] L0 e=%d GEMM down: npu mn=%.4f mx=%.4f "
                    "maxdiff=%.4f cos=%.5f |npu|/|ref|=%.5f\n", e, gmn, gmx, gmd, gcos,
                    (gnr > 0) ? std::sqrt(gnn / gnr) : 0); fflush(stderr);
        }

        for (int j = 0; j < total; j++) {
            if (eids[j] != e) continue;
            int m = j / tk;
            float w = wts[j];
            for (int d = 0; d < hd; d++)
                acc[(size_t)m * hd + d] += bf16f(down_out[(size_t)m * hd + d]) * w;
        }
    };

    std::vector<bool> seen(config_.n_experts, false);
    pool_idx_ = 0;
    int prev_set = -1;

    for (int i = 0; i < total; i++) {
        int e = eids[i];
        if (seen[e]) continue;
        seen[e] = true;

        const int s = pool_idx_ & 3;
        ++pool_idx_;
        ScratchSet& sc = scratch_pool_[s];

        /* NPU MXFP4 dequant → sc.bo_w (BO-to-BO).  get_expert_bo returns a
         * persistent host_only BO (HIT: no SSD/memcpy; MISS: SSD load + sync,
         * LRU-bounded by expert_bo_cap_).  hy3_dequant reads exactly 10,027,008 B
         * (589,824 blocks) and writes 3*proj_elems BF16 = 37.7 MB into sc.bo_w. */
        xrt::bo& wbo = get_expert_bo(lid, e);
        npu_sync_to(sc.bo_w);
        auto drun = deq_krnl(3, 0, 0,
            static_cast<xrt::bo&>(wbo), static_cast<xrt::bo&>(sc.bo_w),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_),
            static_cast<xrt::bo&>(bo_d3_));
        pending_runs_.push_back(std::move(drun));

        /* Cross-ctx barrier (dequant -> gate): also waits the PREVIOUS expert's
         * down (async overlap).  Drain prev before reusing its set's BOs. */
        flush_pending_runs();
        if (prev_set >= 0) { drain_set(prev_set); prev_set = -1; }

        /* gate + up GEMMs: B read from sc.bo_w sub-buffers (b_col_maj [N,K]). */
        npu_sync_to(sc.bo_ha);
        npu_sync_to(sc.bo_hb);
        {
            xrt::bo bo_dq_gate(sc.bo_w, proj_sz, 0);
            auto run = gemm_krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_dq_gate),
                static_cast<xrt::bo&>(sc.bo_ha),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        {
            xrt::bo bo_dq_up(sc.bo_w, proj_sz, proj_sz);
            auto run = gemm2_krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_dq_up),
                static_cast<xrt::bo&>(sc.bo_hb),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        flush_pending_runs();

        /* silu(gate) then silu*up (ew_unified, baked N=65536; down reads only
         * Mx*inter=24576 so the stale tail is ignored — same as DS4). */
        npu_sync_to(sc.bo_silu);
        npu_sync_to(sc.bo_mul);
        npu_ew_async("silu", sc.bo_ha, sc.bo_silu);
        npu_ew_bin_async("mul", sc.bo_silu, sc.bo_hb, sc.bo_mul);
        flush_pending_runs();

        /* down GEMM → sc.bo_down.  NO flush/readback yet — overlaps the NEXT
         * expert's dequant (different hw_context).  owner_e marks a pending drain. */
        {
            xrt::bo bo_dq_dn(sc.bo_w, full_b, 2 * proj_sz);
            xrt::bo bo_c(sc.bo_down, (size_t)Mx * hd * sizeof(bf16_t), 0);
            auto run = gemm_down_krnl(3, 0, 0,
                static_cast<xrt::bo&>(sc.bo_mul), static_cast<xrt::bo&>(bo_dq_dn),
                static_cast<xrt::bo&>(bo_c),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        sc.owner_e = e;
        prev_set = s;
    }

    /* Drain the final expert's down. */
    if (prev_set >= 0) {
        flush_pending_runs();
        drain_set(prev_set);
    }
}

/* ── ARCH_HY3 routed-expert FFN — dispatch-collapsed (FST_HY3_FUSED_FFN) ────
 * Replaces the per-op path's 6-dispatches-per-expert loop (48/layer for 8
 * experts) with a BATCHED multi-core pipeline over all ≤8 unique experts:
 *   1. host-pack 8 experts' MXFP4 into bo_fused_weight_ (80 MB, ~8 ms)
 *   2. ONE dequant_8exp dispatch → bo_batch_w_ (302 MB, expert-major [E,3,N,K])
 *   3. gate MC + up MC (8 cores, 1 expert/core; B sub-buffer @0 / @+proj_sz)
 *   4. silu_b + mul_b (batched over 8*16*1536 = 196608 elems, reused ew kernels)
 *   5. down MC (A = silu*up [E*Mx,1536]; B sub @+2*proj_sz) → S.bo_down
 *   6. host readback + router-weighted accumulate
 * = 6 dispatches/layer (1 dequant + 2 GEMM + 2 ew + 1 down) vs 48.  Batches
 * n_unique>8 (prefill M=16 → up to 128 unique) in groups of 8, reusing the BOs.
 * The dequant output layout (expert-major gate|up|down contiguous) is IDENTICAL
 * to run_batched_ffn's 3x-stride B layout, so the MC kernels read it via the same
 * sub-buffer base shifts — verified size-equal: 8*3*4096*1536*2 == 6*3*4096*2048*2. */
void FSTEngine::process_expert_ffn_hy3_fused(int lid, const bf16_t* h, float* acc, int M,
                                              const int* eids, const float* wts) {
    const int hd = config_.hidden_dim, inter = config_.expert_inter_dim, tk = config_.top_k;
    const int Mx = 16, rep = Mx / M;
    constexpr int E_MC = 8;
    const size_t proj_elems = (size_t)hd * inter;            // 6,291,456
    const size_t proj_sz    = proj_elems * sizeof(bf16_t);   // 12,582,912
    const size_t slot_sz    = 3 * proj_sz;                   // 37,748,736 (expert slot)
    const size_t full_b     = (size_t)E_MC * 3 * proj_elems * sizeof(bf16_t); // 301,989,888
    const size_t expert_pk  = 3 * (size_t)(inter / 32) * 17 * hd;             // 10,027,008
    const int total = M * tk;

    /* Dedup selected experts (≤ M*tk unique, batched in groups of E_MC=8). */
    std::vector<int> slot_eids;
    std::vector<bool> seen(config_.n_experts, false);
    for (int i = 0; i < total; i++) {
        int e = eids[i];
        if (!seen[e]) { seen[e] = true; slot_eids.push_back(e); }
    }
    const int n_unique = (int)slot_eids.size();
    if (n_unique <= 0) return;

    auto& deq_krnl = kernel_cache_->get("hy3_dequant_8exp"); // 8-expert batched dequant
    auto& gemm_mc  = kernel_cache_->get("hy3_gemm_mc");        // gate/up (a_batched=False)
    auto& gemm_dn  = kernel_cache_->get("hy3_gemm_down_mc");   // down   (a_batched=True)
    ScratchSet& S  = scratch_pool_[0];                         // single batch, no ping-pong

    /* Upload activation h replicated to Mx rows (shared read-only A for gate/up). */
    const size_t in_sz = (size_t)Mx * hd * sizeof(bf16_t);
    if (bo_scratch_in_.size() < in_sz) bo_scratch_in_ = xrt::ext::bo(npu_device_, in_sz);
    {
        char* hp = bo_scratch_in_.map<char*>();
        for (int r = 0; r < rep; r++)
            std::memcpy(hp + (size_t)r * M * hd * sizeof(bf16_t), h,
                        (size_t)M * hd * sizeof(bf16_t));
        int filled = rep * M;
        if (filled < Mx)
            std::memset(hp + (size_t)filled * hd * sizeof(bf16_t), 0,
                        (size_t)(Mx - filled) * hd * sizeof(bf16_t));
        npu_sync_to(bo_scratch_in_);
    }

    bool audited = false;
    for (int batch_start = 0; batch_start < n_unique; batch_start += E_MC) {
        const int nbatch = std::min(E_MC, n_unique - batch_start);
        double ts0 = 0, ts1 = 0, ts2 = 0, ts3 = 0, ts4 = 0, ts5 = 0, ts6 = 0;
        const bool step_time = (lid == 1 && batch_start == 0 &&
                                std::getenv("FST_HY3_FUSED_AUDIT"));
        if (step_time) ts0 = now();

        /* Pack nbatch experts' MXFP4 into bo_fused_weight_ (8×10 MB host→host
         * memcpy, ~8 ms).  host_only BO is cache-coherent with the NPU on Ryzen
         * AI, so NO sync(TO_DEVICE) — the dequant DMA reads host memory directly
         * (same coherence get_expert_bo relies on for per-dispatch reads).  The
         * prior 80 MB sync measured ~80 ms pure overhead.  Unused slots zeroed
         * so dequant → 0 → GEMM output exactly 0 (no NaN in the float acc). */
        {
            char* wp = bo_fused_weight_.map<char*>();
            /* Fused path BYPASSES get_expert_bo: pack directly from the pager's
             * RAM-cached Expert structs.  get_expert_bo's BO cache adds a SERIAL
             * 10 MB host_only write per BO-cache MISS (~10 ms each = 80 ms/layer)
             * the fused path doesn't need — dequant_8exp reads this packed BO,
             * not per-expert BOs.  pager hit_rate=1.0 (RAM), so pager_->get is a
             * RAM hit (µs).  Fetch the 8 Expert refs serially (get() takes a
             * mutex), then parallelize the 80 MB write into bo_fused_weight_
             * (host_only uncacheable; 8 cores aggregate ~33 GB/s vs ~1 GB/s
             * single-thread).  sync fences the parallel writes for the NPU DMA. */
            const Expert* exps[E_MC];
            for (int ls = 0; ls < nbatch; ls++)
                exps[ls] = &pager_->get(lid, slot_eids[batch_start + ls]);
            double pm0 = step_time ? now() : 0;
            #pragma omp parallel for schedule(static)
            for (int ls = 0; ls < nbatch; ls++)
                std::memcpy(wp + (size_t)ls * expert_pk,
                            exps[ls]->packed_weights.data(), expert_pk);
            double pm1 = step_time ? now() : 0;
            for (int ls = nbatch; ls < E_MC; ls++)
                std::memset(wp + (size_t)ls * expert_pk, 0, expert_pk);
            /* Parallel host writes to uncacheable host_only memory are NOT
             * ordered w.r.t. the NPU dequant DMA without a sync — a no-sync
             * parallel pack raced (dequant cos 1.0→0.99995, one stale nibble).
             * This sync is a coherence fence, NOT the old ~80 ms cost (that was
             * the BO-cache insert, now bypassed). */
            bo_fused_weight_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            if (step_time) {
                fprintf(stderr, "[hy3-fused-pack] fetch=%.2fms memcpy=%.1fms sync=%.2fms\n",
                        (pm0 - ts0) * 1e3, (pm1 - pm0) * 1e3, (now() - pm1) * 1e3);
                fflush(stderr);
            }
        }
        if (step_time) ts1 = now();

        /* (1) dequant all 8 experts → bo_batch_w_ (expert-major [E,3,N,K]). */
        auto drun = deq_krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_fused_weight_), static_cast<xrt::bo&>(bo_batch_w_),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_),
            static_cast<xrt::bo&>(bo_d3_));
        pending_runs_.push_back(std::move(drun));
        flush_pending_runs();
        if (step_time) ts2 = now();

        /* (2) gate (B @0) + up (B sub @+proj_sz).  A = bo_scratch_in_ (replicated). */
        npu_sync_to(S.bo_ha); npu_sync_to(S.bo_hb);
        {
            auto run = gemm_mc(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_batch_w_),
                static_cast<xrt::bo&>(S.bo_ha),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        {
            xrt::bo boBup(bo_batch_w_, full_b - proj_sz, proj_sz);
            auto run = gemm_mc(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(boBup),
                static_cast<xrt::bo&>(S.bo_hb),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        flush_pending_runs();
        if (step_time) ts3 = now();

        /* (3) silu(gate)*up on HOST.  The ew_unified silu_b/mul_b kernels are
         * SINGLE-core (gen_ew_unified.py: one Worker): 196608 elems on one AIE
         * core = 94 ms (2×47 ms) — the NPU regresses ~100× vs CPU on this 393 KB
         * elementwise.  Same accepted small-op host deviation as GQA/shared/dense
         * (NPU regresses small/elementwise work).  2 device→host + 1 host→device
         * sync of 393 KB ≈ 3-5 ms, CPU silu*up ~0.2 ms.  Float silu is also MORE
         * accurate than the bf16 ew kernel, so the down-GEMM cos holds/improves.
         * Removes 2 NPU dispatches/layer (6→4: deq,gate,up,down). */
        npu_sync_from(S.bo_ha);
        npu_sync_from(S.bo_hb);
        {
            const bf16_t* gate = S.bo_ha.map<bf16_t*>();
            const bf16_t* up   = S.bo_hb.map<bf16_t*>();
            bf16_t* mul        = S.bo_mul.map<bf16_t*>();
            const int nmul = E_MC * Mx * inter;   // 196608
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nmul; i++) {
                float g = bf16f(gate[i]), u = bf16f(up[i]);
                mul[i] = f2bf((g / (1.0f + expf(-g))) * u);
            }
        }
        npu_sync_to(S.bo_mul);
        if (step_time) ts4 = now();

        /* (4) down (A = S.bo_mul [E_MC*Mx,1536]; B sub @+2*proj_sz) → S.bo_down. */
        npu_sync_to(S.bo_down);
        {
            xrt::bo boBdn(bo_batch_w_, full_b - 2 * proj_sz, 2 * proj_sz);
            auto run = gemm_dn(3, 0, 0,
                static_cast<xrt::bo&>(S.bo_mul), static_cast<xrt::bo&>(boBdn),
                static_cast<xrt::bo&>(S.bo_down),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        flush_pending_runs();
        if (step_time) ts5 = now();

        /* FST_HY3_FUSED_AUDIT (lid 1, first batch): localize the fused-path NPU
         * divergence vs host.  (1) dequant cos: bo_batch_w_ slot-0 gate vs host
         * dequant_mxfp4_dense.  (2) GEMM cos: NPU down[0:hd] vs host fp32 FFN
         * computed on the NPU-dequanted slot-0 weights. */
        npu_sync_from(S.bo_down);
        const bf16_t* down = S.bo_down.map<bf16_t*>();
        if (step_time) {
            ts6 = now();
            fprintf(stderr, "[hy3-fused-step] L1 batch0 M=%d nbatch=%d  pack=%.1f deq=%.1f "
                    "gateup=%.1f silumul=%.1f down=%.1f readback=%.1f total=%.1fms\n",
                    M, nbatch, (ts1-ts0)*1e3, (ts2-ts1)*1e3, (ts3-ts2)*1e3, (ts4-ts3)*1e3,
                    (ts5-ts4)*1e3, (ts6-ts5)*1e3, (ts6-ts0)*1e3); fflush(stderr);
        }
        if (lid == 1 && batch_start == 0 && !audited && std::getenv("FST_HY3_FUSED_AUDIT")) {
            audited = true;
            int e0 = slot_eids[0];
            xrt::bo wsub(bo_batch_w_, 3 * proj_sz, 0);   // slot 0 gate|up|down (37.7 MB)
            npu_sync_from(wsub);
            const bf16_t* Wnpu = wsub.map<bf16_t*>();
            const Expert& exp = pager_->get(lid, e0);
            const uint8_t* pk = exp.packed_weights.data();
            std::vector<float> gate_ref((size_t)inter * hd);
            dequant_mxfp4_dense(gate_ref.data(), pk, inter, hd);
            float dmn=1e30f, dmx=-1e30f, dd=0; double dot=0, nn=0, nr=0;
            for (size_t i = 0; i < (size_t)inter * hd; i++) {
                float a = bf16f(Wnpu[i]), b = gate_ref[i];
                if (a < dmn) dmn = a; if (a > dmx) dmx = a;
                float df = std::fabs(a - b); if (df > dd) dd = df;
                dot += (double)a * b; nn += (double)a * a; nr += (double)b * b;
            }
            double dcos = (nn > 0 && nr > 0) ? dot / (std::sqrt(nn) * std::sqrt(nr)) : 0;
            fprintf(stderr, "[hy3-fused-audit] L1 e=%d DEQUANT gate: npu mn=%.4f mx=%.4f "
                    "maxdiff=%.4f cos=%.5f |npu|/|ref|=%.5f\n", e0, dmn, dmx, dd, dcos,
                    (nr > 0) ? std::sqrt(nn / nr) : 0); fflush(stderr);
            const bf16_t* Wg = Wnpu;
            const bf16_t* Wu = Wnpu + (size_t)inter * hd;
            const bf16_t* Wd = Wnpu + 2 * (size_t)inter * hd;
            std::vector<float> act(inter);
            for (int n = 0; n < inter; n++) {
                double g = 0, u = 0;
                for (int k = 0; k < hd; k++) {
                    g += (double)bf16f(h[k]) * (double)bf16f(Wg[(size_t)n * hd + k]);
                    u += (double)bf16f(h[k]) * (double)bf16f(Wu[(size_t)n * hd + k]);
                }
                float gf = (float)g, uf = (float)u;
                act[n] = (gf / (1.0f + expf(-gf))) * uf;
            }
            float gmn=1e30f, gmx=-1e30f, gmd=0; double gdot=0, gnn=0, gnr=0;
            for (int d = 0; d < hd; d++) {
                double a = 0;
                for (int k = 0; k < inter; k++)
                    a += (double)act[k] * (double)bf16f(Wd[(size_t)d * inter + k]);
                float ref = (float)a, npu = bf16f(down[(size_t)0 * hd + d]);
                if (npu < gmn) gmn = npu; if (npu > gmx) gmx = npu;
                float df = std::fabs(npu - ref); if (df > gmd) gmd = df;
                gdot += (double)npu * ref; gnn += (double)npu * npu; gnr += (double)ref * ref;
            }
            double gcos = (gnn > 0 && gnr > 0) ? gdot / (std::sqrt(gnn) * std::sqrt(gnr)) : 0;
            fprintf(stderr, "[hy3-fused-audit] L1 e=%d GEMM down: npu mn=%.4f mx=%.4f "
                    "maxdiff=%.4f cos=%.5f |npu|/|ref|=%.5f\n", e0, gmn, gmx, gmd, gcos,
                    (gnr > 0) ? std::sqrt(gnn / gnr) : 0); fflush(stderr);
        }

        /* (5) host readback + router-weighted accumulate.  Expert slot ls (eid e)
         * contributed down[ls*Mx + m, :]; weight w = wts[m*tk + j] where eids==e. */
        for (int ls = 0; ls < nbatch; ls++) {
            int e = slot_eids[batch_start + ls];
            for (int m = 0; m < M; m++) {
                float w = 0.0f; bool found = false;
                for (int j = 0; j < tk; j++) {
                    if (eids[m * tk + j] == e) { w = wts[m * tk + j]; found = true; break; }
                }
                if (!found || w == 0.0f) continue;
                const bf16_t* row = down + (size_t)(ls * Mx + m) * hd;
                for (int d = 0; d < hd; d++)
                    acc[(size_t)m * hd + d] += bf16f(row[d]) * w;
            }
        }
    }
}

/* ── Hunyuan-3.0 FFN (ARCH_HY3) ───────────────────────────────────────────
 * Host-first correctness implementation (phase 3).  ffn_norm RMSNorm, then:
 *   L0 (lid < first_k_dense_replace): dense SwiGLU on dense_gate/up/down (BF16).
 *   L1..L79: sigmoid router → top-8 routed experts (MXFP4, host dequant + GEMM)
 *            + 1 always-on shared expert (BF16).  ffn_out added to h (residual).
 * The NPU expert path (new 1536-inter kernels) is deferred; the dense + shared
 * BF16 paths move to NPU expert_gemm_vec in a later sub-phase. */
void FSTEngine::process_ffn_hy3(int lid, bf16_t* h, int M) {
    auto& w = hy3_shared_[lid];
    const int hd = config_.hidden_dim;
    const float eps = 1e-5f;

    /* 1. FFN input RMSNorm → hn [M, hd]. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hn((size_t)M * hd);
    rms_cpu(hn.data(), h, w.ffn_norm.data(), hd, M, eps);

    std::vector<float> ffn((size_t)M * hd, 0.0f);

    if (lid < config_.first_k_dense_replace) {
        /* Dense L0 SwiGLU (inter 13312) — no router, no shared expert. */
        swiglu_bf16_host(hn.data(), w.dense_gate.data(), w.dense_up.data(),
                         w.dense_down.data(), M, hd, config_.dense_inter_dim, ffn.data());
    } else {
        /* MoE: router → top-8 routed experts + always-on shared. */
        std::vector<int> eids((size_t)M * config_.top_k);
        std::vector<float> wts((size_t)M * config_.top_k);
        hy3_router_host(eids.data(), wts.data(), hn.data(), w.router.data(),
                        w.router_bias.data(), M, config_.n_experts, config_.top_k,
                        hd, config_.expert_weights_scale);
        /* Predictive prefetch (lever 1): queue this layer's experts + the next
         * `prefetch_ahead_` layers' (Markov: same expert ids) so the SSD worker
         * reads them while this layer's FFN runs on the NPU.  Without this,
         * every get() is a reactive miss that blocks ~83 ms on SSD.  Unique the
         * eids (M*top_k with dups across tokens) so the dedup scan stays small.
         * Gated on prefetch_ahead_>=1 so FST_PREFETCH_AHEAD=0 = old baseline. */
        if (pager_ && pager_->prefetch_ahead() >= 1) {
            std::vector<int> ueids(eids.begin(), eids.end());
            std::sort(ueids.begin(), ueids.end());
            ueids.erase(std::unique(ueids.begin(), ueids.end()), ueids.end());
            pager_->predict_and_prefetch(lid, ueids.data(), (int)ueids.size());
        }
        process_expert_ffn_hy3(lid, hn.data(), ffn.data(), M, eids.data(), wts.data());

        /* Always-on shared expert (BF16, inter 1536) — ungated, full strength. */
        std::vector<float> shared_out((size_t)M * hd, 0.0f);
        swiglu_bf16_host(hn.data(), w.shared_gate.data(), w.shared_up.data(),
                         w.shared_down.data(), M, hd, config_.expert_inter_dim, shared_out.data());
        for (size_t i = 0; i < (size_t)M * hd; i++) ffn[i] += shared_out[i];
    }

    /* 2. residual add (h += ffn). */
    for (size_t i = 0; i < (size_t)M * hd; i++)
        h[i] = f2bf(bf16f(h[i]) + ffn[i]);

    if (std::getenv("FST_HY3_LAYER_TIME")) {
        static double prev = now(), t0 = now();
        double t = now();
        fprintf(stderr, "[hy3-layer] L%d dt=%.1fms cum=%.0fms M=%d\n", lid,
                (t - prev) * 1000.0, (t - t0) * 1000.0, M); fflush(stderr);
        prev = t;
    }
}

/* ── Hunyuan-3.0 GQA attention (ARCH_HY3) ──────────────────────────────────
 * Host-first correctness implementation.  The heavy q/k/v/o proj GEMMs move
 * to NPU (qck for k/v, ob for q/o) in a later speed sub-phase; everything here
 * is fp32-accumulated host math so the YaRN RoPE / per-head norm / GQA / causal
 * softmax are verifiable against HF hy_v3 before going near the device.
 *
 *   h [M, hd] in/out (attention output ADDED — residual).
 *   seq_pos_ .. seq_pos_+M-1 are the M query positions; cache grows to S=seq_pos_+M.
 *
 * Math (HF hy_v3): RMSNorm(attn_norm) → q/k/v proj → per-head Q/K RMSNorm(d=128)
 * → YaRN RoPE (NeoX rotate-half, base 11158840, factor 4.0, orig 262144; mscale
 * folded into cos/sin) → append uncompressed 8×128 KV → QK·(1/√128) causal →
 * softmax → V → o proj → +residual.  eps=1e-5 (h->re). */
void FSTEngine::process_gqa(int lid, bf16_t* h, int M) {
    assert(seq_pos_ + M <= config_.max_seq);
    auto& w = hy3_shared_[lid];
    const int hd    = config_.hidden_dim;       /* 4096 */
    const int nq    = config_.num_q_heads;      /* 64   */
    const int nkv   = config_.num_kv_heads;     /* 8    */
    const int dh    = config_.head_dim;         /* 128  */
    const int half  = dh / 2;                    /* 64   */
    const int qrow  = nq * dh;                   /* 8192 */
    const int kvrow = nkv * dh;                  /* 1024 */
    const int S     = seq_pos_ + M;              /* cache rows after append */
    const float scale = 1.0f / std::sqrt((float)dh);   /* 1/√128 */
    const float eps = 1e-5f;                            /* HY3 rms_norm_eps (h->re) */
    const float base = rope_freq_base_;                 /* 11158840 */
    const float yarn_factor = config_.rope_scaling_factor; /* 4.0 */
    const int   yarn_orig    = config_.yarn_orig_ctx;       /* 262144 */

    /* YaRN inv_freq[64] — computed once per call (64 pow vs the proj GEMMs).
     * find_correction_dim(num_rot) = dh·ln(orig/(num_rot·2π)) / (2·ln(base)),
     * low=floor(corr(βfast=32)), high=ceil(corr(βslow=1)); ramp blends
     * extrapolation (low dim, no scaling) ↔ interpolation (high dim, ÷factor).
     * Matches HF hy_v3 YaRN (βfast/βslow default 32/1). */
    float inv_freq[64];
    {
        const float lcl = (dh * std::log((float)yarn_orig / (32.0f * 2.0f * (float)M_PI))) / (2.0f * std::log(base));
        const float hcl = (dh * std::log((float)yarn_orig / (1.0f  * 2.0f * (float)M_PI))) / (2.0f * std::log(base));
        int low  = (int)std::floor(lcl); if (low < 0) low = 0;
        int high = (int)std::ceil(hcl);  if (high > dh - 1) high = dh - 1;
        if (high <= low) high = low + 1;   /* linear_ramp min==max guard */
        for (int i = 0; i < half; i++) {
            float pos_freq = std::pow(base, (2.0f * (float)i) / (float)dh);
            float extrap = 1.0f / pos_freq;                 /* inv_freq_extrapolation */
            float interp = extrap / yarn_factor;            /* inv_freq_interpolation */
            float ramp = (float)(i - low) / (float)(high - low);
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            float extrap_factor = 1.0f - ramp;               /* 1@low dim, 0@high dim */
            inv_freq[i] = interp * (1.0f - extrap_factor) + extrap * extrap_factor;
        }
    }
    /* mscale folded into cos/sin (HF hy_v3: cos*=mscale, sin*=mscale). */
    const float mscale = (yarn_factor <= 1.0f) ? 1.0f
                        : 0.1f * std::log(yarn_factor) + 1.0f;   /* 1.1386 */

    /* 1. input RMSNorm → an [M, hd] bf16. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> an((size_t)M * hd);
    rms_cpu(an.data(), h, w.attn_norm.data(), hd, M, eps);

    /* 2. q/k/v projections (host fp32 GEMM; B native [N,K]=[out,in]). */
    std::vector<float> q((size_t)M * qrow), k((size_t)M * kvrow), v((size_t)M * kvrow);
    host_gemm_bnk_f32(q.data(), an.data(), w.q_proj.data(), M, nq * dh,  hd);
    host_gemm_bnk_f32(k.data(), an.data(), w.k_proj.data(), M, nkv * dh, hd);
    host_gemm_bnk_f32(v.data(), an.data(), w.v_proj.data(), M, nkv * dh, hd);

    /* 3. per-head Q/K RMSNorm (d=dh, over each head's 128 dims) in fp32.  q is
     * [M, nq, dh], k is [M, nkv, dh], row-major. */
    auto rmsnorm_rows = [&](float* x, const bf16_t* wg, int nrows) {
        for (int r = 0; r < nrows; r++) {
            float* xr = x + (size_t)r * dh;
            float ss = 0.0f;
            for (int d = 0; d < dh; d++) ss += xr[d] * xr[d];
            float rcp = 1.0f / std::sqrt(ss / (float)dh + eps);
            for (int d = 0; d < dh; d++) xr[d] = xr[d] * rcp * bf16f(wg[d]);
        }
    };
    rmsnorm_rows(q.data(), w.q_norm.data(), M * nq);
    rmsnorm_rows(k.data(), w.k_norm.data(), M * nkv);

    /* 4. YaRN RoPE (NeoX rotate-half) per head, per position (abs pos = seq_pos_+m). */
    auto rope_head = [&](float* x, int pos) {
        for (int j = 0; j < half; j++) {
            float ang = (float)pos * inv_freq[j];
            float cf = std::cos(ang) * mscale;
            float sf = std::sin(ang) * mscale;
            float a = x[j], b = x[j + half];
            x[j]        = a * cf - b * sf;     /* rotate_half = cat(-x2, x1) */
            x[j + half] = b * cf + a * sf;
        }
    };
    for (int m = 0; m < M; m++) {
        int pos = seq_pos_ + m;
        for (int hh = 0; hh < nq; hh++)  rope_head(q.data() + ((size_t)m * nq  + hh) * dh, pos);
        for (int hh = 0; hh < nkv; hh++) rope_head(k.data() + ((size_t)m * nkv + hh) * dh, pos);
    }

    /* 5. append the M new K/V rows to the uncompressed cache (v is NOT
     * normed/roped — only q and k are). */
    auto& kv = hy3_kv_cache_[lid];
    for (int m = 0; m < M; m++) {
        bf16_t* krow = kv.k.data() + (size_t)(seq_pos_ + m) * kvrow;
        bf16_t* vrow = kv.v.data() + (size_t)(seq_pos_ + m) * kvrow;
        const float* km = k.data() + (size_t)m * kvrow;
        const float* vm = v.data() + (size_t)m * kvrow;
        for (int i = 0; i < kvrow; i++) { krow[i] = f2bf(km[i]); vrow[i] = f2bf(vm[i]); }
    }
    kv.n = S;

    /* 6. GQA attention.  Each query head hq uses KV head hq/(nq/nkv).  Query at
     * abs pos qpos attends to keys 0..qpos (causal — just cap nkeys, no -inf
     * mask).  fp32 throughout; V read from the bf16 cache. */
    std::vector<float> out((size_t)M * qrow, 0.0f);
    std::vector<float> scores((size_t)S), prob((size_t)S);
    const int group = nq / nkv;   /* 8 Q heads share one KV head */
    for (int m = 0; m < M; m++) {
        const int qpos  = seq_pos_ + m;
        const int nkeys = qpos + 1;            /* causal */
        for (int hq = 0; hq < nq; hq++) {
            const int kvh = hq / group;
            const float* qh = q.data() + ((size_t)m * nq + hq) * dh;
            const bf16_t* kbase = kv.k.data() + (size_t)kvh * dh;   /* stride kvrow */
            float mx = -1e30f;
            for (int kk = 0; kk < nkeys; kk++) {
                const bf16_t* krow = kbase + (size_t)kk * kvrow;
                float s = 0.0f;
                for (int d = 0; d < dh; d++) s += qh[d] * bf16f(krow[d]);
                s *= scale;
                scores[kk] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (int kk = 0; kk < nkeys; kk++) { float p = std::exp(scores[kk] - mx); prob[kk] = p; sum += p; }
            float inv = 1.0f / (sum + 1e-20f);
            float* outh = out.data() + ((size_t)m * nq + hq) * dh;
            const bf16_t* vbase = kv.v.data() + (size_t)kvh * dh;   /* stride kvrow */
            for (int kk = 0; kk < nkeys; kk++) {
                float pv = prob[kk] * inv;
                const bf16_t* vrow = vbase + (size_t)kk * kvrow;
                for (int d = 0; d < dh; d++) outh[d] += pv * bf16f(vrow[d]);
            }
        }
    }

    /* 7. output projection (host fp32 GEMM; o_proj [hd, nq*dh]=[out=4096, in=8192]). */
    std::vector<float> o((size_t)M * hd, 0.0f);
    host_gemm_xnk_f32(o.data(), out.data(), w.o_proj.data(), M, hd, qrow);

    /* 8. residual add (h += o). */
    for (int m = 0; m < M; m++)
        for (int d = 0; d < hd; d++) {
            size_t idx = (size_t)m * hd + d;
            h[idx] = f2bf(bf16f(h[idx]) + o[idx]);
        }
}

/* ── Hunyuan-3.0 NextN MTP head (ARCH_HY3, blk.80) ──────────────────────────
 * Replaces the DSpark draft path.  Host-first correctness: the MTP block is a
 * full GQA+MoE decoder (hy3_shared_[n_layers-1] has attn/ffn/router/shared +
 * the 4 nextn_* tensors), so it reuses process_gqa/process_ffn_hy3 on lid 80.
 * The MTP keeps its OWN KV (hy3_kv_cache_[80]); seq_pos_ is driven per step so
 * RoPE position, KV append offset and causal length all track the MTP step.
 * KV seeding (trunk-KV copy) + global-vs-local position semantics are deferred
 * to the fork-validation step — this first cut is a self-contained predictor. */
int FSTEngine::forward_nextn_step(int last_token_id, const bf16_t* h_prev, int step,
                                   bf16_t* h_post, float* logits) {
    const int hd = config_.hidden_dim;
    const int V  = config_.vocab_size;
    const int mtp_lid = config_.n_layers - 1;          /* blk.80 */
    auto& w = hy3_shared_[mtp_lid];
    const float eps = 1e-5f;

    /* 1. e = embed(last_token_id); e_norm = rms(enorm, e); h_norm = rms(hnorm, h_prev). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> e(hd), e_norm(hd), h_norm(hd);
    std::memcpy(e.data(), host_embedding_table_.data() + (size_t)last_token_id * hd,
                (size_t)hd * sizeof(bf16_t));
    rms_cpu(e_norm.data(), e.data(),  w.nextn_enorm.data(), hd, 1, eps);
    rms_cpu(h_norm.data(), h_prev,    w.nextn_hnorm.data(), hd, 1, eps);

    /* 2. concat [e_norm; h_norm] [2*hd] → eh_proj [hd, 2*hd] → h_mtp [hd] (fp32→bf16). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> concat((size_t)2 * hd);
    std::memcpy(concat.data(),       e_norm.data(), (size_t)hd * sizeof(bf16_t));
    std::memcpy(concat.data() + hd,  h_norm.data(), (size_t)hd * sizeof(bf16_t));
    std::vector<float> hmtp(hd);
    host_gemm_bnk_f32(hmtp.data(), concat.data(), w.nextn_eh_proj.data(), 1, hd, 2 * hd);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> h_mtp(hd);
    for (int d = 0; d < hd; d++) h_mtp[d] = f2bf(hmtp[d]);

    /* 3. blk.80 forward (own KV; seq_pos_=step): GQA + MoE FFN. */
    seq_pos_ = step;
    process_gqa(mtp_lid, h_mtp.data(), 1);
    process_ffn_hy3(mtp_lid, h_mtp.data(), 1);

    /* 4. h_post = rms(shared_head_norm, h_mtp) — next step's h_prev (post-norm
     *    chaining, HY3 gotcha #6: chaining raw residuals gives wrong drafts). */
    rms_cpu(h_post, h_mtp.data(), w.nextn_shared_head_norm.data(), hd, 1, eps);

    /* 5. logits = lm_head @ h_post (host [V,hd] row-major); argmax → draft token. */
    int amx = 0; float mx = -1e30f;
    for (int v = 0; v < V; v++) {
        const bf16_t* row = host_lm_head_.data() + (size_t)v * hd;
        float l = 0.0f;
        for (int d = 0; d < hd; d++) l += bf16f(row[d]) * bf16f(h_post[d]);
        logits[v] = l;
        if (l > mx) { mx = l; amx = v; }
    }
    return amx;
}

/* Iterative NextN drafting up to n_max steps with a p_min confidence cutoff.
 * Chains h_post → h_prev across steps (post-norm).  Returns the draft tokens +
 * greedy-probability confidences.  The MTP KV is reset per draft session. */
FSTEngine::DraftResult FSTEngine::forward_nextn_draft(const bf16_t* h_prev, int last_token_id,
                                                     int n_max, float p_min) {
    const int hd = config_.hidden_dim;
    const int V  = config_.vocab_size;
    DraftResult res;
    res.tokens.reserve(n_max);
    res.confidence.reserve(n_max);

    /* MTP own KV starts fresh for this draft session (separate from trunk KV). */
    hy3_kv_cache_[config_.n_layers - 1].n = 0;

    std::vector<bf16_t, AlignedAllocator<bf16_t>> h_cur(hd), h_post(hd);
    std::memcpy(h_cur.data(), h_prev, (size_t)hd * sizeof(bf16_t));
    std::vector<float> logits(V);
    int last_id = last_token_id;

    for (int k = 0; k < n_max; k++) {
        int draft = forward_nextn_step(last_id, h_cur.data(), k, h_post.data(), logits.data());
        /* confidence = softmax(logits)[draft] (greedy probability). */
        float mx = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > mx) mx = logits[v];
        double sum = 0.0;
        for (int v = 0; v < V; v++) sum += std::exp((double)(logits[v] - mx));
        float conf = (float)(std::exp((double)(logits[draft] - mx)) / sum);
        if (k > 0 && conf < p_min) break;          /* p_min cutoff (exact rule is fork-specific) */
        res.tokens.push_back(draft);
        res.confidence.push_back(conf);
        std::memcpy(h_cur.data(), h_post.data(), (size_t)hd * sizeof(bf16_t));  /* chain */
        last_id = draft;
    }
    return res;
}

void FSTEngine::process_mla(int lid, bf16_t* h, int M) {
    /* DeepSeek V4 Flash latent MLA — FULLY vectorized NPU GEMM path
     * (canonical kernels.mm + zero, 100% aie::mmul, ZERO scalar, ZERO CPU
     * fallback).  4 MLA xclbins: qc (qc/oa/kvc, b_col_maj [N,K]), wqb (wq_b
     * N-tiled), ob (ob N-tiled), qksv (qk/sv, b_row_maj [K,N]).  Weights are
     * kept in native [N,K] (no transpose — matches b_col_maj).  RoPE/softmax/
     * norms stay on host (tiny tensors, not the bottleneck). */
    assert(seq_pos_ + M <= config_.max_seq);
    const bool dbg = audit_on && (lid == 0 || lid == 42);
    if (audit_on) { fprintf(stderr, "[diag] process_mla L%d M=%d M_LAT=%d\n", lid, M, ((M+15)/16)*16*MLA_N_HEADS); fflush(stderr); }
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_MLA_TRACE")) {
        auto dump_bf = [](const char* path, const bf16_t* p, size_t n){
            std::vector<float> tmp(n);
            for (size_t i=0;i<n;i++) tmp[i]=bf16f(p[i]);
            FILE* f=std::fopen(path,"wb");
            if(f){std::fwrite(tmp.data(),sizeof(float),n,f);std::fclose(f);}
            float mx=0;for(float v:tmp)mx=std::max(mx,std::fabs(v));
            fprintf(stderr,"[mla-trace] dump %s n=%zu |mx=%.4f\n",path,n,mx);fflush(stderr);
        };
        dump_bf("/tmp/eng_L00_mla_in.f32", h, (size_t)M*config_.hidden_dim);   /* attn-normed input to wq_a/wkv */
    }
    auto& w = shared_[lid];
    const int hd = config_.hidden_dim, rope_dim = config_.rope_dim;
    const int n_heads = MLA_N_HEADS, head_dim = MLA_HEAD_DIM;
    const int nope_dim = head_dim - rope_dim;
    const int S = seq_pos_ + M;
    const int M_PAD = ((M + 15) / 16) * 16;          /* multiple of 16 (vec M-tile) */
    const int M_LAT = M_PAD * n_heads;               /* qk/sv M (<=2048 for M_PAD<=32) */

    /* h -> bo_mla_h_ [M_PAD, hd] (zero-pad rows M..M_PAD). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> h_pad(M_PAD * hd, 0);
    memcpy(h_pad.data(), h, (size_t)M * hd * sizeof(bf16_t));
    {
        const size_t h_sz = (size_t)M_PAD * hd * sizeof(bf16_t);
        if (bo_mla_h_.size() < h_sz) bo_mla_h_ = xrt::ext::bo(npu_device_, h_sz);
        memcpy(bo_mla_h_.map<char*>(), h_pad.data(), h_sz);
        npu_sync_to(bo_mla_h_);
    }

    /* qc: h[M_PAD,4096] @ wq_a[N=1024,K=4096] -> qc[M_PAD,1024]  (b_col_maj)
     * Persistent-BO path: wq_a is device-resident (bo_wq_a), K_c==K==4096 (qck)
     * -> contiguous sub-buffer, no per-dispatch weight re-sync. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> qc_host(M_PAD * MLA_Q_LORA, 0);
    npu_gemm_mla_vec("qck", qc_host.data(), h_pad.data(), nullptr,
                     M_PAD, MLA_Q_LORA, hd, &w.bo_wq_a, 0);
    /* kvc: h[M_PAD,4096] @ wkv[N=512,K=4096] -> kv[M_PAD,512]  (b_col_maj) */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kv_host(M_PAD * MLA_KV_LORA, 0);
    npu_gemm_mla_vec("qck", kv_host.data(), h_pad.data(), nullptr,
                     M_PAD, MLA_KV_LORA, hd, &w.bo_wkv, 0);

    if (dbg) {
        /* wkv audit (correct [N,K] native indexing): ref[m,n]=sum_k h[m,k]*wkv[n,k]. */
        int wr=0, wn=0; float wa=0;
        for (int m=0;m<M;m++) for (int n=0;n<MLA_KV_LORA;n++){
            float a=std::fabs(bf16f(kv_host[(size_t)m*MLA_KV_LORA+n])); if(a>wa){wa=a;wr=m;wn=n;}}
        double acc=0;
        for (int k=0;k<hd;k++)
            acc += (double)bf16f(h_pad[(size_t)wr*hd+k]) * (double)bf16f(w.wkv[(size_t)wn*hd+k]);
        fprintf(stderr, "[wkv-audit] L%d worst m=%d n=%d: npu=%.4f ref=%.4f ratio=%.3f\n",
                lid, wr, wn, bf16f(kv_host[(size_t)wr*MLA_KV_LORA+wn]), (float)acc,
                acc!=0 ? bf16f(kv_host[(size_t)wr*MLA_KV_LORA+wn])/(float)acc : 0.0f);
        fflush(stderr);
    }

    /* Per-lora RMSNorm (q_norm / kv_norm) on the REAL M rows. */
    if (!w.q_norm.empty())
        for (int m = 0; m < M; m++)
            npu_rmsnorm_weighted(qc_host.data() + m * MLA_Q_LORA,
                                 qc_host.data() + m * MLA_Q_LORA,
                                 w.q_norm.data(), MLA_Q_LORA);
    if (!w.kv_norm.empty())
        for (int m = 0; m < M; m++)
            npu_rmsnorm_weighted(kv_host.data() + m * MLA_KV_LORA,
                                 kv_host.data() + m * MLA_KV_LORA,
                                 w.kv_norm.data(), MLA_KV_LORA);

    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_MLA_TRACE")) {
        auto dump_bf = [](const char* path, const bf16_t* p, size_t n){
            std::vector<float> tmp(n);
            for (size_t i=0;i<n;i++) tmp[i]=bf16f(p[i]);
            FILE* f=std::fopen(path,"wb");
            if(f){std::fwrite(tmp.data(),sizeof(float),n,f);std::fclose(f);}
            float mx=0;for(float v:tmp)mx=std::max(mx,std::fabs(v));
            fprintf(stderr,"[mla-trace] dump %s n=%zu |mx=%.4f\n",path,n,mx);fflush(stderr);
        };
        dump_bf("/tmp/eng_L00_qc.f32", qc_host.data(), (size_t)M*MLA_Q_LORA);   /* post wq_a + q_norm */
        dump_bf("/tmp/eng_L00_kvc.f32", kv_host.data(), (size_t)M*MLA_KV_LORA); /* post wkv + kv_norm (pre-rope) */
    }

    /* wq_b: qc[M_PAD,1024] @ wq_b[N=32768,K=1024] -> q_full[M_PAD,32768]
     * N-tiled into 16 calls of N=2048 (wqb kernel).  wq_b is [N,K] native so
     * each N-tile is a contiguous [2048,1024] block at &wq_b[n_off*1024]. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> q_full(M_PAD * n_heads * head_dim, 0);
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> q_pad(M_PAD * MLA_Q_LORA, 0);
        memcpy(q_pad.data(), qc_host.data(), (size_t)M * MLA_Q_LORA * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> q_tile(M_PAD * 2048);
        const int QN = n_heads * head_dim;   /* 32768 */
        const int WQB_NTILES = QN / 2048;    /* 16 */
        /* FST_PACK_WQB: collapse the 16 N-tile dispatches into ONE run_blob by
         * replicating the wqb op-words 16× in one instruction blob, each copy's
         * weight (bd1) and output (bd2) DDR_PATCH arg_off advanced to that
         * N-tile's offset.  Proven byte-identical to 16 single dispatches
         * (tools/wqb_pack_probe).  Decode-only path (M_PAD==16, one M-tile). */
        if (std::getenv("FST_PACK_WQB") && M_PAD == 16) {
            if (wqb_packed_blob_.empty()) {
                auto base = load_insts_words(kpath("fst_mla_wqb_insts.bin").c_str());
                std::vector<uint32_t> w_offs(WQB_NTILES), o_offs(WQB_NTILES);
                for (int n = 0; n < WQB_NTILES; ++n) {
                    w_offs[n] = (uint32_t)((size_t)n * 2048 * MLA_Q_LORA * sizeof(bf16_t));
                    o_offs[n] = (uint32_t)((size_t)n * 16 * 2048 * sizeof(bf16_t));
                }
                wqb_packed_blob_ = replicate_packed_blob(base, WQB_NTILES, {}, w_offs, o_offs);
                fprintf(stderr, "[pack] wqb 16->1 blob built: %zu words\n", wqb_packed_blob_.size());
            }
            const size_t a_tile_b = (size_t)16 * MLA_Q_LORA * sizeof(bf16_t);
            const size_t c_tile_b = (size_t)16 * 2048 * sizeof(bf16_t);
            if (bo_mla_atile_.size() < a_tile_b)
                bo_mla_atile_ = xrt::ext::bo(npu_device_, a_tile_b);
            if (bo_wqb_pack_out_.size() < (size_t)WQB_NTILES * c_tile_b)
                bo_wqb_pack_out_ = xrt::bo(npu_device_, (size_t)WQB_NTILES * c_tile_b,
                                           xrt::bo::flags::host_only, grp_);
            std::memcpy(bo_mla_atile_.map<char*>(), q_pad.data(), a_tile_b);
            npu_sync_to(bo_mla_atile_);
            {auto* c = bo_wqb_pack_out_.map<bf16_t*>();
             std::memset(c, 0, (size_t)WQB_NTILES * c_tile_b);}
            npu_sync_to(bo_wqb_pack_out_);
            xrt::bo* args[5] = {
                &static_cast<xrt::bo&>(bo_mla_atile_),
                &static_cast<xrt::bo&>(w.bo_wq_b),
                &static_cast<xrt::bo&>(bo_wqb_pack_out_),
                &bo_d1_, &bo_d2_ };
            auto run = kernel_cache_->run_blob(
                kernel_cache_->entry_xclbin_path("wqb"), wqb_packed_blob_, args, 5);
            run.wait();
            npu_sync_from(bo_wqb_pack_out_);
            auto* c = bo_wqb_pack_out_.map<bf16_t*>();
            for (int n = 0; n < WQB_NTILES; ++n) {
                const bf16_t* tile = (const bf16_t*)((const char*)c + (size_t)n * c_tile_b);
                for (int m = 0; m < M_PAD; m++)
                    memcpy(q_full.data() + (size_t)m * QN + n * 2048,
                           tile + (size_t)m * 2048, 2048 * sizeof(bf16_t));
            }
        } else {
        for (int n_off = 0; n_off < QN; n_off += 2048) {
            npu_gemm_mla_vec("wqb", q_tile.data(), q_pad.data(), nullptr,
                             M_PAD, 2048, MLA_Q_LORA,
                             &w.bo_wq_b, (size_t)n_off * MLA_Q_LORA * sizeof(bf16_t));
            for (int m = 0; m < M_PAD; m++)
                memcpy(q_full.data() + (size_t)m * QN + n_off,
                       q_tile.data() + (size_t)m * 2048,
                       2048 * sizeof(bf16_t));
        }
        }
        if (dbg) {
            int wr=0, wn=0; float wa=0;
            for (int m=0;m<M;m++) for (int n=0;n<QN;n++){
                float a=std::fabs(bf16f(q_full[(size_t)m*QN+n])); if(a>wa){wa=a;wr=m;wn=n;}}
            double acc=0;
            for (int k=0;k<MLA_Q_LORA;k++)
                acc += (double)bf16f(q_pad[(size_t)wr*MLA_Q_LORA+k])
                     * (double)bf16f(w.wq_b[(size_t)wn*MLA_Q_LORA+k]);  /* [N,K]: wq_b[wn,k] */
            fprintf(stderr, "[wq_b-audit] L%d worst m=%d n=%d: npu=%.4f ref=%.4f ratio=%.3f\n",
                    lid, wr, wn, bf16f(q_full[(size_t)wr*QN+wn]), (float)acc,
                    acc!=0 ? bf16f(q_full[(size_t)wr*QN+wn])/(float)acc : 0.0f);
            fflush(stderr);
        }
        /* Per-head RMSNorm on expanded q (ds4.c head_rms_norm_inplace). */
        for (int m = 0; m < M; m++)
            for (int hh = 0; hh < n_heads; hh++) {
                bf16_t* head = q_full.data() + (size_t)m * QN + hh * head_dim;
                double ss = 0.0;
                for (int i = 0; i < head_dim; i++) ss += (double)bf16f(head[i]) * bf16f(head[i]);
                float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + 1e-6f);
                for (int i = 0; i < head_dim; i++) head[i] = f2bf(bf16f(head[i]) * scale);
            }
    }

    /* RoPE on q_pe: rotate the rope_dim tail of EVERY head (ds4.c model.py:505
     * applies apply_rotary_emb to q[..., -rd:] for all n_heads).  q_full is
     * [M_PAD, n_heads, head_dim] row-major == [M_PAD*n_heads, head_dim]; the
     * helper gathers/scatters all M*n_heads real rows' tails (the prior code
     * only touched head 0, leaving heads 1..63 un-RoPE'd). */
    rope_qpe_all_heads(q_full.data(), M, lid, /*inverse=*/false);
    /* RoPE on k_pe (the rope_dim tail of kv_latent). */
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> k_pe(M * rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(k_pe.data() + m * rope_dim,
                   kv_host.data() + m * MLA_KV_LORA + nope_dim,
                   rope_dim * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> lut(M * rope_dim);
        build_rope_lut(lut.data(), M, lid);
        npu_rope("rope", k_pe.data(), lut.data(), k_pe.data(), M, rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(kv_host.data() + m * MLA_KV_LORA + nope_dim,
                   k_pe.data() + m * rope_dim, rope_dim * sizeof(bf16_t));
    }

    /* FST_MLA_TRACE: dump q (post-RoPE) [M*64,512] and kv (post-RoPE) [M,512]
     * for L0 prefill, to bisect the MLA output direction error vs HF. */
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_MLA_TRACE")) {
        auto dump_bf = [](const char* path, const bf16_t* p, size_t n){
            std::vector<float> tmp(n);
            for (size_t i=0;i<n;i++) tmp[i]=bf16f(p[i]);
            FILE* f=std::fopen(path,"wb");
            if(f){std::fwrite(tmp.data(),sizeof(float),n,f);std::fclose(f);}
            float mx=0;for(float v:tmp)mx=std::max(mx,std::fabs(v));
            fprintf(stderr,"[mla-trace] dump %s n=%zu |mx=%.4f\n",path,n,mx);fflush(stderr);
        };
        dump_bf("/tmp/eng_L00_q.f32", q_full.data(), (size_t)M*n_heads*head_dim);
        dump_bf("/tmp/eng_L00_kv.f32", kv_host.data(), (size_t)M*MLA_KV_LORA);
    }

    /* KV cache push (real M rows). */
    auto& kvc = kv_cache_[lid];
    memcpy(kvc.kv_latent.data() + seq_pos_ * MLA_KV_LORA, kv_host.data(),
           (size_t)M * MLA_KV_LORA * sizeof(bf16_t));
    {
        auto* dst = kvc.kv_latent_bo.map<bf16_t*>();
        memcpy(dst + seq_pos_ * MLA_KV_LORA, kv_host.data(),
               (size_t)M * MLA_KV_LORA * sizeof(bf16_t));
        kvc.kv_latent_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                              kvc.kv_latent.size() * sizeof(bf16_t), 0);
    }

    /* V4 KV compressor: stream each of the M tokens (attn-normed hidden) through
     * the per-layer compressor so the compressed KV cache is populated up to the
     * last boundary <= seq_pos_+M-1 BEFORE the mixed-attention qk GEMM.  ds4.c
     * ordering: push raw kv, then compressor_decode_one, then mixed attention.
     * Only ratio=4 runs here: ratio=128's first boundary is at pos 127, never
     * reached for S<128 (max_seq), so its cache stays empty and mixed attention
     * degenerates to plain MLA (Stot=S) — no work, matches the plan.  ratio=0
     * layers also skip (n_comp=0).
     *
     * GATED OFF for short prompts: the compressor is faithful to ds4.c (comp
     * rows bounded O(1-4), scores match) but, for S<=34, the compressed rows
     * carry scores comparable to the raw SWA rows and excite the HC 4-stream
     * recurrence's unstable mode — HC grows ±40 (Phase 4) -> ±1632 (with
     * compressor), and output degrades.  The raw 128-row SWA ring already holds
     * every token for S<128, so the compressor buys nothing here and costs 2
     * extra GEMMs/ratio-4-layer + host control.  Re-enable (drop the env guard)
     * for Phase 6 long-context generation (>128 tokens), where the compressor is
     * actually load-bearing.  n_comp stays 0 below -> Stot=S -> mixed attention
     * degenerates exactly to the Phase 4 plain-MLA path. */
    if (w.cmp_ratio == 4 && std::getenv("FST_COMPRESSOR") != nullptr) {
        for (int m = 0; m < M; m++)
            compress_token(lid, h + (size_t)m * hd, seq_pos_ + m);
    }
    auto& cst = cmp_state_[lid];
    const int n_comp = cst.n_comp;
    const int Stot = S + n_comp;   /* total attention columns: raw + compressed */

    /* qk: q_flat[M_LAT,512] @ kvT_concat[512, Stot] -> scores[M_LAT, Stot]
     * (b_row_maj; N-pad Stot->512).  kvT_concat = raw kv^T [512, S] with the
     * compressed rows appended as extra columns [512, n_comp] (comp row r is the
     * SAME 512-vector across all heads).  NO scale here. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> q_flat(M_LAT * head_dim, 0);
    memcpy(q_flat.data(), q_full.data(),
           (size_t)M * n_heads * head_dim * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kvT((size_t)head_dim * Stot, 0);
    for (int s = 0; s < S; s++)
        for (int k = 0; k < head_dim; k++)
            kvT[(size_t)k * Stot + s] = kvc.kv_latent[(size_t)s * head_dim + k];
    for (int c = 0; c < n_comp; c++)
        for (int k = 0; k < head_dim; k++)
            kvT[(size_t)k * Stot + S + c] = cst.cache[(size_t)c * head_dim + k];
    std::vector<bf16_t, AlignedAllocator<bf16_t>> qk_out((size_t)M_LAT * Stot, 0);
    npu_gemm_mla_vec("qksv", qk_out.data(), q_flat.data(), kvT.data(),
                     M_LAT, Stot, head_dim);
    if (dbg) {
        int worst_r=0; float worst_abs=0;
        for (int r=0;r<M*n_heads;r++) for (int s=0;s<Stot;s++){
            float a=std::fabs(bf16f(qk_out[(size_t)r*Stot+s])); if(a>worst_abs){worst_abs=a;worst_r=r;}}
        int ws=0; float ws_abs=0;
        for (int s=0;s<Stot;s++){float a=std::fabs(bf16f(qk_out[(size_t)worst_r*Stot+s])); if(a>ws_abs){ws_abs=a;ws=s;}}
        double acc_k=0;
        const bf16_t* kref = (ws < S) ? (kvc.kv_latent.data() + (size_t)ws*head_dim)
                                     : (cst.cache.data() + (size_t)(ws-S)*head_dim);
        for (int d=0;d<head_dim;d++)
            acc_k += (double)bf16f(q_flat[(size_t)worst_r*head_dim+d]) * (double)bf16f(kref[d]);
        fprintf(stderr, "[qk-audit] L%d worst_r=%d ws=%d: npu=%.4f ref(q@k)=%.4f ratio=%.3f (S=%d n_comp=%d)\n",
                lid, worst_r, ws, bf16f(qk_out[(size_t)worst_r*Stot+ws]), (float)acc_k,
                acc_k!=0 ? bf16f(qk_out[(size_t)worst_r*Stot+ws])/(float)acc_k : 0.0f, S, n_comp);
        fflush(stderr);
    }

    /* Repack scores into sc [M_LAT, 512] (cols 0..Stot-1 real, rest 0). */
    constexpr int SC_STRIDE = 512;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> sc((size_t)M_LAT * SC_STRIDE, 0);
    for (int r = 0; r < M_LAT; r++)
        for (int s = 0; s < Stot; s++)
            sc[(size_t)r * SC_STRIDE + s] = qk_out[(size_t)r * Stot + s];

    /* Softmax with causal mask + per-head sink + scale 1/sqrt(head_dim).
     * Raw columns s in [0,S): causal mask s <= seq_pos_+m (prefill only).
     * Compressed columns c in [0,n_comp): emitted at abs pos q_c=(c+1)*ratio-1,
     * allowed for query m iff q_c <= seq_pos_+m (prefill); decode (M=1) allows all. */
    {
        const float kq_scale = 1.0f / sqrtf((float)head_dim);
        for (int m = 0; m < M; m++)
            for (int hh = 0; hh < n_heads; hh++) {
                int row = m * n_heads + hh;
                const float sink = w.attn_sinks.empty() ? 0.0f : w.attn_sinks[hh];
                float max_score = sink;
                for (int s = 0; s < S; s++) {
                    if (is_prefill_ && s > seq_pos_ + m) continue;
                    float v = bf16f(sc[(size_t)row * SC_STRIDE + s]) * kq_scale;
                    if (v > max_score) max_score = v;
                }
                for (int c = 0; c < n_comp; c++) {
                    int q_c = (c + 1) * w.cmp_ratio - 1;   /* abs pos of comp row c */
                    if (is_prefill_ && q_c > seq_pos_ + m) continue;
                    float v = bf16f(sc[(size_t)row * SC_STRIDE + S + c]) * kq_scale;
                    if (v > max_score) max_score = v;
                }
                float sum = expf(sink - max_score);
                for (int s = 0; s < S; s++) {
                    if (is_prefill_ && s > seq_pos_ + m) continue;
                    sum += expf(bf16f(sc[(size_t)row * SC_STRIDE + s]) * kq_scale - max_score);
                }
                for (int c = 0; c < n_comp; c++) {
                    int q_c = (c + 1) * w.cmp_ratio - 1;
                    if (is_prefill_ && q_c > seq_pos_ + m) continue;
                    sum += expf(bf16f(sc[(size_t)row * SC_STRIDE + S + c]) * kq_scale - max_score);
                }
                float inv = 1.0f / (sum + 1e-12f);
                for (int s = 0; s < S; s++) {
                    if (is_prefill_ && s > seq_pos_ + m)
                        sc[(size_t)row * SC_STRIDE + s] = f2bf(0.0f);
                    else
                        sc[(size_t)row * SC_STRIDE + s] =
                            f2bf(expf(bf16f(sc[(size_t)row * SC_STRIDE + s]) * kq_scale - max_score) * inv);
                }
                for (int c = 0; c < n_comp; c++) {
                    int q_c = (c + 1) * w.cmp_ratio - 1;
                    if (is_prefill_ && q_c > seq_pos_ + m)
                        sc[(size_t)row * SC_STRIDE + S + c] = f2bf(0.0f);
                    else
                        sc[(size_t)row * SC_STRIDE + S + c] =
                            f2bf(expf(bf16f(sc[(size_t)row * SC_STRIDE + S + c]) * kq_scale - max_score) * inv);
                }
            }
        for (int r = 0; r < M_LAT; r++)
            memset(sc.data() + (size_t)r * SC_STRIDE + Stot, 0,
                   (size_t)(SC_STRIDE - Stot) * sizeof(bf16_t));
    }

    /* sv: scores[M_LAT, Stot] @ kv_concat[Stot, 512] -> out[M_LAT, 512]
     * (b_row_maj; K-pad Stot->512).  kv_concat = kv_latent[S,512] ++ cache[n_comp,512]
     * (row-major: row s<S = kv_latent[s], row S+c = cache[c]). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> scores_packed((size_t)M_LAT * Stot, 0);
    for (int r = 0; r < M_LAT; r++)
        memcpy(scores_packed.data() + (size_t)r * Stot,
               sc.data() + (size_t)r * SC_STRIDE,
               (size_t)Stot * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kv_concat((size_t)Stot * head_dim, 0);
    memcpy(kv_concat.data(), kvc.kv_latent.data(), (size_t)S * head_dim * sizeof(bf16_t));
    if (n_comp > 0)
        memcpy(kv_concat.data() + (size_t)S * head_dim, cst.cache.data(),
               (size_t)n_comp * head_dim * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> out_pad(M_LAT * head_dim, 0);
    npu_gemm_mla_vec("qksv", out_pad.data(), scores_packed.data(),
                     kv_concat.data(), M_LAT, head_dim, Stot);
    if(audit_on){float mx=-1e30f,mn=1e30f; int nan=0; for(int i=0;i<M*n_heads*head_dim;i++){float v=bf16f(out_pad[i]); if(std::isnan(v)||std::isinf(v))nan++; else{if(v>mx)mx=v; if(v<mn)mn=v;}} fprintf(stderr,"[dbg] L%d sv-prerope: mn=%.3f mx=%.3f nan=%d\n",lid,mn,mx,nan); fflush(stderr);}
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_MLA_TRACE")) {
        auto dump_bf = [](const char* path, const bf16_t* p, size_t n){
            std::vector<float> tmp(n);
            for (size_t i=0;i<n;i++) tmp[i]=bf16f(p[i]);
            FILE* f=std::fopen(path,"wb");
            if(f){std::fwrite(tmp.data(),sizeof(float),n,f);std::fclose(f);}
            float mx=0;for(float v:tmp)mx=std::max(mx,std::fabs(v));
            fprintf(stderr,"[mla-trace] dump %s n=%zu |mx=%.4f\n",path,n,mx);fflush(stderr);
        };
        dump_bf("/tmp/eng_L00_svpre.f32", out_pad.data(), (size_t)M*n_heads*head_dim);
    }
    if (dbg) {
        int worst_r=0; float worst_abs=0;
        for (int r=0;r<M*n_heads;r++) for (int n=0;n<head_dim;n++){
            float a=std::fabs(bf16f(out_pad[(size_t)r*head_dim+n])); if(a>worst_abs){worst_abs=a;worst_r=r;}}
        int wn=0; float wn_abs=0; float maxdiff=0, ref_mx=0, npu_mx=0;
        for (int n=0;n<head_dim;n++){
            double acc=0; for (int s=0;s<Stot;s++) acc+=(double)bf16f(scores_packed[(size_t)worst_r*Stot+s])*(double)bf16f(kv_concat[(size_t)s*head_dim+n]);
            float ref=(float)acc, npu=bf16f(out_pad[(size_t)worst_r*head_dim+n]);
            if(std::fabs(ref)>ref_mx)ref_mx=std::fabs(ref); if(std::fabs(npu)>npu_mx)npu_mx=std::fabs(npu);
            if(std::fabs(npu-ref)>maxdiff)maxdiff=std::fabs(npu-ref);
            if(std::fabs(npu)>wn_abs){wn_abs=std::fabs(npu);wn=n;}}
        fprintf(stderr,"[sv-audit] L%d worst_r=%d: |npu|<=%.3f |ref|<=%.3f maxdiff=%.3f worst_col=%d\n",
                lid, worst_r, npu_mx, ref_mx, maxdiff, wn);
        fflush(stderr);
    }

    /* Inverse RoPE on the sv output's pe tail of EVERY head (ds4.c sin_sign=-1).
     * out_pad is [M_LAT, head_dim] == [M_PAD*n_heads, head_dim] row-major; the
     * helper inverse-rotates all M*n_heads real rows' tails (prior code only
     * inverse-rotated head 0, leaving heads 1..63 forward-rotated). */
    rope_qpe_all_heads(out_pad.data(), M, lid, /*inverse=*/true);

    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_MLA_TRACE")) {
        auto dump_bf = [](const char* path, const bf16_t* p, size_t n){
            std::vector<float> tmp(n);
            for (size_t i=0;i<n;i++) tmp[i]=bf16f(p[i]);
            FILE* f=std::fopen(path,"wb");
            if(f){std::fwrite(tmp.data(),sizeof(float),n,f);std::fclose(f);}
            float mx=0;for(float v:tmp)mx=std::max(mx,std::fabs(v));
            fprintf(stderr,"[mla-trace] dump %s n=%zu |mx=%.4f\n",path,n,mx);fflush(stderr);
        };
        dump_bf("/tmp/eng_L00_svpost.f32", out_pad.data(), (size_t)M*n_heads*head_dim);
    }

    /* heads = out_pad reinterpreted as [M_PAD, n_heads*head_dim] = [M_PAD, 32768]
     * (M_LAT=M_PAD*64, head_dim=512 -> [M_LAT,512]row-maj == [M_PAD,32768]row-maj). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> heads_pad(M_LAT * head_dim, 0);
    memcpy(heads_pad.data(), out_pad.data(), (size_t)M_LAT * head_dim * sizeof(bf16_t));
    if(audit_on){float mx=-1e30f,mn=1e30f; int nan=0; for(int i=0;i<M*n_heads*head_dim;i++){float v=bf16f(heads_pad[i]); if(std::isnan(v)||std::isinf(v))nan++; else{if(v>mx)mx=v; if(v<mn)mn=v;}} fprintf(stderr,"[dbg] L%d heads(sv): mn=%.3f mx=%.3f nan=%d\n",lid,mn,mx,nan); fflush(stderr);}

    /* OA: 8 grouped GEMMs.  heads_g[m,k] = heads_pad[m, g*4096+k] (strided, stride 32768).
     * wo_a is [N=8192,K=4096] native -> group g's B is a CONTIGUOUS [1024,4096] block
     * at &wo_a[g*1024*4096] (no strided B extraction — the transpose removal made groups
     * contiguous).  low_g[m,n] = heads_g @ wo_a_g  (b_col_maj).  Scatter into low[M_PAD,8192]. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> low(M_PAD * 8192, 0);
    const int heads_cols = n_heads * head_dim;   /* 32768 */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> heads_g(M_PAD * 4096, 0);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> low_g(M_PAD * 1024, 0);
    /* FST_PACK_OA: collapse the 8 grouped O-projection dispatches into ONE
     * run_blob by replicating the qck op-words 8x in one instruction blob, each
     * copy's bd0 (A=heads_g, arg_idx=0), bd1 (B=wo_a, arg_idx=1) and bd2
     * (C=low_g, arg_idx=2) DDR_PATCH arg_off advanced to that group's slice.
     * Decode-only path (M_PAD==16); each group is exactly one qck dispatch
     * (K_c==K==4096, single chunk, single M/N-tile).  Unlike wqb the A matrix
     * differs per group, so bd0 is patched too.  Proven mechanism: wqb 16->1
     * (tools/wqb_pack_probe) byte-identical; OA adds the bd0 arg_off patch. */
    if (std::getenv("FST_PACK_OA") && M_PAD == 16) {
        const int OA_NG = 8;
        const size_t a_tile_b = (size_t)16 * 4096 * sizeof(bf16_t);   /* 128 KiB */
        const size_t c_tile_b = (size_t)16 * 1024 * sizeof(bf16_t);   /* 32 KiB */
        if (oa_packed_blob_.empty()) {
            auto base = load_insts_words(kpath("fst_mla_qck_insts.bin").c_str());
            std::vector<uint32_t> a_offs(OA_NG), w_offs(OA_NG), o_offs(OA_NG);
            for (int g = 0; g < OA_NG; ++g) {
                a_offs[g] = (uint32_t)((size_t)g * a_tile_b);
                w_offs[g] = (uint32_t)((size_t)g * 1024 * 4096 * sizeof(bf16_t));
                o_offs[g] = (uint32_t)((size_t)g * c_tile_b);
            }
            oa_packed_blob_ = replicate_packed_blob(base, OA_NG, a_offs, w_offs, o_offs);
            fprintf(stderr, "[pack] oa 8->1 blob built: %zu words\n", oa_packed_blob_.size());
        }
        if (bo_oa_pack_a_.size() < (size_t)OA_NG * a_tile_b)
            bo_oa_pack_a_ = xrt::bo(npu_device_, (size_t)OA_NG * a_tile_b,
                                    xrt::bo::flags::host_only, grp_);
        if (bo_oa_pack_out_.size() < (size_t)OA_NG * c_tile_b)
            bo_oa_pack_out_ = xrt::bo(npu_device_, (size_t)OA_NG * c_tile_b,
                                      xrt::bo::flags::host_only, grp_);
        /* Pack the 8 per-group A slices (heads_pad[m, g*4096 : g*4096+4096])
         * contiguously into bo_oa_pack_a_. */
        auto* a = bo_oa_pack_a_.map<bf16_t*>();
        for (int g = 0; g < OA_NG; ++g)
            for (int m = 0; m < M_PAD; ++m)
                memcpy(a + (size_t)g * (16 * 4096) + (size_t)m * 4096,
                       heads_pad.data() + (size_t)m * heads_cols + g * 4096,
                       4096 * sizeof(bf16_t));
        npu_sync_to(bo_oa_pack_a_);
        {auto* c = bo_oa_pack_out_.map<bf16_t*>();
         std::memset(c, 0, (size_t)OA_NG * c_tile_b);}
        npu_sync_to(bo_oa_pack_out_);
        xrt::bo* args[5] = {
            &static_cast<xrt::bo&>(bo_oa_pack_a_),
            &static_cast<xrt::bo&>(w.bo_wo_a),
            &static_cast<xrt::bo&>(bo_oa_pack_out_),
            &bo_d1_, &bo_d2_ };
        auto run = kernel_cache_->run_blob(
            kernel_cache_->entry_xclbin_path("qck"), oa_packed_blob_, args, 5);
        run.wait();
        npu_sync_from(bo_oa_pack_out_);
        auto* c = bo_oa_pack_out_.map<bf16_t*>();
        for (int g = 0; g < OA_NG; ++g) {
            const bf16_t* tile = (const bf16_t*)((const char*)c + (size_t)g * c_tile_b);
            for (int m = 0; m < M_PAD; ++m)
                memcpy(low_g.data() + (size_t)m * 1024, tile + (size_t)m * 1024,
                       1024 * sizeof(bf16_t));
            for (int m = 0; m < M_PAD; ++m)
                memcpy(low.data() + (size_t)m * 8192 + g * 1024,
                       low_g.data() + (size_t)m * 1024,
                       1024 * sizeof(bf16_t));
        }
        if (dbg) {
            const bf16_t* tile0 = (const bf16_t*)((const char*)c + 0);
            for (int m = 0; m < M_PAD; ++m)
                memcpy(low_g.data() + (size_t)m * 1024, tile0 + (size_t)m * 1024,
                       1024 * sizeof(bf16_t));
            float amx_n=0, rmx=0, md=0;
            for(int n=0;n<1024;n++){
                double acc=0; for(int k=0;k<4096;k++) acc+=(double)bf16f(heads_pad[(size_t)0*heads_cols+0*4096+k])*(double)bf16f(w.wo_a[(size_t)(0*1024+n)*4096+k]);
                float ref=(float)acc, npu=bf16f(low_g[(size_t)0*1024+n]);
                if(std::fabs(ref)>rmx)rmx=std::fabs(ref); if(std::fabs(npu)>amx_n)amx_n=std::fabs(npu); if(std::fabs(npu-ref)>md)md=std::fabs(npu-ref);}
            fprintf(stderr,"[oa-audit] L%d g0 row0: |npu|<=%.4f |ref|<=%.4f maxdiff=%.4f\n",lid,amx_n,rmx,md); fflush(stderr);
        }
    } else {
    for (int g = 0; g < 8; g++) {
        for (int m = 0; m < M_PAD; m++)
            memcpy(heads_g.data() + (size_t)m * 4096,
                   heads_pad.data() + (size_t)m * heads_cols + g * 4096,
                   4096 * sizeof(bf16_t));
        npu_gemm_mla_vec("qck", low_g.data(), heads_g.data(), nullptr,
                         M_PAD, 1024, 4096,
                         &w.bo_wo_a, (size_t)g * 1024 * 4096 * sizeof(bf16_t));
        for (int m = 0; m < M_PAD; m++)
            memcpy(low.data() + (size_t)m * 8192 + g * 1024,
                   low_g.data() + (size_t)m * 1024,
                   1024 * sizeof(bf16_t));
        if (dbg && g == 0) {
            float amx_n=0, rmx=0, md=0;
            for (int n=0;n<1024;n++){
                double acc=0; for(int k=0;k<4096;k++) acc+=(double)bf16f(heads_pad[(size_t)0*heads_cols+0*4096+k])*(double)bf16f(w.wo_a[(size_t)(0*1024+n)*4096+k]);
                float ref=(float)acc, npu=bf16f(low_g[(size_t)0*1024+n]);
                if(std::fabs(ref)>rmx)rmx=std::fabs(ref); if(std::fabs(npu)>amx_n)amx_n=std::fabs(npu); if(std::fabs(npu-ref)>md)md=std::fabs(npu-ref);}
            fprintf(stderr,"[oa-audit] L%d g0 row0: |npu|<=%.4f |ref|<=%.4f maxdiff=%.4f\n",lid,amx_n,rmx,md); fflush(stderr);
        }
    }
    }
    if(audit_on){float mx=-1e30f,mn=1e30f; int nan=0; for(int i=0;i<M*8192;i++){float v=bf16f(low[i]); if(std::isnan(v)||std::isinf(v))nan++; else{if(v>mx)mx=v; if(v<mn)mn=v;}} fprintf(stderr,"[dbg] L%d low(oa): mn=%.3f mx=%.3f nan=%d\n",lid,mn,mx,nan); fflush(stderr);}
    if (lid == 0 && is_prefill_ && M > 1 && std::getenv("FST_MLA_TRACE")) {
        auto dump_bf = [](const char* path, const bf16_t* p, size_t n){
            std::vector<float> tmp(n);
            for (size_t i=0;i<n;i++) tmp[i]=bf16f(p[i]);
            FILE* f=std::fopen(path,"wb");
            if(f){std::fwrite(tmp.data(),sizeof(float),n,f);std::fclose(f);}
            float mx=0;for(float v:tmp)mx=std::max(mx,std::fabs(v));
            fprintf(stderr,"[mla-trace] dump %s n=%zu |mx=%.4f\n",path,n,mx);fflush(stderr);
        };
        dump_bf("/tmp/eng_L00_low.f32", low.data(), (size_t)M*8192);   /* post wo_a, pre wo_b */
    }

    /* OB: low[M_PAD,8192] @ wo_b[N=4096,K=8192] -> attn_out[M_PAD,4096]
     * N-tiled into 2 calls of N=2048 (ob kernel).  wo_b is [N,K] native -> each
     * N-tile is a contiguous [2048,8192] block at &wo_b[n_off*8192]. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> attn_out(M_PAD * hd, 0);
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> ob_tile(M_PAD * 2048, 0);
        for (int n_off = 0; n_off < hd; n_off += 2048) {
            npu_gemm_mla_vec("ob", ob_tile.data(), low.data(),
                             w.wo_b.data() + (size_t)n_off * 8192,
                             M_PAD, 2048, 8192);
            for (int m = 0; m < M_PAD; m++)
                memcpy(attn_out.data() + (size_t)m * hd + n_off,
                       ob_tile.data() + (size_t)m * 2048,
                       2048 * sizeof(bf16_t));
        }
    }

    if (dbg) {
        /* O-proj self-check row0 (correct [N,K] native indexing): ref low[n]=sum_k
         * heads[k]*wo_a[g*1024+n,k]; ref attn[d]=sum_k low[k]*wo_b[d,k]. */
        std::vector<float> low_ref(8192, 0.0f);
        for (int g=0;g<8;g++) for(int n=0;n<1024;n++){
            double acc=0; for(int k=0;k<4096;k++) acc+=(double)bf16f(heads_pad[(size_t)0*heads_cols+g*4096+k])*(double)bf16f(w.wo_a[(size_t)(g*1024+n)*4096+k]);
            low_ref[g*1024+n]=(float)acc;}
        float lmx_npu=0,lmx_ref=0,lmd=0;
        for(int n=0;n<8192;n++){float a=std::fabs(bf16f(low[(size_t)0*8192+n])),b=std::fabs(low_ref[n]); if(a>lmx_npu)lmx_npu=a; if(b>lmx_ref)lmx_ref=b; float d=std::fabs(bf16f(low[(size_t)0*8192+n])-low_ref[n]); if(d>lmd)lmd=d;}
        std::vector<float> ao_ref(hd,0.0f);
        for(int d=0;d<hd;d++){double acc=0; for(int k=0;k<8192;k++) acc+=(double)low_ref[k]*(double)bf16f(w.wo_b[(size_t)d*8192+k]); ao_ref[d]=(float)acc;}
        float amx_npu=0,amx_ref=0,amd=0;
        for(int d=0;d<hd;d++){float a=std::fabs(bf16f(attn_out[(size_t)0*hd+d])),b=std::fabs(ao_ref[d]); if(a>amx_npu)amx_npu=a; if(b>amx_ref)amx_ref=b; float df=std::fabs(bf16f(attn_out[(size_t)0*hd+d])-ao_ref[d]); if(df>amd)amd=df;}
        fprintf(stderr,"[ob-audit] L%d O-proj row0: low |npu|<=%.4f |ref|<=%.4f md=%.4f | attn |npu|<=%.4f |ref|<=%.4f md=%.4f\n",
                lid, lmx_npu, lmx_ref, lmd, amx_npu, amx_ref, amd);
        fflush(stderr);
    }
    if(audit_on){float mx=-1e30f,mn=1e30f;int nan=0;for(int i=0;i<M*hd;i++){float v=bf16f(attn_out[i]);if(std::isnan(v)||std::isinf(v))nan++;else{if(v>mx)mx=v;if(v<mn)mn=v;}} fprintf(stderr,"[dbg] L%d attn_out: mn=%.3f mx=%.3f nan=%d\n",lid,mn,mx,nan); fflush(stderr);}

    memcpy(h, attn_out.data(), (size_t)M * hd * sizeof(bf16_t));
}

xrt::bo& FSTEngine::get_expert_bo(int lid, int e) {
    CacheKey k{lid, e};
    auto it = expert_bo_cache_.find(k);
    if (it != expert_bo_cache_.end()) {
        /* HIT: the BO is the authoritative copy — return it directly, WITHOUT
         * touching the pager (no SSD read, no vector alloc, no sync_to). */
        expert_bo_lru_.remove(k);
        expert_bo_lru_.push_front(k);
        return it->second;
    }
    /* MISS: load the expert from SSD via the pager (this is the ONLY path that
     * touches the pager), then copy its packed weights once into a new BO. */
    const Expert& exp = pager_->get(lid, e);
    const size_t bytes = exp.packed_weights.size();
    /* Evict LRU until there's room for `bytes` under expert_bo_cap_. */
    while (expert_bo_bytes_ + bytes > expert_bo_cap_ && !expert_bo_lru_.empty()) {
        CacheKey old = expert_bo_lru_.back();
        expert_bo_lru_.pop_back();
        auto oit = expert_bo_cache_.find(old);
        if (oit != expert_bo_cache_.end()) {
            expert_bo_bytes_ -= oit->second.size();
            expert_bo_cache_.erase(oit);
        }
    }
    /* Create a persistent host_only BO holding the packed weights; sync once.
     * The dequant kernel reads this BO directly every subsequent dispatch with
     * NO host memcpy / NO per-dispatch sync_to (host_only BO is device-readable). */
    xrt::bo bo(npu_device_, bytes, xrt::bo::flags::host_only, grp_);
    std::memcpy(bo.map<char*>(), exp.packed_weights.data(), bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    expert_bo_lru_.push_front(k);
    expert_bo_bytes_ += bytes;
    auto ins = expert_bo_cache_.emplace(k, std::move(bo));
    return ins.first->second;
}

xrt::bo& FSTEngine::get_draft_expert_bo(int lid, int e) {
    CacheKey k{lid, e};
    auto it = draft_expert_bo_cache_.find(k);
    if (it != draft_expert_bo_cache_.end()) {
        /* HIT: the BO is the authoritative copy — return it directly, WITHOUT
         * touching draft_pager_ (no SSD read, no vector alloc, no sync_to). */
        draft_expert_bo_lru_.remove(k);
        draft_expert_bo_lru_.push_front(k);
        return it->second;
    }
    /* MISS: load the draft expert from SSD via draft_pager_, then copy its
     * packed weights once into a new host_only BO.  This is the ONLY path that
     * touches draft_pager_ for this expert. */
    const Expert& exp = draft_pager_->get(lid, e);
    const size_t bytes = exp.packed_weights.size();
    while (draft_expert_bo_bytes_ + bytes > DRAFT_EXPERT_BO_CAP && !draft_expert_bo_lru_.empty()) {
        CacheKey old = draft_expert_bo_lru_.back();
        draft_expert_bo_lru_.pop_back();
        auto oit = draft_expert_bo_cache_.find(old);
        if (oit != draft_expert_bo_cache_.end()) {
            draft_expert_bo_bytes_ -= oit->second.size();
            draft_expert_bo_cache_.erase(oit);
        }
    }
    /* Persistent host_only BO: dequant reads it every subsequent dispatch with
     * NO host memcpy / NO per-dispatch sync_to (host_only BO is device-readable). */
    xrt::bo bo(npu_device_, bytes, xrt::bo::flags::host_only, grp_);
    std::memcpy(bo.map<char*>(), exp.packed_weights.data(), bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    draft_expert_bo_lru_.push_front(k);
    draft_expert_bo_bytes_ += bytes;
    auto ins = draft_expert_bo_cache_.emplace(k, std::move(bo));
    return ins.first->second;
}

// ── Batched multi-core FFN core (FST_MC_FFN) ────────────────────────────────
// Collapses the per-expert FFN dispatch loop (dequant+gate+up+silu+mul+down =
// 6 dispatches/expert) into ONE multi-core dispatch per op across E_MC=6 experts
// in parallel + ONE batched silu/mul.  HW-verified bit-equivalent to the shipped
// single-core GEMM (tools/mc_gemm_probe: vec_mc/down_mc cos 0.9988 == single-core
// 0.9988 — same aie::mmul, 6 workers in parallel, NOT a new kernel).
//
// bo_batch_w_ holds E_MC=6 slots x 3 (gate|up|down) x proj_elems in 3x-stride:
// expert i gate @ i*3*proj_elems, up @ +proj_elems, down @ +2*proj_elems.  The
// multicore B TAP views [E*3N, K] and walks exactly N rows per worker (the 3x
// stride picks gate/up/down slices); the engine shifts the B sub-buffer base for
// up (+N*K) and down (+2*N*K).  Unused slots (batch < E_MC) are zeroed so their
// GEMM output is exactly 0 — guarding the float accumulation against 0*Inf=NaN
// from garbage weights.  Caller uploads h (replicated to Mx rows) to
// bo_scratch_in_ before calling.
void FSTEngine::run_batched_ffn(int M, int Mx, int hd,
        const std::vector<int>& slot_eids,
        const std::function<void(int slot, xrt::bo& out_sub)>& prepare_slot,
        const std::function<float(int slot, int m)>& weight_of,
        float* acc, int dbg_lid, bool dbg_shared) {
    constexpr int E_MC = 6;
    const size_t proj_elems = (size_t)hd * INTER_DIM;
    const size_t proj_sz    = proj_elems * sizeof(bf16_t);
    const size_t slot_sz    = 3 * proj_sz;                         // gate|up|down
    const size_t full_b     = (size_t)E_MC * 3 * proj_elems * sizeof(bf16_t);
    const int n_unique = (int)slot_eids.size();
    if (n_unique <= 0) return;

    auto& gemm_mc    = kernel_cache_->get("gemm_mc");       // gate/up vec_mc
    auto& gemm_dn_mc = kernel_cache_->get("gemm_down_mc"); // down (own xclbin)
    ScratchSet& S = scratch_pool_[0];                        // single batch -> no ping-pong

    for (int batch_start = 0; batch_start < n_unique; batch_start += E_MC) {
        const int nbatch = std::min(E_MC, n_unique - batch_start);

        // Zero unused local slots -> their GEMM output is exactly 0 (no NaN/Inf
        // leaking into the float acc via 0*Inf).  Used slots are overwritten on-
        // device by prepare_slot (dequant) or host memcpy+sync (shared).
        for (int ls = nbatch; ls < E_MC; ls++) {
            xrt::bo sub(bo_batch_w_, slot_sz, (size_t)ls * slot_sz);
            std::memset(sub.map<char*>(), 0, slot_sz);
            npu_sync_to(sub);
        }
        // Fill local slots 0..nbatch-1 with experts batch_start+ls (dequant/memcpy).
        for (int ls = 0; ls < nbatch; ls++) {
            xrt::bo sub(bo_batch_w_, slot_sz, (size_t)ls * slot_sz);
            prepare_slot(batch_start + ls, sub);
        }
        flush_pending_runs();

        // gate (B = full bo_batch_w_ @0) + up (B sub @+proj_sz).  A = bo_scratch_in_
        // (h replicated to Mx rows), replicated to all 6 workers.
        npu_sync_to(S.bo_ha); npu_sync_to(S.bo_hb);
        {
            auto run = gemm_mc(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_batch_w_),
                static_cast<xrt::bo&>(S.bo_ha),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        {
            xrt::bo boBup(bo_batch_w_, full_b - proj_sz, proj_sz);
            auto run = gemm_mc(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(boBup),
                static_cast<xrt::bo&>(S.bo_hb),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        flush_pending_runs();

        // silu(gate) then silu*up — batched over all E_MC*Mx rows (196608 elems).
        npu_sync_to(S.bo_silu); npu_sync_to(S.bo_mul);
        npu_ew_async("silu_b", S.bo_ha, S.bo_silu);
        npu_ew_bin_async("mul_b", S.bo_silu, S.bo_hb, S.bo_mul);
        flush_pending_runs();

        // down (down_mc): A = S.bo_mul per-worker [E_MC*Mx, K], B = down slice
        // @2*proj_sz.  C = S.bo_down [E_MC*Mx, hd].
        npu_sync_to(S.bo_down);
        {
            xrt::bo boBdn(bo_batch_w_, full_b - 2 * proj_sz, 2 * proj_sz);
            auto run = gemm_dn_mc(3, 0, 0,
                static_cast<xrt::bo&>(S.bo_mul), static_cast<xrt::bo&>(boBdn),
                static_cast<xrt::bo&>(S.bo_down),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        flush_pending_runs();

        // Read back C_down [E_MC*Mx, hd] and accumulate router-weighted contribs.
        npu_sync_from(S.bo_down);
        const bf16_t* down = S.bo_down.map<bf16_t*>();
        for (int ls = 0; ls < nbatch; ls++) {
            int gslot = batch_start + ls;
            for (int m = 0; m < M; m++) {
                float w = weight_of(gslot, m);
                if (w == 0.0f) continue;
                const bf16_t* row = down + (size_t)(ls * Mx + m) * hd;
                for (int d = 0; d < hd; d++)
                    acc[(size_t)m * hd + d] += bf16f(row[d]) * w;
            }
        }
    }
}

void FSTEngine::process_expert_ffn(int lid, const bf16_t* h, bf16_t* out, int M, const int* eids, int tk, const float* r_weights) {
    const int Mx = 16, total = M * tk, hd = config_.hidden_dim;   // M=16 std
    const int rep = Mx / M;
    const size_t proj_elems = (size_t)hd * INTER_DIM;
    const size_t proj_sz = proj_elems * sizeof(bf16_t);  // 4096*2048*2 = 16MB

    std::vector<float> acc(M * hd, 0.0f);
    /* Only host buffer left in the FFN: the final down projection output,
     * read back once per expert (DEFERRED to the next expert's first cross-
     * context barrier) to accumulate the router-weighted sum.  gate/up/silu/
     * mul NEVER touch the CPU (pure NPU BO-to-BO chain). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> down_out(Mx * hd);

    const size_t in_sz = (size_t)Mx * hd * sizeof(bf16_t);
    if (bo_scratch_in_.size() < in_sz) bo_scratch_in_ = xrt::ext::bo(npu_device_, in_sz);

    /* Upload activation h replicated to Mx rows ONCE for the whole layer
     * (rows M..Mx-1 zero-padded).  bo_scratch_in_ is read-only across all
     * experts (the down GEMM reads sc.bo_mul, not bo_scratch_in_), so it is
     * safe to share across in-flight experts. */
    {
        char* hp = bo_scratch_in_.map<char*>();
        for (int r = 0; r < rep; r++)
            std::memcpy(hp + (size_t)r * M * hd * sizeof(bf16_t), h,
                        (size_t)M * hd * sizeof(bf16_t));
        int filled = rep * M;
        if (filled < Mx)
            std::memset(hp + (size_t)filled * hd * sizeof(bf16_t), 0,
                        (size_t)(Mx - filled) * hd * sizeof(bf16_t));
        npu_sync_to(bo_scratch_in_);
    }

    /* ── FST_MC_FFN: batched multi-core path (6 experts in parallel per op). ──
     * Same GEMM math as the default async path (HW-verified cos 0.9988).  Trades
     * the down‖dequant overlap for ~3× fewer dispatches; net win at decode (M=1,
     * n_unique<=6 -> 1 batch) where there is no cross-expert overlap to lose. */
    if (std::getenv("FST_MC_FFN")) {
        std::vector<int> slot_eids;
        std::vector<bool> seen_mc(config_.n_experts, false);
        for (int i = 0; i < total; i++) {
            int e = eids[i];
            if (!seen_mc[e]) { seen_mc[e] = true; slot_eids.push_back(e); }
        }
        auto& deq_krnl_mc = kernel_cache_->get("dequant");
        auto prepare_slot = [&](int slot, xrt::bo& out_sub) {
            xrt::bo& wbo = get_expert_bo(lid, slot_eids[slot]);  // host_only BO, no memcpy
            auto run = deq_krnl_mc(3, 0, 0,
                static_cast<xrt::bo&>(wbo), static_cast<xrt::bo&>(out_sub),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_),
                static_cast<xrt::bo&>(bo_d3_));
            pending_runs_.push_back(std::move(run));
        };
        auto weight_of = [&](int slot, int m) -> float {
            int e = slot_eids[slot];
            for (int j = 0; j < tk; j++)
                if (eids[m * tk + j] == e) return r_weights[m * tk + j];
            return 0.0f;
        };
        run_batched_ffn(M, Mx, hd, slot_eids, prepare_slot, weight_of, acc.data(), lid, false);
        for (int i = 0; i < M * hd; i++) out[i] = f2bf(acc[i]);
        return;
    }

    auto& deq_krnl      = kernel_cache_->get("dequant");
    auto& gemm_krnl     = kernel_cache_->get("gemm");       // gate (M=16,K=4096,N=2048)
    auto& gemm2_krnl    = kernel_cache_->get("gemm2");      // up   (M=16,K=4096,N=2048)
    auto& gemm_down_krnl= kernel_cache_->get("gemm_down");  // down (M=16,K=2048,N=4096)

    /* drain_set: read back set s's down output (after flush_pending_runs has
     * waited it) and accumulate its owner expert's router-weighted contribution
     * into acc.  The diagnostic audit hooks read down_out / sc.bo_w here. */
    auto drain_set = [&](int s, int expert_count) {
        int e = scratch_pool_[s].owner_e;
        if (e < 0) return;
        const size_t c_sz = (size_t)Mx * hd * sizeof(bf16_t);
        npu_sync_from(scratch_pool_[s].bo_down);
        std::memcpy(down_out.data(), scratch_pool_[s].bo_down.map<char*>(), c_sz);
        scratch_pool_[s].owner_e = -1;

        if (audit_on && lid == 0 && expert_count == 1) {
            float mn = 1e30f, mx = -1e30f; int nan = 0;
            for (size_t i = 0; i < (size_t)Mx * hd; i++) {
                float v = bf16f(down_out[i]);
                if (std::isnan(v) || std::isinf(v)) nan++;
                else { if (v < mn) mn = v; if (v > mx) mx = v; }
            }
            fprintf(stderr, "[dbg] L0 e=%d down_out: mn=%.4f mx=%.4f nan=%d\n",
                    e, mn, mx, nan);
            fflush(stderr);
        }
        /* CPU reference of the full FFN for m=0 (no-clamp, apples-to-apples vs
         * the NPU BO-to-BO chain).  Uses sc.bo_w (NPU-dequanted weights). */
        if ((lid == 0 || lid == 26) && expert_count == 1 && std::getenv("FST_FFN_AUDIT")) {
            npu_sync_from(scratch_pool_[s].bo_w);
            const bf16_t* W = scratch_pool_[s].bo_w.map<bf16_t*>();
            const bf16_t* Wg = W;
            const bf16_t* Wu = W + proj_elems;
            const bf16_t* Wd = W + 2 * proj_elems;
            std::vector<float> act(INTER_DIM);
            for (int n = 0; n < INTER_DIM; n++) {
                double g = 0, u = 0;
                for (int k = 0; k < hd; k++) {
                    g += (double)bf16f(h[k]) * (double)bf16f(Wg[(size_t)n * hd + k]);
                    u += (double)bf16f(h[k]) * (double)bf16f(Wu[(size_t)n * hd + k]);
                }
                float gf = (float)g, uf = (float)u;
                act[n] = (gf / (1.0f + expf(-gf))) * uf;
            }
            float nmx_npu = 0, nmx_ref = 0, md = 0; double dot = 0, nn = 0, nr = 0;
            for (int d = 0; d < hd; d++) {
                double a = 0;
                for (int k = 0; k < INTER_DIM; k++)
                    a += (double)act[k] * (double)bf16f(Wd[(size_t)d * INTER_DIM + k]);
                float ref = (float)a;
                float npu = bf16f(down_out[(size_t)0 * hd + d]);
                if (std::fabs(npu) > nmx_npu) nmx_npu = std::fabs(npu);
                if (std::fabs(ref) > nmx_ref) nmx_ref = std::fabs(ref);
                float diff = std::fabs(npu - ref); if (diff > md) md = diff;
                dot += (double)npu * (double)ref; nn += (double)npu*npu; nr += (double)ref*ref;
            }
            double cos_f = (nn > 0 && nr > 0) ? dot / (std::sqrt(nn) * std::sqrt(nr)) : 0;
            double ratio = (nr > 0) ? std::sqrt(nn / nr) : 0;
            fprintf(stderr, "[ffn-cpu] L%d e=%d m=0: |npu|mx=%.4f |ref|mx=%.4f maxdiff=%.4f"
                    " cos=%.5f |npu|/|ref|=%.5f\n",
                    lid, e, nmx_npu, nmx_ref, md, cos_f, ratio);
            fflush(stderr);
        }

        for (int j = 0; j < total; j++) {
            if (eids[j] != e) continue;
            int m = j / tk;
            float w = r_weights[j];
            for (int d = 0; d < hd; d++)
                acc[m * hd + d] += bf16f(down_out[m * hd + d]) * w;
            if (lid == 0 && m == 13 && std::getenv("FST_ROUTED_DUMP")) {
                char path[128]; std::snprintf(path, sizeof(path), "/tmp/eng_L0_e%d_m13.f32", e);
                std::vector<float> tmp(hd);
                for (int d = 0; d < hd; d++) tmp[d] = bf16f(down_out[(size_t)m * hd + d]);
                FILE* f = std::fopen(path, "wb");
                if (f) { std::fwrite(tmp.data(), sizeof(float), hd, f); std::fclose(f); }
                float dm = 0; for (int d = 0; d < hd; d++) { float v=std::fabs(tmp[d]); if(v>dm)dm=v; }
                fprintf(stderr, "[routed-dump] L0 e=%d m=13 w=%.4f |down|mx=%.4f -> %s\n",
                        e, w, dm, path); fflush(stderr);
            }
            if ((lid == 0 || lid == 26) && m == 0 && std::getenv("FST_FFN_AUDIT")) {
                float dm = 0;
                for (int d = 0; d < hd; d++) {
                    float v = std::fabs(bf16f(down_out[(size_t)m * hd + d]));
                    if (v > dm) dm = v;
                }
                fprintf(stderr, "[m0-exp] L%d e=%d w=%.4f |down[0]|mx=%.4f\n",
                        lid, e, w, dm);
                fflush(stderr);
            }
        }
    };

    std::vector<bool> seen(config_.n_experts, false);
    pool_idx_ = 0;
    int prev_set = -1;
    int expert_count = 0;

    for (int i = 0; i < total; i++) {
        int e = eids[i];
        if (seen[e]) continue;
        seen[e] = true;
        ++expert_count;

        const int s = pool_idx_ & 3;
        ++pool_idx_;
        ScratchSet& sc = scratch_pool_[s];

        /* ── NPU MXFP4 DEQUANT → NPU GEMM CHAIN (BO-to-BO, no host copy) ──
         * 17 bytes/block → 32 BF16 (1 E8M0 scale + 16 FP4 nibbles).  Weights
         * packed B[N,K] row-major; GEMM kernels are transposed-B so no device
         * transpose.  Dequant writes directly to sc.bo_w. */
        /* Upload packed weights + prime sc.bo_w.  Done while the PREVIOUS
         * expert's down GEMM may still be in-flight on the gemm_down context:
         * safe because bo_scratch_packed_ and sc.bo_w (set s) are distinct BOs
         * from the previous set's bo_down/bo_mul — no BO is reused mid-flight. */
        bool ffn_t = std::getenv("FST_FFN_TIME");
        double t_up=0, t_dq=0, t_gu=0, t_sm=0;
        double u0 = ffn_t ? now() : 0;
        /* Persistent host_only BO for the packed weights: dequant reads it
         * directly with NO per-dispatch host memcpy / NO per-dispatch sync_to
         * (the ~2 ms upload that was t_up).  Created once per (lid,e) on first
         * use, LRU-bounded to EXPERT_BO_CAP.  BO-cache HIT returns without any
         * pager touch (no SSD read, no vector alloc) — the BO is authoritative. */
        xrt::bo& wbo = get_expert_bo(lid, e);
        npu_sync_to(sc.bo_w);
        if (ffn_t) t_up = now() - u0;

        /* FST_SEQ_BUILDER self-test: run dequant via the in-process blob path
         * (run_blob) instead of the static file-based module, to prove the
         * dynamic NpuSequenceBuilder -> aiebu -> xrt::module -> kernel -> run
         * path produces identical results.  Default (flag unset) = unchanged. */
        if (std::getenv("FST_SEQ_BUILDER")) {
            xrt::bo* dargs[5] = { &static_cast<xrt::bo&>(wbo),
                                  &static_cast<xrt::bo&>(sc.bo_w),
                                  &static_cast<xrt::bo&>(bo_d1_),
                                  &static_cast<xrt::bo&>(bo_d2_),
                                  &static_cast<xrt::bo&>(bo_d3_) };
            auto drun = kernel_cache_->run_registered_blob("dequant", dargs, 5);
            pending_runs_.push_back(std::move(drun));
        } else {
        auto drun = deq_krnl(3, 0, 0,
            static_cast<xrt::bo&>(wbo),
            static_cast<xrt::bo&>(sc.bo_w),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_),
            static_cast<xrt::bo&>(bo_d3_));
        pending_runs_.push_back(std::move(drun));
        }

        /* Cross-context barrier: dequant(ctx8) -> gate(ctx1).  This flush also
         * waits the PREVIOUS expert's down (still in pending_runs_), which is
         * the async overlap point: prev_down ‖ this_dequant ran concurrently on
         * different hw_contexts.  Drain the previous expert now (readback +
         * accumulate) before reusing its set's down_out BO. */
        double dq0 = ffn_t ? now() : 0;
        flush_pending_runs();
        if (ffn_t) t_dq = now() - dq0;
        if (prev_set >= 0) { drain_set(prev_set, expert_count - 1); prev_set = -1; }

        if (audit_on && lid == 0 && expert_count == 1) {
            npu_sync_from(sc.bo_w);
            auto* p = sc.bo_w.map<bf16_t*>();
            float mx=-1e30f,mn=1e30f; int nan=0;
            for (size_t i=0;i<3*proj_elems;i++){float v=bf16f(p[i]); if(std::isnan(v)||std::isinf(v))nan++; else{if(v>mx)mx=v; if(v<mn)mn=v;}}
            fprintf(stderr,"[dbg] L0 dequant out: mn=%.4f mx=%.4f nan=%d (3*proj=%zu)\n",mn,mx,nan,3*proj_elems); fflush(stderr);
            if (std::getenv("FST_DEQUANT_DUMP")) {
                std::vector<float> tmp(proj_elems);
                for (size_t i = 0; i < proj_elems; i++) tmp[i] = bf16f(p[i]);
                char path[128]; std::snprintf(path, sizeof(path), "/tmp/eng_L0_e%d_gate.f32", e);
                FILE* f = std::fopen(path, "wb");
                if (f) { std::fwrite(tmp.data(), sizeof(float), proj_elems, f); std::fclose(f); }
                fprintf(stderr, "[dequant-dump] L0 e=%d gate -> %s (mx=%.4f)\n", e, path, mx); fflush(stderr);
            }
        }

        /* ── gate + up GEMMs: B read from sc.bo_w sub-buffers (b_col_maj). ── */
        npu_sync_to(sc.bo_ha);
        npu_sync_to(sc.bo_hb);
        {
            xrt::bo bo_dq_gate(sc.bo_w, proj_sz, 0);
            auto run = gemm_krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_dq_gate),
                static_cast<xrt::bo&>(sc.bo_ha),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        {
            xrt::bo bo_dq_up(sc.bo_w, proj_sz, proj_sz);
            auto run = gemm2_krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_dq_up),
                static_cast<xrt::bo&>(sc.bo_hb),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        double gu0 = ffn_t ? now() : 0;
        flush_pending_runs();   // cross-ctx: gate/up(ctx1) -> silu/mul(ctx9)
        if (ffn_t) t_gu = now() - gu0;

        /* ── silu(gate) then silu*up (ew_unified, no swiglu clamp). ── */
        npu_sync_to(sc.bo_silu);
        npu_sync_to(sc.bo_mul);
        npu_ew_async("silu", sc.bo_ha, sc.bo_silu);
        npu_ew_bin_async("mul", sc.bo_silu, sc.bo_hb, sc.bo_mul);
        double sm0 = ffn_t ? now() : 0;
        flush_pending_runs();   // cross-ctx: mul(ctx9) -> down(ctx2)
        if (ffn_t) t_sm = now() - sm0;

        /* ── down GEMM → sc.bo_down.  NO flush, NO readback yet — let this
         * down overlap the NEXT expert's dequant (different hw_context).  The
         * down run stays in pending_runs_ until the next expert's dequant->gate
         * barrier (or layer end).  owner_e marks this set as having a pending
         * down to drain. */
        {
            const size_t full_b = (size_t)hd * INTER_DIM * sizeof(bf16_t);
            const size_t full_c = (size_t)Mx * hd * sizeof(bf16_t);
            xrt::bo bo_dq_dn(sc.bo_w, full_b, 2 * proj_sz);
            xrt::bo bo_c(sc.bo_down, full_c, 0);
            auto run = gemm_down_krnl(3, 0, 0,
                static_cast<xrt::bo&>(sc.bo_mul), static_cast<xrt::bo&>(bo_dq_dn),
                static_cast<xrt::bo&>(bo_c),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        sc.owner_e = e;
        prev_set = s;
        if (ffn_t)
            fprintf(stderr, "[ffn-t] L%d e=%d up=%.2fms deq_flush=%.2fms gateup_flush=%.2fms silumul_flush=%.2fms\n",
                    lid, e, t_up*1000, t_dq*1000, t_gu*1000, t_sm*1000);
    }

    /* Drain the final expert's down. */
    if (prev_set >= 0) {
        flush_pending_runs();
        drain_set(prev_set, expert_count);
    }

    for (int i = 0; i < M * hd; i++)
        out[i] = f2bf(acc[i]);

}

void FSTEngine::process_shared_expert(int lid, const bf16_t* h, bf16_t* out, int M) {
    auto& w = shared_[lid];
    if (w.shared_gate.empty()) return;

    const int hd = config_.hidden_dim;
    const int Mx = 16, rep = Mx / M;   // M=16 std

    /* Pure NPU BO-to-BO chain — NO dequant (shared weights are preloaded BF16
     * in w.bo_shared_{gate,up,down}, stored [N,K] b_col_maj = exactly the
     * kernel B shapes), NO host GEMM, NO host interleave.  down is the N=4096
     * single-call kernel -> C=[Mx,4096] row-major into bo_scratch_out_.
     * Caller sums routed + shared (ds4.c 7873: ffn_out = moe + shared), so
     * `out` is overwritten with the shared down output (rows 0..M-1). */
    const size_t in_sz  = (size_t)Mx * hd * sizeof(bf16_t);
    const size_t out_sz = (size_t)Mx * hd * sizeof(bf16_t);
    if (bo_scratch_in_.size()  < in_sz)  bo_scratch_in_  = xrt::ext::bo(npu_device_, in_sz);
    if (bo_scratch_out_.size() < out_sz) bo_scratch_out_ = xrt::ext::bo(npu_device_, out_sz);

    {
        char* hp = bo_scratch_in_.map<char*>();
        for (int r = 0; r < rep; r++)
            std::memcpy(hp + (size_t)r * M * hd * sizeof(bf16_t), h,
                        (size_t)M * hd * sizeof(bf16_t));
        int filled = rep * M;
        if (filled < Mx)
            std::memset(hp + (size_t)filled * hd * sizeof(bf16_t), 0,
                        (size_t)(Mx - filled) * hd * sizeof(bf16_t));
        npu_sync_to(bo_scratch_in_);
    }

    /* ── FST_MC_FFN: shared expert via the batched path (1 real slot, 5 zeroed).
     * The 3 preloaded BF16 [N,K] BOs are staged into bo_batch_w_ slot 0 in 3x-
     * stride (gate|up|down); weight_of=1 for slot 0 (shared contributes to every
     * token).  acc starts at 0 so acc == shared down output (caller adds moe). */
    if (std::getenv("FST_MC_FFN")) {
        const size_t proj_elems = (size_t)hd * INTER_DIM;
        const size_t proj_sz = proj_elems * sizeof(bf16_t);
        std::vector<int> slot_eids = {0};
        auto prepare_slot = [&](int /*slot*/, xrt::bo& out_sub) {
            char* dst = out_sub.map<char*>();
            std::memcpy(dst,                       w.bo_shared_gate.map<char*>(), proj_sz);
            std::memcpy(dst + proj_sz,             w.bo_shared_up.map<char*>(),   proj_sz);
            std::memcpy(dst + 2 * proj_sz,         w.bo_shared_down.map<char*>(), proj_sz);
            npu_sync_to(out_sub);
        };
        auto weight_of = [&](int slot, int /*m*/) -> float {
            return (slot == 0) ? 1.0f : 0.0f;
        };
        std::vector<float> acc(M * hd, 0.0f);
        run_batched_ffn(M, Mx, hd, slot_eids, prepare_slot, weight_of, acc.data(), lid, true);
        for (int i = 0; i < M * hd; i++) out[i] = f2bf(acc[i]);
        return;
    }

    auto& gemm_krnl       = kernel_cache_->get("gemm");
    auto& gemm2_krnl      = kernel_cache_->get("gemm2");
    auto& gemm_down_krnl  = kernel_cache_->get("gemm_down");

    ScratchSet& sc = scratch_pool_[0];   // single expert — no cross-expert overlap
    npu_sync_to(sc.bo_ha);
    npu_sync_to(sc.bo_hb);

    {
        auto run = gemm_krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(w.bo_shared_gate),
            static_cast<xrt::bo&>(sc.bo_ha),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
    }
    {
        auto run = gemm2_krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(w.bo_shared_up),
            static_cast<xrt::bo&>(sc.bo_hb),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
    }
    flush_pending_runs();

    /* silu(gate), silu*up on device (no swiglu_limit clamp — mirrors draft). */
    npu_sync_to(sc.bo_silu);
    npu_sync_to(sc.bo_mul);
    npu_ew_async("silu", sc.bo_ha, sc.bo_silu);
    npu_ew_bin_async("mul", sc.bo_silu, sc.bo_hb, sc.bo_mul);
    flush_pending_runs();

    /* down (N=4096 single call): B = w.bo_shared_down [4096,2048] -> C [Mx,4096]. */
    npu_sync_to(bo_scratch_out_);
    {
        auto run = gemm_down_krnl(3, 0, 0,
            static_cast<xrt::bo&>(sc.bo_mul), static_cast<xrt::bo&>(w.bo_shared_down),
            static_cast<xrt::bo&>(bo_scratch_out_),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
    }
    flush_pending_runs();

    /* Single readback: shared down output (rows 0..M-1) into out. */
    {
        npu_sync_from(bo_scratch_out_);
        const bf16_t* sx = bo_scratch_out_.map<bf16_t*>();
        for (int i = 0; i < M * hd; i++)
            out[i] = sx[i];
    }

    if (audit_on && lid == 0) {
        float mn = 1e30f, mx = -1e30f; int nan = 0;
        for (int m = 0; m < M; m++)
            for (int d = 0; d < hd; d++) {
                float v = bf16f(out[(size_t)m * hd + d]);
                if (std::isnan(v) || std::isinf(v)) nan++;
                else { if (v < mn) mn = v; if (v > mx) mx = v; }
            }
        fprintf(stderr, "[dbg] L0 shared down_out: mn=%.4f mx=%.4f nan=%d\n",
                mn, mx, nan);
        fflush(stderr);
    }
}

int FSTEngine::sample_token(float* logits, int vocab_size, float temperature, float top_p) {
    /* Greedy / deterministic decoding.  temperature<=0 would divide by zero
     * below; return the plain argmax instead.  No effect on temperature>0
     * callers (the existing sampled path is unchanged). */
    if (temperature <= 0.f) {
        int best = 0;
        for (int v = 1; v < vocab_size; v++)
            if (logits[v] > logits[best]) best = v;
        return best;
    }
    for (int v = 0; v < vocab_size; v++) logits[v] /= temperature;

    float mx = logits[0];
    for (int v = 1; v < vocab_size; v++) if (logits[v] > mx) mx = logits[v];
    float sum = 0.f;
    for (int v = 0; v < vocab_size; v++) { logits[v] = expf(logits[v] - mx); sum += logits[v]; }
    for (int v = 0; v < vocab_size; v++) logits[v] /= sum;

    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + std::min(vocab_size, 10000), indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    std::vector<int> kept; kept.reserve(vocab_size);
    float cum = 0.f;
    for (int v = 0; v < vocab_size; v++) { kept.push_back(indices[v]); cum += logits[indices[v]]; if (cum >= top_p) break; }

    float rsum = 0.f;
    for (int id : kept) rsum += logits[id];
    for (int id : kept) logits[id] /= rsum;

    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(rng);
    float acc = 0.f;
    for (int id : kept) { acc += logits[id]; if (r <= acc) return id; }
    return kept.back();
}

// ==================================================================
//  DSpark Speculative Decoding
// ==================================================================

void FSTEngine::load_draft_model(const std::string& draft_fst_path) {
    int fd = ::open(draft_fst_path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("Cannot open draft model: " + draft_fst_path);

    auto hb = pread(fd, 128, 0);
    auto* h = (const FSTH*)hb.data();

    draft_cfg_.hidden_dim   = (int)h->hd;
    draft_cfg_.n_layers     = (int)h->nl;
    draft_cfg_.n_experts    = (int)h->ne;
    draft_cfg_.top_k        = (int)h->tk;
    draft_cfg_.vocab_size   = (int)h->vs;
    draft_cfg_.block_size   = (int)(h->r1 & 0xFFFF);
    draft_cfg_.markov_rank  = (int)((h->r1 >> 16) & 0xFFFF);
    draft_cfg_.noise_token_id = (int)h->r2;
    draft_cfg_.max_seq      = config_.max_seq;

    fprintf(stderr, "[draft] %d stages, %d experts, block_size=%d, markov_rank=%d, noise_tid=%d\n",
            draft_cfg_.n_layers, draft_cfg_.n_experts,
            draft_cfg_.block_size, draft_cfg_.markov_rank,
            draft_cfg_.noise_token_id);

    ::close(fd);

    load_draft_shared_weights(draft_fst_path);

    /* Draft pager: the draft model is ~2 GB, cache all experts in RAM */
    draft_pager_ = std::make_unique<ExpertPager>(draft_fst_path, 4000ULL * 1024 * 1024);

    /* Draft KV cache */
    draft_kv_cache_.resize(draft_cfg_.n_layers);
    for (auto& kv : draft_kv_cache_) {
        kv.kv_latent.resize(draft_cfg_.max_seq * MLA_KV_LORA, 0);
        kv.k_pe.resize(draft_cfg_.max_seq * MLA_N_HEADS * MLA_ROPE_DIM, 0);
        const size_t kv_bytes = kv.kv_latent.size() * sizeof(bf16_t);
        const size_t kpe_bytes = kv.k_pe.size() * sizeof(bf16_t);
        kv.kv_latent_bo = xrt::bo(npu_device_, kv_bytes, xrt::bo::flags::host_only,
                                  kernel_cache_->data_group_id());
        kv.k_pe_bo = xrt::bo(npu_device_, kpe_bytes, xrt::bo::flags::host_only,
                             kernel_cache_->data_group_id());
        memset(kv.kv_latent_bo.map<char*>(), 0, kv_bytes);
        memset(kv.k_pe_bo.map<char*>(), 0, kpe_bytes);
        kv.kv_latent_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_bytes, 0);
        kv.k_pe_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, kpe_bytes, 0);
    }

    /* Set the main pager's router weights for draft-driven prefetch.
     * We use layer 0's router as a representative router for all layers. */
    if (!shared_.empty() && !shared_[0].router.empty()) {
        pager_->set_router_weights(shared_[0].router.data(),
                                    shared_[0].router_bias.empty() ? nullptr : shared_[0].router_bias.data(),
                                    config_.n_experts, config_.hidden_dim);
    }

    draft_loaded_ = true;
    fprintf(stderr, "[draft] model loaded successfully\n");
}

void FSTEngine::load_draft_shared_weights(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    auto hb = pread(fd, 128, 0);
    auto* h = (const FSTH*)hb.data();

    draft_shared_.resize(draft_cfg_.n_layers);
    size_t ne = (size_t)h->sdc;
    auto db = pread(fd, ne * 64, (off_t)h->sdo);

    /* Build shape-keyed lookup (same as main model loader) */
    struct T { int l; uint64_t o, sz; int q; };
    std::unordered_map<std::string, std::vector<T>> sm;
    for (size_t i = 0; i < ne; i++) {
        auto* e = (const FSTE*)(db.data() + i * 64);
        char k[64];
        snprintf(k, sizeof(k), "%llux%llux%llu",
                 (unsigned long long)e->s0, (unsigned long long)e->s1, (unsigned long long)e->s2);
        sm[k].push_back({(int)e->lid, e->off, e->sz, (int)e->qt});
    }

    auto pop = [&](const char* k, int lid, bool f32,
                   std::vector<float>& fv,
                   std::vector<bf16_t, AlignedAllocator<bf16_t>>& bv) -> bool {
        auto it = sm.find(k);
        if (it == sm.end() || it->second.empty()) return false;
        size_t bi = 0;
        for (size_t j = 0; j < it->second.size(); j++)
            if (it->second[j].l == lid || it->second[j].l == DRAFT_GLOBAL_LAYER) { bi = j; break; }
        auto& ti = it->second[bi];
        auto raw = pread(fd, (size_t)ti.sz, (off_t)ti.o);
        size_t nel = (size_t)ti.sz / 2;
        if (ti.q == 1) { /* Q8_0 */
            size_t s0 = 1, s1 = 1;
            sscanf(k, "%zux%zu", &s0, &s1);
            nel = s0 * (s1 ? s1 : 1);
            size_t nb = (nel + 31) / 32;
            auto* sc = (const uint16_t*)raw.data();
            auto* v8 = (const int8_t*)(raw.data() + nb * 2);
            std::vector<float> fq(nel);
            for (size_t b = 0; b < nb; b++) {
                float sf = fp16_to_f32(sc[b]);   /* fp16 scale (was BF16<<16 bug) */
                if (fabsf(sf) < 1e-12f) sf = 1.f;
                for (int j = 0; j < 32; j++) {
                    size_t ix = b * 32 + j;
                    if (ix < nel) fq[ix] = (float)v8[ix] * sf;
                }
            }
            if (f32) { fv = std::move(fq); }
            else { bv.resize(nel); for (size_t x = 0; x < nel; x++) bv[x] = f2bf(fq[x]); }
        } else {
            bv.resize(nel);
            memcpy(bv.data(), raw.data(), ti.sz);
            if (f32) {
                fv.resize(nel);
                for (size_t x = 0; x < nel; x++) fv[x] = bf16f(bv[x]);
                bv.clear();
            }
        }
        it->second.erase(it->second.begin() + bi);
        return true;
    };
    auto pb = [&](const char* k, int l, std::vector<bf16_t, AlignedAllocator<bf16_t>>& v) {
        std::vector<float> d; pop(k, l, false, d, v);
    };
    auto pf = [&](const char* k, int l, std::vector<float>& v) {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> d; pop(k, l, true, v, d);
    };
    /* TID-based per-layer raw read (mirrors the main loader's pop_tid_layer).
     * The HC fn matrices share the 16384x24 dim-key (collide in the shape-keyed
     * pop), so HC + attn_sinks are read unambiguously by TID.  db/ne are the
     * shared-directory entries (FSTE, 64 bytes each). */
    auto pop_tid_layer = [&](uint32_t target_tid, int lid) -> std::vector<uint8_t> {
        for (size_t i = 0; i < ne; i++) {
            auto* e = (const FSTE*)(db.data() + i * 64);
            if (e->tid != target_tid || (int)e->lid != lid) continue;
            return pread(fd, (size_t)e->sz, (off_t)e->off);
        }
        return {};
    };
    auto pop_tid = [&](uint32_t target_tid) -> std::vector<uint8_t> {
        for (size_t i = 0; i < ne; i++) {
            auto* e = (const FSTE*)(db.data() + i * 64);
            if (e->tid != target_tid) continue;
            std::vector<uint8_t> b((size_t)e->sz);
            ssize_t r = ::pread(fd, b.data(), (size_t)e->sz, (off_t)e->off);
            if (r < 0)
                fprintf(stderr, "[draft] pread FAIL tid=%u sz=%zu off=%llu: %s\n",
                        target_tid, (size_t)e->sz, (unsigned long long)e->off, strerror(errno));
            else if (r < (ssize_t)e->sz)
                fprintf(stderr, "[draft] pread SHORT tid=%u: got %zd of %zu\n",
                        target_tid, r, (size_t)e->sz);
            return b;
        }
        return {};
    };

    for (int l = 0; l < draft_cfg_.n_layers; l++) {
        auto& w = draft_shared_[l];
        pb("1024x4096x1", l, w.wq_a);  /* [N,K] native — b_col_maj */
        pb("32768x1024x1", l, w.wq_b); /* [N,K] native — b_col_maj */
        pb("512x4096x1", l, w.wkv);    /* [N,K] native — b_col_maj */
        pb("8192x4096x1", l, w.wo_a);  /* [N,K] native — b_col_maj */
        pb("4096x8192x1", l, w.wo_b);  /* [N,K] native — b_col_maj */
        pb("4096x1x1", l, w.attn_norm);
        pb("4096x1x1", l, w.ffn_norm);
        pb("1024x1x1", l, w.q_norm);
        pb("512x1x1", l, w.kv_norm);
        pb("2048x4096x1", l, w.shared_gate); transpose_bf16(w.shared_gate,2048,4096);
        pb("2048x4096x1", l, w.shared_up);   transpose_bf16(w.shared_up,2048,4096);
        pb("4096x2048x1", l, w.shared_down); transpose_bf16(w.shared_down,4096,2048);
        {pf("256x4096x1", l, w.router); transpose_f32(w.router,256,4096);}
        /* main_norm (TID20, 4096x1) is the RMSNorm ds4.c applies to the MAIN
         * model's hidden state BEFORE projecting it into the draft:
         *   draft_input = e_proj(embed(token)) + h_proj(hnorm(prev_main_hc))
         * It exists ONLY at lid=0 (HF: mtp.0.main_norm.weight; mtp.1/mtp.2 have
         * none).  Load it ONLY at l==0.  Loading it unconditionally for every l
         * would, at l=2, consume the draft_norm (TID21) entry — both share shape
         * 4096x1x1 and the shape-keyed pop resolves by call order.  At l==0 the
         * remaining 4096x1x1 entries (after attn_norm TID3 / ffn_norm TID4 are
         * consumed above) are just [TID20], so this grabs main_norm and leaves
         * TID21 (lid=2) for the draft_norm load below.  main_proj likewise
         * exists only at lid=0 (mtp.0.main_proj); pb() returns false (empty)
         * for l=1,2, leaving main_proj stage-0-only as the HF checkpoint does. */
        if (l == 0) pb("4096x1x1", l, w.main_norm);
        /* main_proj: HF Linear(dim*3=12288, dim=4096) -> weight [4096, 12288] =
         * [N=4096, K=12288] native b_col_maj.  Project main_hidden[1,12288] ->
         * [1,4096] via N-tiled qc kernel.  NO transpose (the old
         * transpose_bf16(4096,12288) reversed it to [12288,4096] and the engine
         * projected 4096->12288 BACKWARDS — the core draft bug). */
        pb("4096x12288x1", l, w.main_proj);
        /* ── HC (Hybrid Connection) per-stage, read by TID (the fn matrices
         * share the 16384x24 dim-key and collide in the shape-keyed pop).  The
         * checkpoint stores fn as BF16 [HC_MIX_DIM, HC_DIM] = [24, 16384];
         * transpose to [HC_DIM, HC_MIX_DIM] so npu_gemm_hc (A@B, B=[K,N])
         * computes mix = flat @ fn.  scale is F32 [3]; base is F32 [24].
         * attn_sinks (TID 22) is F32 [n_heads].  Mirrors the main loader. */
        {
            const size_t fn_n = (size_t)HC_DIM * HC_MIX_DIM;
            auto a_fn = pop_tid_layer(23, l);
            if (a_fn.size() >= fn_n * sizeof(bf16_t)) {
                w.hc_attn_fn.resize(fn_n);
                memcpy(w.hc_attn_fn.data(), a_fn.data(), fn_n * sizeof(bf16_t));
                transpose_bf16(w.hc_attn_fn, HC_MIX_DIM, HC_DIM);  /* [24,16384]->[16384,24] */
            }
            auto a_sc = pop_tid_layer(24, l);
            if (a_sc.size() >= 3 * sizeof(float)) { w.hc_attn_scale.resize(3); memcpy(w.hc_attn_scale.data(), a_sc.data(), 3 * sizeof(float)); }
            auto a_bs = pop_tid_layer(25, l);
            if (a_bs.size() >= HC_MIX_DIM * sizeof(float)) { w.hc_attn_base.resize(HC_MIX_DIM); memcpy(w.hc_attn_base.data(), a_bs.data(), HC_MIX_DIM * sizeof(float)); }
            auto f_fn = pop_tid_layer(26, l);
            if (f_fn.size() >= fn_n * sizeof(bf16_t)) {
                w.hc_ffn_fn.resize(fn_n);
                memcpy(w.hc_ffn_fn.data(), f_fn.data(), fn_n * sizeof(bf16_t));
                transpose_bf16(w.hc_ffn_fn, HC_MIX_DIM, HC_DIM);
            }
            auto f_sc = pop_tid_layer(27, l);
            if (f_sc.size() >= 3 * sizeof(float)) { w.hc_ffn_scale.resize(3); memcpy(w.hc_ffn_scale.data(), f_sc.data(), 3 * sizeof(float)); }
            auto f_bs = pop_tid_layer(28, l);
            if (f_bs.size() >= HC_MIX_DIM * sizeof(float)) { w.hc_ffn_base.resize(HC_MIX_DIM); memcpy(w.hc_ffn_base.data(), f_bs.data(), HC_MIX_DIM * sizeof(float)); }
            auto sink_raw = pop_tid_layer(22, l);
            if (sink_raw.size() >= MLA_N_HEADS * sizeof(float)) { w.attn_sinks.resize(MLA_N_HEADS); memcpy(w.attn_sinks.data(), sink_raw.data(), MLA_N_HEADS * sizeof(float)); }
        }
        if (w.attn_norm.empty()) w.attn_norm.assign(draft_cfg_.hidden_dim, f2bf(1.f));
        if (w.ffn_norm.empty()) w.ffn_norm.assign(draft_cfg_.hidden_dim, f2bf(1.f));
    }

    /* Last stage: markov head, confidence head, draft norm */
    int last = draft_cfg_.n_layers - 1;
    if (last >= 0) {
        auto& w = draft_shared_[last];
        /* markov_w1: [vocab, rank] BF16 = 129280 x 256 */
        pb("129280x256x1", DRAFT_GLOBAL_LAYER, w.markov_w1);
        /* markov_w2: [vocab, rank] BF16 = 129280 x 256 (same shape) */
        pb("129280x256x1", DRAFT_GLOBAL_LAYER, w.markov_w2);
        /* confidence_proj: [1, dim+rank] = 1 x 4352 */
        pb("4352x1x1", DRAFT_GLOBAL_LAYER, w.confidence_proj);
        /* draft final norm: [dim] = 4096.  Read from lid=last (the draft's OWN
         * final norm mtp.{last}.norm.weight = TID_DRAFT_NORM @ lid=last), NOT
         * from DRAFT_GLOBAL_LAYER (which previously shadowed it with the MAIN
         * model's top-level norm.weight -- a wrong-norm bug). */
        pb("4096x1x1", last, w.draft_norm);
        /* Last-stage HC output collapse (global TIDs 29-31): hc_head_fn is BF16
         * [N_HC, HC_DIM] = [4, 16384]; transpose to [HC_DIM, N_HC] for npu_gemm_hc.
         * scale is F32 [1]; base is F32 [N_HC=4].  Mirrors the main loader's
         * model_.hc_head_* load. */
        {
            const size_t fn_n = (size_t)HC_DIM * N_HC;
            auto hfn = pop_tid(29);
            if (hfn.size() >= fn_n * sizeof(bf16_t)) {
                w.hc_head_fn.resize(fn_n);
                memcpy(w.hc_head_fn.data(), hfn.data(), fn_n * sizeof(bf16_t));
                transpose_bf16(w.hc_head_fn, N_HC, HC_DIM);  /* [4,16384]->[16384,4] */
            }
            auto hsc = pop_tid(30);
            if (hsc.size() >= 1 * sizeof(float)) { w.hc_head_scale.resize(1); memcpy(w.hc_head_scale.data(), hsc.data(), 1 * sizeof(float)); }
            auto hbs = pop_tid(31);
            if (hbs.size() >= N_HC * sizeof(float)) { w.hc_head_base.resize(N_HC); memcpy(w.hc_head_base.data(), hbs.data(), N_HC * sizeof(float)); }
        }
    }

    /* Global: embedding + lm_head (shared with main model) */
    /* Debug: print all tensor IDs and sizes in the draft FST */
    fprintf(stderr, "[draft] FST tensors (%zu entries):\n", ne);
    for (size_t i = 0; i < ne && i < 30; i++) {
        auto* e = (const FSTE*)(db.data() + i * 64);
        fprintf(stderr, "  tid=%u lid=%u sid=%u qt=%u nd=%u s0=%llu s1=%llu off=%llu sz=%zu\n",
                e->tid, e->lid, e->sid, e->qt, e->nd,
                (unsigned long long)e->s0, (unsigned long long)e->s1,
                (unsigned long long)e->off, (size_t)e->sz);
    }

    {
        auto emb_raw = pop_tid(DRAFT_TID_EMBED);
        if (!emb_raw.empty()) {
            draft_embedding_table_.resize(emb_raw.size() / sizeof(bf16_t));
            memcpy(draft_embedding_table_.data(), emb_raw.data(), emb_raw.size());
        }
    }
    {
        auto lm_raw = pop_tid(DRAFT_TID_LM_HEAD);
        if (!lm_raw.empty()) {
            draft_lm_head_.resize(lm_raw.size() / sizeof(bf16_t));
            memcpy(draft_lm_head_.data(), lm_raw.data(), lm_raw.size());
        }
    }

    ::close(fd);
    fprintf(stderr, "[draft] loaded %d stages with shared weights\n", draft_cfg_.n_layers);
}

void FSTEngine::ensure_draft_layer_loaded(int lid) {
    if (lid == draft_current_loaded_layer_) return;

    if (draft_current_loaded_layer_ >= 0) {
        auto& old = draft_shared_[draft_current_loaded_layer_];
        old.bo_wq_a = {}; old.bo_wq_b = {}; old.bo_wkv = {};
        old.bo_wo_a = {}; old.bo_wo_b = {};
        old.bo_shared_gate = {}; old.bo_shared_up = {}; old.bo_shared_down = {};
        old.bo_main_proj = {};
    }

    auto make_bo = [&](const void* data, size_t bytes) -> xrt::bo {
        if (bytes == 0) return {};
        xrt::bo bo(npu_device_, bytes, xrt::bo::flags::host_only, grp_);
        memcpy(bo.map<char*>(), data, bytes);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return bo;
    };

    auto& w = draft_shared_[lid];
    w.bo_wq_a        = make_bo(w.wq_a.data(), w.wq_a.size() * sizeof(bf16_t));
    w.bo_wq_b        = make_bo(w.wq_b.data(), w.wq_b.size() * sizeof(bf16_t));
    w.bo_wkv         = make_bo(w.wkv.data(), w.wkv.size() * sizeof(bf16_t));
    w.bo_wo_a        = make_bo(w.wo_a.data(), w.wo_a.size() * sizeof(bf16_t));
    w.bo_wo_b        = make_bo(w.wo_b.data(), w.wo_b.size() * sizeof(bf16_t));
    w.bo_shared_gate = make_bo(w.shared_gate.data(), w.shared_gate.size() * sizeof(bf16_t));
    w.bo_shared_up   = make_bo(w.shared_up.data(), w.shared_up.size() * sizeof(bf16_t));
    w.bo_shared_down = make_bo(w.shared_down.data(), w.shared_down.size() * sizeof(bf16_t));

    /* FST_DRAFT_DEBUG: dump stage-0 draft weights (first 5 values + sizes) to
     * verify they are loaded (not swapped/zeroed).  main_proj/main_norm too. */
    if (lid == 0) {
        const char* dv = std::getenv("FST_DRAFT_DEBUG");
        if (dv && std::atoi(dv) >= 1) {
            auto p5 = [](const char* nm, const auto& v) {
                fprintf(stderr, "[draft-w] %-12s size=%zu first5:", nm, v.size());
                int n = std::min<size_t>(5, v.size());
                for (size_t k = 0; k < n; k++) {
                    float f = 0;
                    if constexpr (std::is_same_v<typename std::decay<decltype(v[0])>::type, bf16_t>)
                        f = bf16f(v[k]);
                    else f = (float)v[k];
                    fprintf(stderr, " %.4f", f);
                }
                fprintf(stderr, "\n");
            };
            p5("wq_a", w.wq_a); p5("wo_a", w.wo_a); p5("wq_b", w.wq_b);
            p5("wkv", w.wkv);   p5("wo_b", w.wo_b);
            p5("attn_norm", w.attn_norm); p5("ffn_norm", w.ffn_norm);
            p5("main_norm", w.main_norm); p5("draft_norm", w.draft_norm);
            p5("main_proj", w.main_proj);
            /* router is float [n_experts,hd] after transpose_f32(256,4096) */
            fprintf(stderr, "[draft-w] %-12s size=%zu first5:", "router", w.router.size());
            for (size_t k = 0; k < std::min<size_t>(5, w.router.size()); k++)
                fprintf(stderr, " %.4f", w.router[k]);
            fprintf(stderr, "\n"); fflush(stderr);
        }
    }

    draft_current_loaded_layer_ = lid;
}

void FSTEngine::process_draft_mla(int lid, bf16_t* h, int M, const bf16_t* main_x) {
    /* HF-faithful DSpark attention (decode path, start_pos>0).  The KV comes
     * from main_x (the projected main hidden), NOT the block's own growing
     * causal cache (the old engine bug):
     *   main_kv = kv_norm(wkv(main_x))            [512], rotary @ anchor
     *   kv_window[anchor % win] = main_kv           (sliding window, 1 per token)
     *   kv_block = kv_norm(wkv(x))                  [M,512], rotary @ anchor+1+m
     *   kv = cat([window[0..filled), kv_block])     [filled+M, 512]
     *   o = dense_attn(q, kv, sink)                  (NO causal mask; sink in denom)
     * anchor = seq_pos_ (set by forward_draft).  block[m] is the candidate at
     * position anchor+1+m (block[0] = last-accepted embedding seeds the first
     * candidate, matching the SD verify where draft[0] <-> main_preds[0] are
     * both for seq_pos_+1).  For anchor < win the window slots 0..anchor hold
     * main_kv for positions 0..anchor in order.  Same 4 MLA xclbins (qc/wqb/
     * ob/qksv) as the main model; plain RoPE (theta_scale, no YaRN). */
    auto& w = draft_shared_[lid];
    const int hd = draft_cfg_.hidden_dim, rope_dim = MLA_ROPE_DIM;
    const int n_heads = MLA_N_HEADS, head_dim = MLA_HEAD_DIM;
    const int nope_dim = head_dim - rope_dim;       /* 448 */
    const int win = draft_cfg_.window_size;          /* 128 */
    const int anchor = seq_pos_;
    const int M_PAD = ((M + 15) / 16) * 16;
    const int M_LAT = M_PAD * n_heads;
    const float theta_scale = std::pow(rope_freq_base_, -2.0f / rope_dim);

    if (anchor >= win)
        fprintf(stderr, "[draft-mla] WARNING anchor=%d >= win=%d (circular order unhandled; short-context only)\n",
                anchor, win);
    const int filled = std::min(anchor + 1, win);   /* main_kvs in window (slots 0..anchor) */

    /* ── main_kv from main_x, rotary @ anchor, write to window slot ── */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> mx_pad(M_PAD * hd, 0);
    memcpy(mx_pad.data(), main_x, (size_t)hd * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> main_kv_b(M_PAD * MLA_KV_LORA, 0);
    npu_gemm_mla_vec("qck", main_kv_b.data(), mx_pad.data(), w.wkv.data(),
                     M_PAD, MLA_KV_LORA, hd);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> main_kv(MLA_KV_LORA);
    if (!w.kv_norm.empty())
        npu_rmsnorm_weighted(main_kv.data(), main_kv_b.data(), w.kv_norm.data(), MLA_KV_LORA);
    else
        memcpy(main_kv.data(), main_kv_b.data(), MLA_KV_LORA * sizeof(bf16_t));
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> pe(rope_dim);
        memcpy(pe.data(), main_kv.data() + nope_dim, rope_dim * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> lut(rope_dim);
        float th = (float)anchor;
        for (int d = 0; d < rope_dim; d += 2) {
            lut[d] = f2bf(cosf(th)); lut[d + 1] = f2bf(sinf(th)); th *= theta_scale;
        }
        npu_rope("rope", pe.data(), lut.data(), pe.data(), 1, rope_dim);
        memcpy(main_kv.data() + nope_dim, pe.data(), rope_dim * sizeof(bf16_t));
    }
    auto& kvc = draft_kv_cache_[lid];
    memcpy(kvc.kv_latent.data() + (size_t)(anchor % win) * MLA_KV_LORA,
           main_kv.data(), MLA_KV_LORA * sizeof(bf16_t));

    /* ── q from h (block candidates), rotary @ anchor+1+m ── */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> h_pad(M_PAD * hd, 0);
    memcpy(h_pad.data(), h, (size_t)M * hd * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> qc_host(M_PAD * MLA_Q_LORA, 0);
    npu_gemm_mla_vec("qck", qc_host.data(), h_pad.data(), w.wq_a.data(),
                     M_PAD, MLA_Q_LORA, hd);
    if (!w.q_norm.empty())
        for (int m = 0; m < M; m++)
            npu_rmsnorm_weighted(qc_host.data() + m * MLA_Q_LORA,
                                 qc_host.data() + m * MLA_Q_LORA,
                                 w.q_norm.data(), MLA_Q_LORA);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> q_full(M_PAD * n_heads * head_dim, 0);
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> q_pad(M_PAD * MLA_Q_LORA, 0);
        memcpy(q_pad.data(), qc_host.data(), (size_t)M * MLA_Q_LORA * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> q_tile(M_PAD * 2048);
        const int QN = n_heads * head_dim;
        for (int n_off = 0; n_off < QN; n_off += 2048) {
            npu_gemm_mla_vec("wqb", q_tile.data(), q_pad.data(),
                             w.wq_b.data() + (size_t)n_off * MLA_Q_LORA,
                             M_PAD, 2048, MLA_Q_LORA);
            for (int m = 0; m < M_PAD; m++)
                memcpy(q_full.data() + (size_t)m * QN + n_off,
                       q_tile.data() + (size_t)m * 2048,
                       2048 * sizeof(bf16_t));
        }
    }
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> q_pe(M * rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(q_pe.data() + m * rope_dim,
                   q_full.data() + (size_t)m * n_heads * head_dim + nope_dim,
                   rope_dim * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> lut(M * rope_dim);
        for (int m = 0; m < M; m++) {
            float th = (float)(anchor + 1 + m);
            for (int d = 0; d < rope_dim; d += 2) {
                lut[m * rope_dim + d] = f2bf(cosf(th)); lut[m * rope_dim + d + 1] = f2bf(sinf(th)); th *= theta_scale;
            }
        }
        npu_rope("rope", q_pe.data(), lut.data(), q_pe.data(), M, rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(q_full.data() + (size_t)m * n_heads * head_dim + nope_dim,
                   q_pe.data() + m * rope_dim, rope_dim * sizeof(bf16_t));
    }

    /* ── kv_block from h, rotary @ anchor+1+m ── */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kv_host(M_PAD * MLA_KV_LORA, 0);
    npu_gemm_mla_vec("qck", kv_host.data(), h_pad.data(), w.wkv.data(),
                     M_PAD, MLA_KV_LORA, hd);
    if (!w.kv_norm.empty())
        for (int m = 0; m < M; m++)
            npu_rmsnorm_weighted(kv_host.data() + m * MLA_KV_LORA,
                                 kv_host.data() + m * MLA_KV_LORA,
                                 w.kv_norm.data(), MLA_KV_LORA);
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> k_pe(M * rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(k_pe.data() + m * rope_dim,
                   kv_host.data() + m * MLA_KV_LORA + nope_dim,
                   rope_dim * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> lut(M * rope_dim);
        for (int m = 0; m < M; m++) {
            float th = (float)(anchor + 1 + m);
            for (int d = 0; d < rope_dim; d += 2) {
                lut[m * rope_dim + d] = f2bf(cosf(th)); lut[m * rope_dim + d + 1] = f2bf(sinf(th)); th *= theta_scale;
            }
        }
        npu_rope("rope", k_pe.data(), lut.data(), k_pe.data(), M, rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(kv_host.data() + m * MLA_KV_LORA + nope_dim,
                   k_pe.data() + m * rope_dim, rope_dim * sizeof(bf16_t));
    }

    /* ── KV = [window[0..filled) , kv_block[0..M)] -> [S, 512] ── */
    const int S = filled + M;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kv_all((size_t)S * head_dim, 0);
    for (int s = 0; s < filled; s++)
        memcpy(kv_all.data() + (size_t)s * head_dim,
               kvc.kv_latent.data() + (size_t)s * MLA_KV_LORA,  /* slot s = position s (anchor<win) */
               head_dim * sizeof(bf16_t));
    for (int m = 0; m < M; m++)
        memcpy(kv_all.data() + (size_t)(filled + m) * head_dim,
               kv_host.data() + (size_t)m * MLA_KV_LORA, head_dim * sizeof(bf16_t));

    /* ── qk: q_flat[M_LAT, head_dim] @ kv_all.T[head_dim, S] -> [M_LAT, S] ── */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> q_flat(M_LAT * head_dim, 0);
    memcpy(q_flat.data(), q_full.data(), (size_t)M * n_heads * head_dim * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> kvT((size_t)head_dim * S, 0);
    for (int s = 0; s < S; s++)
        for (int k = 0; k < head_dim; k++)
            kvT[(size_t)k * S + s] = kv_all[(size_t)s * head_dim + k];
    std::vector<bf16_t, AlignedAllocator<bf16_t>> qk_out((size_t)M_LAT * S, 0);
    npu_gemm_mla_vec("qksv", qk_out.data(), q_flat.data(), kvT.data(),
                     M_LAT, S, head_dim);

    /* ── softmax with per-head sink, DENSE (no causal mask) ── */
    constexpr int SC_STRIDE = 512;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> sc((size_t)M_LAT * SC_STRIDE, 0);
    for (int r = 0; r < M_LAT; r++)
        for (int s = 0; s < S; s++)
            sc[(size_t)r * SC_STRIDE + s] = qk_out[(size_t)r * S + s];
    {
        const float kq_scale = 1.0f / sqrtf((float)head_dim);
        for (int m = 0; m < M; m++)
            for (int hh = 0; hh < n_heads; hh++) {
                int row = m * n_heads + hh;
                float sink = w.attn_sinks.empty() ? 0.0f : w.attn_sinks[hh];
                float max_score = sink;
                for (int s = 0; s < S; s++) {
                    float v = bf16f(sc[(size_t)row * SC_STRIDE + s]) * kq_scale;
                    if (v > max_score) max_score = v;
                }
                float sum = expf(sink - max_score);
                for (int s = 0; s < S; s++)
                    sum += expf(bf16f(sc[(size_t)row * SC_STRIDE + s]) * kq_scale - max_score);
                float inv = 1.0f / (sum + 1e-12f);
                for (int s = 0; s < S; s++)
                    sc[(size_t)row * SC_STRIDE + s] =
                        f2bf(expf(bf16f(sc[(size_t)row * SC_STRIDE + s]) * kq_scale - max_score) * inv);
            }
        for (int r = 0; r < M_LAT; r++)
            memset(sc.data() + (size_t)r * SC_STRIDE + S, 0,
                   (size_t)(SC_STRIDE - S) * sizeof(bf16_t));
    }

    /* ── sv: scores[M_LAT,S] @ kv_all[S,head_dim] -> [M_LAT, head_dim] ── */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> scores_packed((size_t)M_LAT * S, 0);
    for (int r = 0; r < M_LAT; r++)
        memcpy(scores_packed.data() + (size_t)r * S,
               sc.data() + (size_t)r * SC_STRIDE,
               (size_t)S * sizeof(bf16_t));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> out_pad(M_LAT * head_dim, 0);
    npu_gemm_mla_vec("qksv", out_pad.data(), scores_packed.data(),
                     kv_all.data(), M_LAT, head_dim, S);

    /* ── inverse rotary on o last 64 dims @ anchor+1+m ── */
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> pe(M * rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(pe.data() + m * rope_dim,
                   out_pad.data() + (size_t)m * n_heads * head_dim + nope_dim,
                   rope_dim * sizeof(bf16_t));
        std::vector<bf16_t, AlignedAllocator<bf16_t>> lut(M * rope_dim);
        for (int m = 0; m < M; m++) {
            float th = (float)(anchor + 1 + m);
            for (int d = 0; d < rope_dim; d += 2) {
                lut[m * rope_dim + d] = f2bf(cosf(th)); lut[m * rope_dim + d + 1] = f2bf(-sinf(th)); th *= theta_scale;
            }
        }
        npu_rope("rope", pe.data(), lut.data(), pe.data(), M, rope_dim);
        for (int m = 0; m < M; m++)
            memcpy(out_pad.data() + (size_t)m * n_heads * head_dim + nope_dim,
                   pe.data() + m * rope_dim, rope_dim * sizeof(bf16_t));
    }

    /* ── wo_a (grouped) + wo_b -> attn_out [M, hd] ── */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> heads_pad(M_LAT * head_dim, 0);
    memcpy(heads_pad.data(), out_pad.data(), (size_t)M_LAT * head_dim * sizeof(bf16_t));
    const int heads_cols = n_heads * head_dim;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> low(M_PAD * 8192, 0);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> heads_g(M_PAD * 4096, 0);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> low_g(M_PAD * 1024, 0);
    for (int g = 0; g < 8; g++) {
        for (int m = 0; m < M_PAD; m++)
            memcpy(heads_g.data() + (size_t)m * 4096,
                   heads_pad.data() + (size_t)m * heads_cols + g * 4096,
                   4096 * sizeof(bf16_t));
        npu_gemm_mla_vec("qck", low_g.data(), heads_g.data(),
                         w.wo_a.data() + (size_t)g * 1024 * 4096,
                         M_PAD, 1024, 4096);
        for (int m = 0; m < M_PAD; m++)
            memcpy(low.data() + (size_t)m * 8192 + g * 1024,
                   low_g.data() + (size_t)m * 1024,
                   1024 * sizeof(bf16_t));
    }
    std::vector<bf16_t, AlignedAllocator<bf16_t>> attn_out(M_PAD * hd, 0);
    {
        std::vector<bf16_t, AlignedAllocator<bf16_t>> ob_tile(M_PAD * 2048, 0);
        for (int n_off = 0; n_off < hd; n_off += 2048) {
            npu_gemm_mla_vec("ob", ob_tile.data(), low.data(),
                             w.wo_b.data() + (size_t)n_off * 8192,
                             M_PAD, 2048, 8192);
            for (int m = 0; m < M_PAD; m++)
                memcpy(attn_out.data() + (size_t)m * hd + n_off,
                       ob_tile.data() + (size_t)m * 2048,
                       2048 * sizeof(bf16_t));
        }
    }

    memcpy(h, attn_out.data(), (size_t)M * hd * sizeof(bf16_t));
}

void FSTEngine::process_draft_expert_ffn(int lid, const bf16_t* h, bf16_t* out, int M) {
    auto& w = draft_shared_[lid];
    const int hd = draft_cfg_.hidden_dim, tk = draft_cfg_.top_k;
    const int Mx = 16, rep = Mx / M, total = M * tk;   // M=16 std
    const int M_PAD = ((M + 15) / 16) * 16;            // M=16 std
    const size_t proj_elems = (size_t)hd * INTER_DIM;
    const size_t proj_sz = proj_elems * sizeof(bf16_t);

    /* Pad hidden state to M_PAD rows for router.  h is the moe-normed input
     * (read-only); the routed result is written to `out` (separate buffer),
     * mirroring the main process_expert_ffn so the normed input is preserved
     * for the shared expert. */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> h_pad(M_PAD * hd, 0);
    memcpy(h_pad.data(), h, (size_t)M * hd * sizeof(bf16_t));

    std::vector<int> eids(M_PAD * tk);
    std::vector<float> ewts(M_PAD * tk);
    npu_router(eids.data(), ewts.data(), h_pad.data(), w.router.data(),
               w.router_bias.empty() ? nullptr : w.router_bias.data(),
               M_PAD, draft_cfg_.n_experts, tk, hd, lid, nullptr);

    std::vector<float> acc(M * hd, 0.0f);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> down_out(Mx * hd);

    /* Upload shared activation (h replicated to Mx rows) ONCE for the
     * whole layer.  bo_scratch_in_ holds hp for every expert because the
     * down GEMM reads its input from sc.bo_mul, not bo_scratch_in_. */
    {
        char* hp = bo_scratch_in_.map<char*>();
        for (int r = 0; r < rep; r++)
            memcpy(hp + (size_t)r * M * hd * sizeof(bf16_t), h, (size_t)M * hd * sizeof(bf16_t));
        int filled = rep * M;
        if (filled < Mx)
            memset(hp + (size_t)filled * hd * sizeof(bf16_t), 0,
                   (size_t)(Mx - filled) * hd * sizeof(bf16_t));
        npu_sync_to(bo_scratch_in_);
    }

    /* ── FST_MC_FFN: batched multi-core path (6 experts in parallel per op). ──
     * Replaces the per-expert 6-dispatch loop with ~12 dispatches per 6-expert
     * batch (6 dequant + gate/up/silu/mul/down + readback).  Same GEMM math as
     * the default path (HW-verified cos 0.9988 == single-core). */
    if (std::getenv("FST_MC_FFN")) {
        std::vector<int> slot_eids;
        std::vector<bool> seen_mc(draft_cfg_.n_experts, false);
        for (int i = 0; i < total; i++) {
            int e = eids[i];
            if (!seen_mc[e]) { seen_mc[e] = true; slot_eids.push_back(e); }
        }
        auto& deq_krnl_mc = kernel_cache_->get("dequant");
        auto prepare_slot = [&](int slot, xrt::bo& out_sub) {
            /* Persistent host_only BO for this draft expert's packed weights:
             * dequant reads it directly — NO per-dispatch memcpy, NO per-dispatch
             * sync_to (BO-cache HIT after warmup).  Mirrors the main FFN path. */
            xrt::bo& wbo = get_draft_expert_bo(lid, slot_eids[slot]);
            auto run = deq_krnl_mc(3, 0, 0,
                static_cast<xrt::bo&>(wbo), static_cast<xrt::bo&>(out_sub),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_),
                static_cast<xrt::bo&>(bo_d3_));
            pending_runs_.push_back(std::move(run));
        };
        auto weight_of = [&](int slot, int m) -> float {
            int e = slot_eids[slot];
            for (int j = 0; j < tk; j++)
                if (eids[m * tk + j] == e) return ewts[m * tk + j];
            return 0.0f;
        };
        run_batched_ffn(M, Mx, hd, slot_eids, prepare_slot, weight_of, acc.data(), lid, false);
        for (int i = 0; i < M * hd; i++) out[i] = f2bf(acc[i]);
        if (std::getenv("FST_DRAFT_FFNC")) {
            int batches = ((int)slot_eids.size() + 5) / 6;
            fprintf(stderr, "[draft-ffn-mc] L%d M=%d total=%d n_unique=%zu batches=%d disp~=%d\n",
                    lid, M, total, slot_eids.size(), batches,
                    (int)slot_eids.size() + batches * 5);
        }
        return;
    }

    auto& deq_krnl = kernel_cache_->get("dequant");
    auto& gemm_krnl = kernel_cache_->get("gemm");
    auto& gemm2_krnl = kernel_cache_->get("gemm2");
    auto& gemm_down_krnl = kernel_cache_->get("gemm_down");

    std::vector<bool> seen(draft_cfg_.n_experts, false);
    int exp_idx = 0;

    for (int i = 0; i < total; i++) {
        int e = eids[i];
        if (seen[e]) continue;
        seen[e] = true;

        ScratchSet& sc = scratch_pool_[exp_idx & 1];   // 2-set ping-pong from the 4-pool
        ++exp_idx;

        /* ── NPU MXFP4 DEQUANT → NPU GEMM CHAIN (BO-to-BO, no host copy) ── */
        /* Persistent host_only BO for this draft expert's packed weights: dequant
         * reads it directly — NO per-dispatch memcpy, NO per-dispatch sync_to of
         * the packed weights (BO-cache HIT after warmup).  Mirrors the main FFN
         * path's get_expert_bo.  Eliminates the ~1.3ms memcpy + sync_to that was
         * charged on every draft expert dispatch. */
        xrt::bo& wbo = get_draft_expert_bo(lid, e);
        npu_sync_to(sc.bo_w);
        npu_sync_to(bo_d1_); npu_sync_to(bo_d2_); npu_sync_to(bo_d3_);

        /* NPU dequant packed → sc.bo_w directly (BO-to-BO, no host copy) */
        {
            auto drun = deq_krnl(3, 0, 0,
                static_cast<xrt::bo&>(wbo),
                static_cast<xrt::bo&>(sc.bo_w),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_), static_cast<xrt::bo&>(bo_d3_));
            pending_runs_.push_back(std::move(drun));
        }
        flush_pending_runs();

        /* Stage 2: gate+up GEMMs (ctx: expert_gemm_vec, b_col_maj).  Reads
         * B[N,K] directly from dequant output (sc.bo_w) — NO transpose. */
        npu_sync_to(sc.bo_ha);
        npu_sync_to(sc.bo_hb);
        {
            xrt::bo bo_dq_gate(sc.bo_w, proj_sz, 0);
            auto run = gemm_krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_dq_gate),
                static_cast<xrt::bo&>(sc.bo_ha),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        {
            xrt::bo bo_dq_up(sc.bo_w, proj_sz, proj_sz);
            auto run = gemm2_krnl(3, 0, 0,
                static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(bo_dq_up),
                static_cast<xrt::bo&>(sc.bo_hb),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        flush_pending_runs();

        /* Stage 3: silu(gate) then silu*up on device (ctx: ew) */
        npu_sync_to(sc.bo_silu);
        npu_sync_to(sc.bo_mul);
        npu_ew_async("silu", sc.bo_ha, sc.bo_silu);
        npu_ew_bin_async("mul", sc.bo_silu, sc.bo_hb, sc.bo_mul);
        flush_pending_runs();

        /* Stage 4: down GEMM (ctx: expert_gemm_down, b_col_maj, N=4096 single
         * call).  B is the full [N=4096, K=2048] down weight in sc.bo_w at
         * 2*proj_sz; the kernel writes C=[Mx,4096] row-major directly into
         * bo_scratch_out_ — NO two-half split, NO host interleave (the old
         * 2-half + flat-memcpy readback mis-interleaved cols 2048..4095 with
         * C_lo row m+1; tolerated in SD only).  Single readback below. */
        npu_sync_to(bo_scratch_out_);
        {
            const size_t full_b = (size_t)hd * INTER_DIM * sizeof(bf16_t);    /* 4096*2048*2 */
            const size_t full_c = (size_t)Mx * hd * sizeof(bf16_t);           /* 32*4096*2  */
            xrt::bo bo_dq_dn(sc.bo_w, full_b, 2 * proj_sz);
            xrt::bo bo_c(bo_scratch_out_, full_c, 0);
            auto run = gemm_down_krnl(3, 0, 0,
                static_cast<xrt::bo&>(sc.bo_mul), static_cast<xrt::bo&>(bo_dq_dn),
                static_cast<xrt::bo&>(bo_c),
                static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
            pending_runs_.push_back(std::move(run));
        }
        flush_pending_runs();

        /* Single readback: down_out for router-weighted accumulation */
        {
            const size_t c_sz = (size_t)Mx * hd * sizeof(bf16_t);
            npu_sync_from(bo_scratch_out_);
            memcpy(down_out.data(), bo_scratch_out_.map<char*>(), c_sz);
        }

        for (int j = 0; j < total; j++) {
            if (eids[j] != e) continue;
            int m = j / tk;
            float wt = ewts[j];
            for (int d = 0; d < hd; d++)
                acc[m * hd + d] += bf16f(down_out[m * hd + d]) * wt;
        }
    }

    if (std::getenv("FST_DRAFT_FFNC"))
        fprintf(stderr, "[draft-ffn] L%d M=%d total=%d n_unique=%d disp=%d (per-exp=%d)\n",
                lid, M, total, exp_idx, exp_idx * 6, exp_idx * 6);

    for (int i = 0; i < M * hd; i++)
        out[i] = f2bf(acc[i]);
}

void FSTEngine::process_draft_shared_expert(int lid, const bf16_t* h, bf16_t* out, int M) {
    auto& w = draft_shared_[lid];
    if (w.shared_gate.empty()) return;

    const int hd = draft_cfg_.hidden_dim;
    int Mx = 16, rep = Mx / M;   // M=16 std

    /* Pure NPU chain, no host math:
     *   gate+up (c1) -> silu+mul (c5) -> down (c2)
     * Only the final down output is read back to add the residual. */
    {
        char* hp = bo_scratch_in_.map<char*>();
        for (int r = 0; r < rep; r++)
            memcpy(hp + (size_t)r * M * hd * sizeof(bf16_t), h, (size_t)M * hd * sizeof(bf16_t));
        int filled = rep * M;
        if (filled < Mx)
            memset(hp + (size_t)filled * hd * sizeof(bf16_t), 0,
                   (size_t)(Mx - filled) * hd * sizeof(bf16_t));
        npu_sync_to(bo_scratch_in_);
    }

    /* ── FST_MC_FFN: draft shared expert via the batched path (1 real slot). */
    if (std::getenv("FST_MC_FFN")) {
        const size_t proj_elems = (size_t)hd * INTER_DIM;
        const size_t proj_sz = proj_elems * sizeof(bf16_t);
        std::vector<int> slot_eids = {0};
        auto prepare_slot = [&](int /*slot*/, xrt::bo& out_sub) {
            char* dst = out_sub.map<char*>();
            std::memcpy(dst,               w.bo_shared_gate.map<char*>(), proj_sz);
            std::memcpy(dst + proj_sz,     w.bo_shared_up.map<char*>(),   proj_sz);
            std::memcpy(dst + 2 * proj_sz, w.bo_shared_down.map<char*>(), proj_sz);
            npu_sync_to(out_sub);
        };
        auto weight_of = [&](int slot, int /*m*/) -> float {
            return (slot == 0) ? 1.0f : 0.0f;
        };
        std::vector<float> acc(M * hd, 0.0f);
        run_batched_ffn(M, Mx, hd, slot_eids, prepare_slot, weight_of, acc.data(), lid, true);
        for (int i = 0; i < M * hd; i++) out[i] = f2bf(acc[i]);
        return;
    }

    ScratchSet& sc = scratch_pool_[0];   // single shared expert — no overlap

    npu_sync_to(sc.bo_ha);
    npu_sync_to(sc.bo_hb);

    auto& gemm_krnl  = kernel_cache_->get("gemm");
    auto& gemm2_krnl = kernel_cache_->get("gemm2");
    auto& gemm_down_krnl = kernel_cache_->get("gemm_down");

    {
        auto run_g = gemm_krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(w.bo_shared_gate),
            static_cast<xrt::bo&>(sc.bo_ha),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run_g));
    }
    {
        auto run_u = gemm2_krnl(3, 0, 0,
            static_cast<xrt::bo&>(bo_scratch_in_), static_cast<xrt::bo&>(w.bo_shared_up),
            static_cast<xrt::bo&>(sc.bo_hb),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run_u));
    }
    flush_pending_runs();

    npu_sync_to(sc.bo_silu);
    npu_sync_to(sc.bo_mul);
    npu_ew_async("silu", sc.bo_ha, sc.bo_silu);
    npu_ew_bin_async("mul", sc.bo_silu, sc.bo_hb, sc.bo_mul);
    flush_pending_runs();

    npu_sync_to(bo_scratch_out_);
    {
        auto run = gemm_down_krnl(3, 0, 0,
            static_cast<xrt::bo&>(sc.bo_mul), static_cast<xrt::bo&>(w.bo_shared_down),
            static_cast<xrt::bo&>(bo_scratch_out_),
            static_cast<xrt::bo&>(bo_d1_), static_cast<xrt::bo&>(bo_d2_));
        pending_runs_.push_back(std::move(run));
    }
    flush_pending_runs();

    {
        const size_t c_sz = (size_t)M * hd * sizeof(bf16_t);
        npu_sync_from(bo_scratch_out_);
        const bf16_t* sx = bo_scratch_out_.map<bf16_t*>();
        for (int i = 0; i < M * hd; i++)
            out[i] = sx[i];
    }
}

void FSTEngine::process_draft_layer(int lid, bf16_t* h, int M, const bf16_t* main_x) {
    /* HF-faithful DSpark block (Block.forward): each sublayer is wrapped by
     * hc_pre (4-stream -> single-stream cur + post/comb gates) and hc_post
     * (gated block output + combined old streams -> new 4-stream).  The
     * sublayers (attn/ffn) operate on the single [M, hd] stream.  main_x (the
     * projected main hidden, shared across stages) feeds the attention KV.
     * h is the 4-stream HC residual [M, N_HC, hd].  Falls back to a plain
     * stream-0 add when HC weights are absent. */
    ensure_draft_layer_loaded(lid);
    auto& w = draft_shared_[lid];
    const int hd = draft_cfg_.hidden_dim;
    const int hc_stride = N_HC * hd;
    const bool has_hc = !w.hc_attn_fn.empty();

    std::vector<bf16_t, AlignedAllocator<bf16_t>> an(M * hd), mn(M * hd), cur(M * hd);
    std::vector<float> post_a(M * N_HC), comb_a(M * N_HC * N_HC);
    std::vector<float> post_f(M * N_HC), comb_f(M * N_HC * N_HC);
    std::vector<bf16_t, AlignedAllocator<bf16_t>> new_hc((size_t)M * hc_stride);

    /* ── Attention sublayer ── */
    if (has_hc)
        hc_pre(M, h, w.hc_attn_fn.data(), w.hc_attn_scale.data(), w.hc_attn_base.data(),
               cur.data(), post_a.data(), comb_a.data());
    else
        for (int m = 0; m < M; m++)
            memcpy(cur.data() + m * hd, h + (size_t)m * hc_stride, (size_t)hd * sizeof(bf16_t));
    for (int m = 0; m < M; m++)
        npu_rmsnorm_weighted(an.data() + m * hd, cur.data() + m * hd, w.attn_norm.data(), hd);
    process_draft_mla(lid, an.data(), M, main_x);   /* an = attn_out; KV from main_x */
    if (has_hc) {
        hc_post(M, an.data(), h, post_a.data(), comb_a.data(), new_hc.data());
        memcpy(h, new_hc.data(), (size_t)M * hc_stride * sizeof(bf16_t));
    } else {
        for (int m = 0; m < M; m++)
            for (int d = 0; d < hd; d++)
                h[(size_t)m * hc_stride + d] = f2bf(bf16f(h[(size_t)m * hc_stride + d]) + bf16f(an[m * hd + d]));
    }

    /* ── FFN sublayer ── */
    if (has_hc)
        hc_pre(M, h, w.hc_ffn_fn.data(), w.hc_ffn_scale.data(), w.hc_ffn_base.data(),
               cur.data(), post_f.data(), comb_f.data());
    else
        for (int m = 0; m < M; m++)
            memcpy(cur.data() + m * hd, h + (size_t)m * hc_stride, (size_t)hd * sizeof(bf16_t));
    for (int m = 0; m < M; m++)
        npu_rmsnorm_weighted(mn.data() + m * hd, cur.data() + m * hd, w.ffn_norm.data(), hd);
    /* Routed + shared on the moe-normed input (separate out buffers, mirrors
     * the main process_layer so mn stays the normed input for BOTH experts). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> routed_out(M * hd, f2bf(0.0f));
    std::vector<bf16_t, AlignedAllocator<bf16_t>> shared_out(M * hd, f2bf(0.0f));
    process_draft_expert_ffn(lid, mn.data(), routed_out.data(), M);
    process_draft_shared_expert(lid, mn.data(), shared_out.data(), M);
    flush_pending_runs();
    std::vector<bf16_t, AlignedAllocator<bf16_t>> ffn_out(M * hd);
    for (int i = 0; i < M * hd; i++)
        ffn_out[i] = f2bf(bf16f(routed_out[i]) + bf16f(shared_out[i]));
    if (has_hc) {
        hc_post(M, ffn_out.data(), h, post_f.data(), comb_f.data(), new_hc.data());
        memcpy(h, new_hc.data(), (size_t)M * hc_stride * sizeof(bf16_t));
    } else {
        for (int m = 0; m < M; m++)
            for (int d = 0; d < hd; d++)
                h[(size_t)m * hc_stride + d] = f2bf(bf16f(h[(size_t)m * hc_stride + d]) + bf16f(ffn_out[m * hd + d]));
    }

    /* FST_DRAFT_DEBUG>=2: stage-0 stream-0 rms after attn/ffn. */
    if (lid == 0) {
        const char* dv = std::getenv("FST_DRAFT_DEBUG");
        if (dv && std::atoi(dv) >= 2) {
            auto rms = [](const bf16_t* p, int n) {
                double s = 0; for (int i = 0; i < n; i++) s += (double)bf16f(p[i]) * bf16f(p[i]);
                return std::sqrt(s / n);
            };
            fprintf(stderr, "[draft-s0] post-attn an rms=%.3f | post-ffn ffn_out rms=%.3f | h[s0] rms=%.3f\n",
                    rms(an.data(), hd), rms(ffn_out.data(), hd), rms(h, hd));
            fflush(stderr);
        }
    }
}

FSTEngine::DraftResult FSTEngine::forward_draft(const bf16_t* main_hidden,
                                                  int last_token_id,
                                                  float temperature) {
    /* Hard reset all hw_contexts to maximize capacity for draft model kernels.
     * The draft model needs many xclbins (qc, kvc, oa, ob, qk, sv, router, etc.)
     * and we need to ensure we don't exceed the 9-context driver limit. */
    kernel_cache_->hard_reset();

    DraftResult result;
    const int gamma = draft_cfg_.block_size - 1;  /* 4 candidate tokens */
    const int hd = draft_cfg_.hidden_dim;
    const int V = draft_cfg_.vocab_size;
    const int rank = draft_cfg_.markov_rank;

    /* ── main_x = main_norm(main_proj(main_hidden[12288])) -> [hd], computed ONCE.
     * HF forward_embed: main_x is shared across all 3 DSpark stages and feeds
     * the attention KV (main_kv = kv_norm(wkv(main_x))).  main_proj/main_norm
     * exist only at stage 0 (mtp.0).  The anchor is the LAST ACCEPTED token's
     * position = main seq_pos_ - 1 (the main model's seq_pos_ at call time is the
     * bonus position, one past the last accepted); the draft predicts positions
     * anchor+1..anchor+block_size.  The iter-1 content (main_hidden for the first
     * sampled token, which the main has not yet processed) is approximated by
     * the last prefill token's HC-mean — a bounded off-by-one that self-corrects
     * from iter 2 on (main_hidden then comes from the previous verify's last
     * accepted token). */
    auto main_x_vec = compute_draft_main_x(1, main_hidden);   /* [1, hd] */
    if (draft_main_x_.size() != (size_t)hd) draft_main_x_.assign(hd, f2bf(0.0f));
    memcpy(draft_main_x_.data(), main_x_vec.data(), (size_t)hd * sizeof(bf16_t));

    /* Step 1: Create draft input — [last_token, noise, noise, noise, noise].
     * HF forward_embed: draft_input_ids[:,0] = input_ids (the known token at the
     * anchor), the rest are noise_token_id.  Embed + HC-expand to 4 streams. */
    std::vector<int> draft_input_ids(draft_cfg_.block_size);
    draft_input_ids[0] = last_token_id;
    for (int i = 1; i < draft_cfg_.block_size; i++)
        draft_input_ids[i] = draft_cfg_.noise_token_id;

    std::vector<bf16_t, AlignedAllocator<bf16_t>> draft_h((size_t)draft_cfg_.block_size * N_HC * hd);
    for (int i = 0; i < draft_cfg_.block_size; i++) {
        int tid = draft_input_ids[i];
        std::vector<bf16_t, AlignedAllocator<bf16_t>> emb(hd, f2bf(0.0f));
        if (tid >= 0 && tid < V && !draft_embedding_table_.empty())
            memcpy(emb.data(), draft_embedding_table_.data() + (size_t)tid * hd,
                   hd * sizeof(bf16_t));
        for (int s = 0; s < N_HC; s++)
            memcpy(draft_h.data() + ((size_t)i * N_HC + s) * hd,
                   emb.data(), (size_t)hd * sizeof(bf16_t));
    }

    /* Step 2: Run through draft stages (HF forward_spec loop over mtp).
     * The draft block is a mini-PREFILL (causal within the block).  anchor =
     * last accepted position; set the draft's working seq_pos_ = anchor so
     * process_draft_mla writes the anchor's main_kv to slot anchor%win and
     * builds block candidates at anchor+1+m. */
    int saved_seq_pos = seq_pos_;
    int anchor = seq_pos_ - 1;            /* last accepted token's position */
    if (anchor < 0) anchor = 0;
    seq_pos_ = anchor;
    is_prefill_ = true;                    /* causal mask within the 5-token block */

    for (int l = 0; l < draft_cfg_.n_layers; l++) {
        fprintf(stderr, "[draft] forward_draft: layer %d, M=%d, anchor=%d\n",
                l, draft_cfg_.block_size, anchor);
        process_draft_layer(l, draft_h.data(), draft_cfg_.block_size, draft_main_x_.data());
        fprintf(stderr, "[draft] forward_draft: layer %d done\n", l);
    }

    /* Step 3: forward_head — hc_head collapse 4-stream -> plain, then draft
     * final norm + LM head.  Uses the DRAFT's own hc_head_fn/scale/base (TIDs
     * 29-31, loaded at the last stage), NOT the main model's. */
    auto& last_w = draft_shared_[draft_cfg_.n_layers - 1];
    const int bs = draft_cfg_.block_size;
    const int bs_pad = ((bs + 15) / 16) * 16;  /* M=16 std */

    fprintf(stderr, "[draft] post-layer: bs=%d bs_pad=%d hd=%d n_layers=%d anchor=%d\n",
            bs, bs_pad, hd, draft_cfg_.n_layers, anchor);
    fprintf(stderr, "[draft] last_w.draft_norm.size=%zu hc_head_fn.size=%zu\n",
            last_w.draft_norm.size(), last_w.hc_head_fn.size());

    /* hc_head collapse [bs, N_HC, hd] -> [bs, hd] (mirror output_hc_head with
     * the draft's weights; falls back to stream-0 copy when HC head absent). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> draft_plain(bs * hd, f2bf(0.0f));
    if (last_w.hc_head_fn.empty()) {
        for (int i = 0; i < bs; i++)
            memcpy(draft_plain.data() + i * hd,
                   draft_h.data() + (size_t)i * N_HC * hd, (size_t)hd * sizeof(bf16_t));
    } else {
        std::vector<float, AlignedAllocator<float>> flat((size_t)bs * HC_DIM);
        for (int i = 0; i < bs; i++)
            hc_rmsnorm_no_weight(flat.data() + (size_t)i * HC_DIM,
                                 draft_h.data() + (size_t)i * N_HC * hd, HC_DIM, HC_EPS);
        std::vector<bf16_t, AlignedAllocator<bf16_t>> flat_bf((size_t)bs * HC_DIM);
        for (size_t i = 0; i < (size_t)bs * HC_DIM; i++) flat_bf[i] = f2bf(flat[i]);
        std::vector<bf16_t, AlignedAllocator<bf16_t>> pre_bf((size_t)bs * N_HC, 0);
        for (int m0 = 0; m0 < bs; m0 += 8) {
            int mc = std::min(8, bs - m0);
            std::vector<bf16_t, AlignedAllocator<bf16_t>> a_chunk(8 * HC_DIM, 0);
            memcpy(a_chunk.data(), flat_bf.data() + (size_t)m0 * HC_DIM,
                   (size_t)mc * HC_DIM * sizeof(bf16_t));
            std::vector<bf16_t, AlignedAllocator<bf16_t>> c_chunk(8 * N_HC, 0);
            npu_gemm_hc(c_chunk.data(), a_chunk.data(), last_w.hc_head_fn.data(), 8, N_HC, HC_DIM);
            memcpy(pre_bf.data() + (size_t)m0 * N_HC, c_chunk.data(),
                   (size_t)mc * N_HC * sizeof(bf16_t));
        }
        const float scale = last_w.hc_head_scale.empty() ? 1.0f : last_w.hc_head_scale[0];
        const float* base = last_w.hc_head_base.empty() ? nullptr : last_w.hc_head_base.data();
        for (int i = 0; i < bs; i++) {
            float w[N_HC];
            for (int h = 0; h < N_HC; h++) {
                float z = bf16f(pre_bf[(size_t)i * N_HC + h]) * scale + (base ? base[h] : 0.0f);
                w[h] = 1.0f / (1.0f + expf(-z)) + HC_EPS;
            }
            const bf16_t* r = draft_h.data() + (size_t)i * N_HC * hd;
            bf16_t* o = draft_plain.data() + i * hd;
            for (int d = 0; d < hd; d++) {
                float acc = 0.0f;
                for (int h = 0; h < N_HC; h++) acc += w[h] * bf16f(r[(size_t)h * hd + d]);
                o[d] = f2bf(acc);
            }
        }
    }

    /* Apply draft final norm (per-token). */
    std::vector<bf16_t, AlignedAllocator<bf16_t>> default_norm(hd, f2bf(1.0f));
    const bf16_t* norm_w = last_w.draft_norm.data();
    if (last_w.draft_norm.size() == 0) {
        fprintf(stderr, "[draft] WARNING: draft_norm empty, using identity norm\n");
        norm_w = default_norm.data();
    }
    for (int i = 0; i < bs; i++)
        npu_rmsnorm_weighted(draft_plain.data() + i * hd,
                             draft_plain.data() + i * hd,
                             norm_w, hd);

    /* LM head on NPU: lm_head_gemm kernel, N-tiled by npu_gemm.
     * draft_lm_head_ is [V, D] row-major; the kernel needs B as [D, N_PAD]. */
    const int N_PAD = lm_head_n_pad_;
    std::vector<bf16_t, AlignedAllocator<bf16_t>> hs_pad(bs_pad * hd, 0);
    memcpy(hs_pad.data(), draft_plain.data(), (size_t)bs * hd * sizeof(bf16_t));

    if (draft_lm_head_padded_.empty() &&
        draft_lm_head_.size() == (size_t)V * hd) {
        draft_lm_head_padded_.assign((size_t)N_PAD * hd, (bf16_t)0);
        for (int v = 0; v < V; v++)
            for (int d = 0; d < hd; d++)
                draft_lm_head_padded_[(size_t)d * N_PAD + v] = draft_lm_head_[v * hd + d];
    }

    fprintf(stderr, "[draft] LM head: bs=%d bs_pad=%d hd=%d V=%d N_PAD=%d\n",
            bs, bs_pad, hd, V, N_PAD);

    std::vector<bf16_t, AlignedAllocator<bf16_t>> bf16_out(bs_pad * N_PAD, 0);
    if (!draft_lm_head_padded_.empty()) {
        npu_gemm("lm_head_gemm", bf16_out.data(), hs_pad.data(),
                 draft_lm_head_padded_.data(), bs_pad, N_PAD, hd);
    }

    std::vector<float> all_logits(bs * V);
    for (int m = 0; m < bs; m++) {
        for (int v = 0; v < V; v++) {
            float acc = bf16f(bf16_out[m * N_PAD + v]);
            if (__builtin_isnan(acc) || __builtin_isinf(acc)) acc = 0.0f;
            all_logits[m * V + v] = acc;
        }
    }
    fprintf(stderr, "[debug] After LM Head GEMM (NPU)\n");

    /* Step 4: Markov head — sequentially generate tokens (greedy argmax for SD).
     * SD verify accepts draft.tokens[i] ONLY if it EQUALS the main argmax
     * main_preds[i]; a sampled draft token almost never equals the argmax, so
     * we use greedy argmax (draft argmax vs main argmax). */
    result.tokens.resize(gamma);
    result.confidence.resize(gamma);

    std::vector<int> output_ids(draft_cfg_.block_size + 1);
    output_ids[0] = last_token_id;

    fprintf(stderr, "[draft] Markov head: w1.size=%zu w2.size=%zu rank=%d V=%d\n",
            last_w.markov_w1.size(), last_w.markov_w2.size(), rank, V);

    for (int i = 0; i < gamma; i++) {
        float* logits = all_logits.data() + (i + 1) * V;
        int prev_tok = output_ids[i];
        fprintf(stderr, "[draft] Markov i=%d prev_tok=%d\n", i, prev_tok);
        if (prev_tok >= 0 && prev_tok < V && !last_w.markov_w1.empty() && !last_w.markov_w2.empty()) {
            const bf16_t* emb = last_w.markov_w1.data() + (size_t)prev_tok * rank;
            for (int v = 0; v < V; v++) {
                const bf16_t* w2 = last_w.markov_w2.data() + (size_t)v * rank;
                float dot = 0.0f;
                for (int r = 0; r < rank; r++)
                    dot += bf16f(emb[r]) * bf16f(w2[r]);
                logits[v] += dot;
            }
        }
        int ln = 0;
        for (int v = 0; v < V && v < 100; v++) if (logits[v] != logits[v]) ln++;
        if (ln > 0) fprintf(stderr, "[draft] NaN in logits[%d] before argmax: %d/100\n", i, ln);
        int tok = 0; float best = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > best) { best = logits[v]; tok = v; }
        output_ids[i + 1] = tok;
        result.tokens[i] = tok;
    }

    /* Step 5: Confidence scores — max softmax probability per candidate. */
    for (int i = 0; i < gamma; i++) {
        float* logits = all_logits.data() + (i + 1) * V;
        float mx = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > mx) mx = logits[v];
        float sum = 0.0f;
        for (int v = 0; v < V; v++) { logits[v] = expf(logits[v] - mx); sum += logits[v]; }
        float max_prob = 0.0f;
        for (int v = 0; v < V; v++) {
            float p = logits[v] / sum;
            if (p > max_prob) max_prob = p;
        }
        result.confidence[i] = max_prob;
    }
    fprintf(stderr, "[debug] After Markov Head\n");

    /* FST_DRAFT_DEBUG>=2: dump per-position argmax + top-5 logits. */
    {
        const char* dv = std::getenv("FST_DRAFT_DEBUG");
        if (dv && std::atoi(dv) >= 2) {
            for (int i = 0; i < gamma; i++) {
                float* lg = all_logits.data() + (i + 1) * V;
                int top[5]; float topv[5];
                for (int t = 0; t < 5; t++) { top[t] = -1; topv[t] = -1e30f; }
                for (int v = 0; v < V; v++) {
                    float l = lg[v];
                    for (int t = 0; t < 5; t++) if (l > topv[t]) {
                        for (int k = 4; k > t; k--) { topv[k]=topv[k-1]; top[k]=top[k-1]; }
                        topv[t]=l; top[t]=v; break;
                    }
                }
                fprintf(stderr, "[draft-lm] pos=%d draft_tok=%d(conf=%.3f) top5:",
                        i + 1, result.tokens[i], result.confidence[i]);
                for (int t = 0; t < 5; t++)
                    fprintf(stderr, " %d(%.2f)", top[t], topv[t]);
                fprintf(stderr, "\n"); fflush(stderr);
            }
        }
    }

    /* Restore main model's seq_pos and prefill flag (draft uses its own). */
    seq_pos_ = saved_seq_pos;
    is_prefill_ = false;

    return result;
}

void FSTEngine::generate_dspark(const std::vector<int>& token_ids, int mt,
                                 float temperature, float top_p,
                                 void(*cb)(int,const char*,void*), void* u,
                                 bool skip_prefill) {
    if (!draft_loaded_)
        throw std::runtime_error("Draft model not loaded — call load_draft_model() first");

    /* ── Controlled probe of lm_head_gemm kernel + multi-tile dispatch ──
     * ones[8,4096] @ ones[4096,Np] -> every output element == K=4096.  Uses the
     * SAME npu_gemm("lm_head_gemm", ...) path the engine uses (multi-tile when
     * Np>2048).  If this doesn't read ~4096, the lm_head kernel / multi-tile
     * dispatch is broken independent of any model weights.  */
    if (std::getenv("FST_AUDIT")) {
        for (int Np : {2048, 4096}) {
            const int Mp = 8, Kp = 4096;
            std::vector<bf16_t, AlignedAllocator<bf16_t>> A((size_t)Mp * Kp), B((size_t)Kp * Np), C((size_t)Mp * Np);
            for (auto& x : A) x = f2bf(1.0f);
            for (auto& x : B) x = f2bf(1.0f);
            for (auto& x : C) x = f2bf(0.0f);
            npu_gemm("lm_head_gemm", C.data(), A.data(), B.data(), Mp, Np, Kp);
            int nok = 0; float mn = 1e30f, mx = -1e30f;
            for (size_t i = 0; i < (size_t)Mp * Np; i++) {
                float v = bf16f(C[i]);
                if (std::fabs(v - 4096.0f) < 8.0f) nok++;
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            fprintf(stderr, "[lm-probe] ones@ones Mp=8 K=4096 Np=%d (%s): %d/%d ok mn=%.2f mx=%.2f (expect ~4096)\n",
                    Np, (Np <= 2048 ? "1 tile" : (Np <= 4096 ? "2 tiles" : "64 tiles")),
                    nok, Mp * Np, mn, mx);
            fflush(stderr);
        }
        /* router_gemm probe: same builder, N=256 (outer 4 x inner 64 = 256 total
         * iters, under the ~256 AIE unroll limit).  If this reads ~4096 while
         * lm_head_gemm reads 64, the lm_head N=2048 specialization is truncated
         * by the AIE program-memory unroll limit (32 x 64 = 2048 total iters).  */
        {
            const int Mp = 8, Kp = 4096, Np = 256;
            std::vector<bf16_t, AlignedAllocator<bf16_t>> A((size_t)Mp * Kp), B((size_t)Kp * Np), C((size_t)Mp * Np);
            for (auto& x : A) x = f2bf(1.0f);
            for (auto& x : B) x = f2bf(1.0f);
            for (auto& x : C) x = f2bf(0.0f);
            npu_gemm("router_gemm", C.data(), A.data(), B.data(), Mp, Np, Kp);
            int nok = 0; float mn = 1e30f, mx = -1e30f;
            for (size_t i = 0; i < (size_t)Mp * Np; i++) {
                float v = bf16f(C[i]);
                if (std::fabs(v - 4096.0f) < 8.0f) nok++;
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            fprintf(stderr, "[router-probe] ones@ones Mp=8 K=4096 Np=256: %d/%d ok mn=%.2f mx=%.2f (expect ~4096)\n",
                    nok, Mp * Np, mn, mx);
            fflush(stderr);
        }
    }

    int N = mt > 0 ? mt : 3;
    const int gamma = draft_cfg_.block_size - 1;  /* 4 */
    const int hd = config_.hidden_dim;
    const float confidence_threshold = 0.5f;  /* survival prob threshold */

    auto t0 = std::chrono::steady_clock::now();

    try {
        /* ── PREFILL: process the prompt through the main model ── */
        std::vector<bf16_t, AlignedAllocator<bf16_t>> hidden;
        int M = (int)token_ids.size();
        int prefill_M = M;
        int tid = 0;

        if (skip_prefill) {
            /* FAST DEBUG MODE: skip the 43-layer prefill entirely.
             * Inject a zero hidden state for M=1 and fake first token.
             * This lets us test the SD loop (draft + verify) in seconds. */
            fprintf(stderr, "[DSpark] SKIP-PREFILL: injecting fake hidden state\n");
            hidden.resize(hd, f2bf(0.0f));
            M = 1;
            prefill_M = 1;
            seq_pos_ = 1;
            is_prefill_ = false;
            tid = 46;  /* arbitrary token ID */
            if (cb) cb(tid, "", u);
            fprintf(stderr, "[DSpark] Fake first token: %d\n", tid);
        } else {
            hidden = forward_embeddings(token_ids);
            /* Expand to the 4-stream HC residual [M, N_HC, hd] (each stream
             * starts as a copy of the plain embedding).  Without this, hidden
             * is only [M, hd] but prefill_tiled indexes it with hc_stride =
             * N_HC*hd, reading/writing ~4x past the end -> heap corruption. */
            {
                std::vector<bf16_t, AlignedAllocator<bf16_t>> hc((size_t)M * N_HC * hd);
                for (int m = 0; m < M; m++)
                    for (int s = 0; s < N_HC; s++)
                        memcpy(hc.data() + ((size_t)m * N_HC + s) * hd,
                               hidden.data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
                hidden = std::move(hc);
            }
            is_prefill_ = true;

            fprintf(stderr, "[DSpark] Prefill: %d tokens through %d layers (M=16 tiled)...\n", M, config_.n_layers);
            prefill_tiled(hidden.data(), M, token_ids.data());

            fprintf(stderr, "[DSpark] Prefill complete, sampling first token...\n");
            /* Get the first token from the main model */
            std::vector<float> logits(config_.vocab_size);
            apply_final_norm_and_lm_head(hidden.data() + (size_t)(M - 1) * N_HC * hd, 1, logits.data());
            tid = sample_token(logits.data(), config_.vocab_size, temperature, top_p);
            if (cb) cb(tid, "", u);
            fprintf(stderr, "[DSpark] First token: %d\n", tid);
        }

        /* Draft prefill: process the prompt through the draft model to
         * build its KV cache.  The draft model needs to have seen the
         * same context as the main model before it can generate draft
         * tokens in the decode phase.  Skip in fast debug mode. */
        if (skip_prefill) {
            draft_seq_pos_ = 1;
            fprintf(stderr, "[DSpark] SKIP-PREFILL: draft KV cache skipped\n");
        } else {
            /* HF-faithful draft prefill: the draft's sliding-window KV is filled
             * from main_x = main_norm(main_proj(main_hidden)) where main_hidden
             * is the concat of the 3 target layers' HC-means captured during the
             * MAIN prefill (target_hc_[0..2], each [M, hd]).  No attention is run
             * — the draft's own block KV is recomputed each decode forward; only
             * the per-token main_kv (one per prompt position) persists in the
             * window.  NOTE: relies on a single-slab main prefill (prompt<=16);
             * slabbed prefill would mis-index target_hc_ (capture stores
             * slab-relative rows). */
            int draft_M = M;
            bool have_target = !target_hc_[0].empty() && !target_hc_[1].empty() &&
                               !target_hc_[2].empty() &&
                               (int)target_hc_[0].size() >= draft_M * hd &&
                               (int)target_hc_[1].size() >= draft_M * hd &&
                               (int)target_hc_[2].size() >= draft_M * hd;
            if (have_target) {
                std::vector<bf16_t, AlignedAllocator<bf16_t>> main_hidden((size_t)draft_M * 3 * hd);
                for (int m = 0; m < draft_M; m++) {
                    memcpy(main_hidden.data() + (size_t)m * 3 * hd,
                           target_hc_[0].data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
                    memcpy(main_hidden.data() + (size_t)m * 3 * hd + hd,
                           target_hc_[1].data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
                    memcpy(main_hidden.data() + (size_t)m * 3 * hd + 2 * hd,
                           target_hc_[2].data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
                }
                auto main_x = compute_draft_main_x(draft_M, main_hidden.data());  /* [M, hd] */
                fprintf(stderr, "[DSpark] Draft prefill: filling KV window for %d positions...\n", draft_M);
                fill_draft_window(draft_M, main_x.data(), 0);
                fprintf(stderr, "[DSpark] Draft prefill complete (KV window filled).\n");
            } else {
                fprintf(stderr, "[DSpark] Draft prefill: target_hc_ missing (size %zu/%zu/%zu for M=%d) "
                                "— KV window left empty; decode will self-fill.\n",
                        target_hc_[0].size(), target_hc_[1].size(), target_hc_[2].size(), draft_M);
            }
            draft_seq_pos_ = draft_M;  /* advance draft position */
            is_prefill_ = false;
        }

        auto t1 = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        seq_pos_ += M;
        /* The first draft conditions on the LAST PREFILL token's main_hidden
         * (target_hc_ row M-1).  The first sampled token has not yet been run
         * through the main layers, so its own HC is unavailable — this is the
         * bounded iter-1 off-by-one; iter 2+ use the previous verify's last
         * accepted token (set to n_accepted after each accept). */
        target_anchor_idx_ = M - 1;

        int tokens_generated = 1;
        is_prefill_ = false;

        /* ── DECODE LOOP with DSpark speculative decoding ── */
        int sd_iter = 0;
        while (tokens_generated < N) {
            fprintf(stderr, "[DSpark] SD iter %d: drafting %d tokens...\n", sd_iter++, gamma);
            /* a) DRAFT PHASE: generate gamma candidate tokens + confidence.
             * main_hidden[12288] = concat of the 3 target layers' HC-means for
             * the last accepted token (target_hc_ row target_anchor_idx_).
             * forward_draft computes main_x = main_norm(main_proj(main_hidden))
             * once and feeds the draft attention KV from it (HF-faithful). */
            kernel_cache_->set_phase(AiebuKernelCache::DRAFT_PHASE);
            std::vector<bf16_t, AlignedAllocator<bf16_t>> main_hidden(3 * hd, f2bf(0.0f));
            int ai = target_anchor_idx_;
            if (!target_hc_[0].empty() && ai >= 0 &&
                (int)target_hc_[0].size() >= (ai + 1) * hd &&
                (int)target_hc_[1].size() >= (ai + 1) * hd &&
                (int)target_hc_[2].size() >= (ai + 1) * hd) {
                memcpy(main_hidden.data(), target_hc_[0].data() + (size_t)ai * hd, (size_t)hd * sizeof(bf16_t));
                memcpy(main_hidden.data() + hd, target_hc_[1].data() + (size_t)ai * hd, (size_t)hd * sizeof(bf16_t));
                memcpy(main_hidden.data() + 2 * hd, target_hc_[2].data() + (size_t)ai * hd, (size_t)hd * sizeof(bf16_t));
            } else {
                fprintf(stderr, "[DSpark] WARNING: target_hc_ row %d missing (sz %zu/%zu/%zu) — zero main_hidden\n",
                        ai, target_hc_[0].size(), target_hc_[1].size(), target_hc_[2].size());
            }
            DraftResult draft = forward_draft(main_hidden.data(), tid, temperature);

            fprintf(stderr, "[DSpark] Draft tokens:");
            for (int i = 0; i < gamma; i++)
                fprintf(stderr, " %d(%.2f)", draft.tokens[i], draft.confidence[i]);
            fprintf(stderr, "\n");

            /* b) SCHEDULE: determine verification length based on confidence.
             * Keep tokens with survival probability > threshold. */
            int scheduled_len = 0;
            for (int i = 0; i < gamma; i++) {
                if (draft.confidence[i] > confidence_threshold)
                    scheduled_len++;
                else
                    break;  /* stop at first low-confidence token */
            }
            /* Always verify at least 1 token (the first draft token) */
            if (scheduled_len == 0) scheduled_len = 1;
            /* Cap at gamma */
            if (scheduled_len > gamma) scheduled_len = gamma;

            sd_total_draft_tokens_ += scheduled_len;

            /* c) VERIFY PHASE: pass M = 1 + scheduled_len tokens to main model.
             * The first token is the last accepted token (re-verify), followed
             * by the draft tokens. */
            int verify_M = 1 + scheduled_len;
            std::vector<int> verify_tokens(verify_M);
            verify_tokens[0] = tid;  /* last accepted token */
            for (int i = 0; i < scheduled_len; i++)
                verify_tokens[i + 1] = draft.tokens[i];

            /* d) Run the 43-layer pipeline with M=verify_M.
             * verify_plain: [verify_M, hd] plain embeddings (for draft-driven
             * prefetch).  verify_hidden: [verify_M, N_HC, hd] 4-stream HC
             * residual (for process_layer, which indexes with hc_stride). */
            std::vector<bf16_t, AlignedAllocator<bf16_t>> verify_plain = forward_embeddings(verify_tokens);
            std::vector<bf16_t, AlignedAllocator<bf16_t>> verify_hidden((size_t)verify_M * N_HC * hd);
            for (int m = 0; m < verify_M; m++)
                for (int s = 0; s < N_HC; s++)
                    memcpy(verify_hidden.data() + ((size_t)m * N_HC + s) * hd,
                           verify_plain.data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));

            /* Draft-driven prefetch: queue the predicted experts for EVERY
             * verify layer up front.  NO barrier — the background worker loads
             * them from SSD while the NPU runs.  When process_layer(L) calls
             * get(L, eid), it blocks on a per-expert condition variable ONLY
             * if that specific expert isn't cached yet, allowing the NPU to
             * process other layers/experts in the meantime.  This avoids the
             * >50 GB RAM / swap issue caused by loading all 258 experts
             * (3.4 GB) upfront before any NPU work starts. */
            if (pager_ && scheduled_len > 0) {
                bf16_t* draft_embs = verify_plain.data() + hd;  /* skip first token */
                for (int pl = 0; pl < config_.n_layers; pl++)
                    pager_->predict_and_prefetch_from_draft(pl, draft.tokens.data(),
                                                             scheduled_len, draft_embs,
                                                             hd, config_.n_experts, config_.top_k);
            }

            fprintf(stderr, "[DSpark] Verify phase: M=%d tokens through %d layers...\n", verify_M, config_.n_layers);
            kernel_cache_->set_phase(AiebuKernelCache::MAIN_PHASE);
            /* Snapshot the V4 compressor state before the verify block.  The
             * verify mini-prefill calls compress_token for each verify token,
             * advancing the per-layer state + emitting comp rows.  On
             * rejection we restore this snapshot so the rejected suffix's comp
             * rows + state mutations are undone (mirrors the KV-cache rollback
             * below).  NOTE: the accepted prefix's comp rows are also dropped
             * on a rejection round and not re-emitted (Phase 6: replay accepted
             * prefix from saved attn-normed hidden); the raw KV cache still
             * covers those positions, so for S<128 the effect is minor. */
            auto cmp_snap = cmp_state_;
            /* The verify block is a mini-prefill: token i must attend only to
             * tokens <= i within the block (causal).  is_prefill_=true makes
             * process_mla apply its `s > seq_pos_+m` mask — WITHOUT this, the
             * re-run of `tid` at position 0 attends to the draft token at
             * position 1 (future leakage), corrupting main_preds[0] (the bonus
             * token) into garbage that then poisons the next draft iter. */
            is_prefill_ = true;
            for (int l = 0; l < config_.n_layers; l++)
                process_layer(l, verify_hidden.data(), verify_M, verify_tokens.data());
            is_prefill_ = false;

            fprintf(stderr, "[DSpark] Verify complete, checking accept/reject...\n");
            /* e) CAUSAL MASK: handled inside process_mla (is_prefill_=true
             *    but we set it for the block).  The causal mask ensures
             *    token i can only attend to tokens <= i within the block. */

            /* f) ACCEPT/REJECT: compare main model argmax with draft tokens.
             * Get the main model's predictions for each position. */
            std::vector<int> main_preds(verify_M);
            kernel_cache_->set_phase(AiebuKernelCache::LM_HEAD_PHASE);
            for (int m = 0; m < verify_M; m++) {
                std::vector<float> vm_logits(config_.vocab_size);
                apply_final_norm_and_lm_head(verify_hidden.data() + (size_t)m * N_HC * hd, 1, vm_logits.data());
                /* NaN/inf scan of the verify logits (diagnostics). */
                int n_nan = 0, n_inf = 0;
                float lg_max = -INFINITY, lg_min = INFINITY;
                for (int v = 0; v < config_.vocab_size; v++) {
                    float l = vm_logits[v];
                    if (__builtin_isnan(l)) { n_nan++; continue; }
                    if (__builtin_isinf(l)) { n_inf++; continue; }
                    if (l > lg_max) lg_max = l;
                    if (l < lg_min) lg_min = l;
                }
                /* Argmax: the main model's prediction for position m */
                int argmax = 0;
                float max_logit = vm_logits[0];
                for (int v = 1; v < config_.vocab_size; v++)
                    if (vm_logits[v] > max_logit) { max_logit = vm_logits[v]; argmax = v; }
                main_preds[m] = argmax;
                fprintf(stderr, "[SD-verify] pos %d: argmax=%d maxlogit=%.3f lg_range=[%.3f,%.3f] nan=%d inf=%d\n",
                        m, argmax, max_logit, lg_min, lg_max, n_nan, n_inf); fflush(stderr);
            }

            /* Accept the longest valid prefix.
             * main_preds[0] is the prediction after the last accepted token —
             * this is the "bonus" token.  main_preds[1..] are predictions after
             * each draft token.  We accept draft token i if main_preds[i] == draft.tokens[i]. */
            int n_accepted = 0;
            for (int i = 0; i < scheduled_len; i++) {
                if (main_preds[i] == draft.tokens[i]) {
                    n_accepted++;
                } else {
                    break;  /* first mismatch stops acceptance */
                }
            }

            /* The bonus token is main_preds[n_accepted] (the main model's
             * prediction after the last accepted draft token). */
            int bonus_token = main_preds[n_accepted];

            fprintf(stderr, "[SD-accept] n_accepted=%d/%d bonus=%d (draft were:",
                     n_accepted, scheduled_len, bonus_token);
            for (int i = 0; i < scheduled_len; i++)
                fprintf(stderr, " %d%s", draft.tokens[i],
                        (i < n_accepted) ? "(acc)" : "");
            fprintf(stderr, ")\n"); fflush(stderr);

            sd_accepted_tokens_ += n_accepted;
            sd_verify_passes_++;

            /* Fill the DSpark sliding-window KV for the ACCEPTED positions
             * (verify indices 0..n_accepted = the re-verified last token + the
             * accepted drafts) from the main_hidden captured during THIS verify
             * (target_hc_, [verify_M, hd] per layer).  Without this the window
             * has gaps at the accepted-draft positions (forward_draft writes only
             * the anchor), so future dense attention would miss their KV.
             * base_pos = seq_pos_ (still the verify start here; advanced below).
             * The next draft conditions on the last accepted token = verify
             * index n_accepted. */
            {
                int acc_count = n_accepted + 1;   /* positions S..S+n_accepted */
                bool have = !target_hc_[0].empty() && !target_hc_[1].empty() &&
                            !target_hc_[2].empty() &&
                            (int)target_hc_[0].size() >= acc_count * hd &&
                            (int)target_hc_[1].size() >= acc_count * hd &&
                            (int)target_hc_[2].size() >= acc_count * hd;
                if (have) {
                    std::vector<bf16_t, AlignedAllocator<bf16_t>> mh((size_t)acc_count * 3 * hd);
                    for (int m = 0; m < acc_count; m++) {
                        memcpy(mh.data() + (size_t)m * 3 * hd,
                               target_hc_[0].data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
                        memcpy(mh.data() + (size_t)m * 3 * hd + hd,
                               target_hc_[1].data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
                        memcpy(mh.data() + (size_t)m * 3 * hd + 2 * hd,
                               target_hc_[2].data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
                    }
                    auto mx = compute_draft_main_x(acc_count, mh.data());
                    fill_draft_window(acc_count, mx.data(), seq_pos_);
                }
            }
            target_anchor_idx_ = n_accepted;   /* last accepted = next draft's anchor source */

            /* g) KV CACHE: update for all accepted tokens.
             * We already processed verify_M tokens through the pipeline,
             * so the KV cache has been updated for positions seq_pos_..seq_pos_+verify_M-1.
             * We need to rollback the rejected tokens. */
            int tokens_to_keep = 1 + n_accepted;  /* last accepted + accepted drafts */
            seq_pos_ += tokens_to_keep;

            /* Output accepted tokens */
            for (int i = 0; i < n_accepted; i++) {
                if (cb) cb(draft.tokens[i], "", u);
                tokens_generated++;
                if (tokens_generated >= N) break;
                if (draft.tokens[i] == eos_token_id()) goto done;
            }

            /* Output the bonus token */
            if (tokens_generated < N) {
                if (cb) cb(bonus_token, "", u);
                fprintf(stderr, "[SD-emit] bonus token %d -> cb (tokens_generated=%d)\n",
                        bonus_token, tokens_generated + 1); fflush(stderr);
                tokens_generated++;
                tid = bonus_token;
                if (bonus_token == eos_token_id()) goto done;
            }

            /* Rollback KV cache for rejected tokens */
            int rejected = verify_M - tokens_to_keep;
            if (rejected > 0) {
                /* Zero out the KV cache entries for rejected positions */
                for (int l = 0; l < config_.n_layers; l++) {
                    auto& kvc = kv_cache_[l];
                    int rollback_start = seq_pos_;
                    for (int s = rollback_start; s < rollback_start + rejected; s++) {
                        if (s < config_.max_seq) {
                            memset(kvc.kv_latent.data() + s * MLA_KV_LORA, 0,
                                   MLA_KV_LORA * sizeof(bf16_t));
                            memset(kvc.k_pe.data() + s * MLA_N_HEADS * MLA_ROPE_DIM, 0,
                                   MLA_N_HEADS * MLA_ROPE_DIM * sizeof(bf16_t));
                        }
                    }
                }
                /* Restore the V4 compressor state to the pre-verify snapshot so
                 * the rejected suffix's comp rows + streaming-state mutations are
                 * undone (matches the KV rollback above). */
                cmp_state_ = cmp_snap;
            }

            /* Update hidden state for next iteration: keep the FULL 4-stream HC
             * residual of the last accepted token (so the next iteration's
             * output_hc_head collapse has all N_HC streams to work on — the old
             * `hd`-stride assign kept only stream-0, an over-read). */
            hidden.assign(verify_hidden.begin() + (size_t)n_accepted * N_HC * hd,
                          verify_hidden.begin() + (size_t)(n_accepted + 1) * N_HC * hd);
            M = 1;

            if (seq_pos_ >= config_.max_seq - gamma - 2) seq_pos_ = 0;
        }

    done:;
        auto t2 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
        double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double tok_per_sec = (tokens_generated > 1) ? (double)(tokens_generated - 1) / (decode_ms / 1000.0) : 0.0;
        double accept_rate = sd_total_draft_tokens_ > 0
                             ? (double)sd_accepted_tokens_ / (double)sd_total_draft_tokens_ : 0.0;

        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        long peak_rss_kb = ru.ru_maxrss;

        fprintf(stderr, "\n");
        fprintf(stderr, "  Prefill Time: %.1f ms (%d tokens)\n", prefill_ms, prefill_M);
        fprintf(stderr, "  Decode Tokens/sec: %.2f\n", tok_per_sec);
        fprintf(stderr, "  Peak RAM (RSS): %.1f GB\n", (double)peak_rss_kb / 1048576.0);
        fprintf(stderr, "  DSpark Accept Rate: %.3f (%ld/%ld draft tokens)\n",
                accept_rate, sd_accepted_tokens_, sd_total_draft_tokens_);
        fprintf(stderr, "  DSpark Verify Passes: %ld\n", sd_verify_passes_);
        if (pager_) {
            fprintf(stderr, "  Expert Pager: gets=%ld hits=%ld misses=%ld hit_rate=%.3f prefetched=%ld draft_prefetch=%ld cache=%zuMB\n",
                    pager_->gets(), pager_->hits(), pager_->misses(),
                    pager_->hit_rate(), pager_->prefetched(), pager_->draft_prefetch(),
                    pager_->cache_size() >> 20);
        }
        auto cs = kernel_cache_->stats();
        fprintf(stderr, "  NPU Contexts: creates=%d evictions=%d hits=%d active=%zu\n",
                cs.creates, cs.evictions, cs.hits, kernel_cache_->active_contexts());

    } catch (const xrt::run::aie_error& e) {
        fprintf(stderr, "\n[FATAL] XRT/AIE error: %s\n", e.what()); std::exit(1);
    } catch (const xrt::run::command_error& e) {
        fprintf(stderr, "\n[FATAL] XRT command error: %s\n", e.what()); std::exit(1);
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[FATAL] %s\n", e.what()); std::exit(1);
    }
}

/* ── Hunyuan-3.0 greedy autoregressive generation (ARCH_HY3, --no-sd) ───────
 * Host-first E2E: prefill the prompt through the trunk (blk.0..n_layers-2 —
 * the MTP block blk.80 is NOT a trunk layer) → output_norm (model_.final_norm)
 * → shared lm_head (host_lm_head_) → greedy argmax; decode one token/step the
 * same way.  Reuses process_gqa/process_ffn_hy3 (via process_layer's HY3 branch)
 * on a plain [M,hd] hidden (no HC streams).  seq_pos_ tracks absolute position:
 * prefill fills 0..M-1 per layer (layer-major, full causal); decode continues at
 * M, M+1, …  No draft/MTP (MTP drafting + SD acceptance are a later phase).
 * Greedy (temperature/top_p ignored for argmax). */
void FSTEngine::generate_hy3(const std::vector<int>& token_ids, int mt, float /*temperature*/,
                             float /*top_p*/, void(*cb)(int,const char*,void*), void* u) {
    const int N    = mt > 0 ? mt : 3;
    const int hd   = config_.hidden_dim;
    const int V    = config_.vocab_size;
    const int ndec = config_.n_layers - 1;           /* trunk = blk.0..ndec-1 (excludes MTP) */
    const int CHUNK = 16;
    const float eps = 1e-5f;

    auto t0 = std::chrono::steady_clock::now();
    auto t1 = t0;
    try {
        /* 1. prefill: embed prompt → [M, hd] plain, run trunk blk.0..ndec-1.
         *    Layer-major: each layer sees all M positions with full causal
         *    (kv[l] accumulates rows 0..M-1; seq_pos_ reset per layer). */
        int M = (int)token_ids.size();
        std::vector<bf16_t, AlignedAllocator<bf16_t>> hidden = forward_embeddings(token_ids);
        is_prefill_ = true;
        for (int l = 0; l < ndec; l++) {
            seq_pos_ = 0;
            for (int off = 0; off < M; ) {
                int m = std::min(CHUNK, M - off);
                process_layer(l, hidden.data() + (size_t)off * hd, m, token_ids.data() + off);
                seq_pos_ += m;
                off += m;
            }
        }
        seq_pos_ = M;                                /* next decode token is at position M */

        /* output_norm + shared lm_head → greedy argmax (host matvec). */
        auto argmax_head = [&](const bf16_t* h, float* logits) -> int {
            std::vector<bf16_t, AlignedAllocator<bf16_t>> hn(hd);
            rms_cpu(hn.data(), h, model_.final_norm, hd, 1, eps);
            int amx = 0; float mx = -1e30f;
            for (int v = 0; v < V; v++) {
                const bf16_t* row = host_lm_head_.data() + (size_t)v * hd;
                float s = 0.0f;
                for (int d = 0; d < hd; d++) s += bf16f(row[d]) * bf16f(hn[d]);
                logits[v] = s;
                if (s > mx) { mx = s; amx = v; }
            }
            return amx;
        };
        std::vector<float> logits(V);
        int tid = argmax_head(hidden.data() + (size_t)(M - 1) * hd, logits.data());
        fprintf(stderr, "[hy3-gen] prefill-last=%d (M=%d)\n", tid, M); fflush(stderr);
        if (cb) cb(tid, "", u);

        t1 = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int tokens_generated = 1;

        /* 2. decode loop: embed prev token → [1,hd] → trunk → output_norm+lm_head → argmax. */
        for (int tok = 1; tok < N; tok++) {
            std::vector<bf16_t, AlignedAllocator<bf16_t>> dec(hd, f2bf(0.0f));
            if (tid >= 0 && tid < V)
                std::memcpy(dec.data(), host_embedding_table_.data() + (size_t)tid * hd,
                            (size_t)hd * sizeof(bf16_t));
            for (int l = 0; l < ndec; l++)
                process_layer(l, dec.data(), 1, &tid);
            tid = argmax_head(dec.data(), logits.data());
            fprintf(stderr, "[hy3-gen] decode=%d (pos=%d)\n", tid, seq_pos_); fflush(stderr);
            if (cb) cb(tid, "", u);
            tokens_generated++;
            seq_pos_ += 1;
            if (seq_pos_ >= config_.max_seq - 1) break;   /* HY3 eos handling deferred */
        }

        auto t2 = std::chrono::steady_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double tps = (tokens_generated > 1) ? (double)(tokens_generated - 1) / (decode_ms / 1000.0) : 0.0;

        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        fprintf(stderr, "\n  HY3 Prefill: %.1f ms (%d prompt tokens, %d trunk layers)\n",
                prefill_ms, M, ndec);
        fprintf(stderr, "  HY3 Decode Tokens/sec: %.2f\n", tps);
        fprintf(stderr, "  Peak RAM (RSS): %.1f GB\n", (double)ru.ru_maxrss / 1048576.0);
        if (pager_)
            fprintf(stderr, "  Expert Pager: gets=%ld hits=%ld misses=%ld hit_rate=%.3f "
                    "wait=%ldms/%ld (max=%ldus) prefetched=%ld cache=%zuMB\n",
                    pager_->gets(), pager_->hits(), pager_->misses(), pager_->hit_rate(),
                    pager_->wait_us()/1000, pager_->wait_count(), pager_->max_wait_us(),
                    pager_->prefetched(), pager_->cache_size() >> 20);
        auto cs = kernel_cache_->stats();
        fprintf(stderr, "  NPU Contexts: creates=%d evictions=%d hits=%d active=%zu\n",
                cs.creates, cs.evictions, cs.hits, kernel_cache_->active_contexts());

    } catch (const xrt::run::aie_error& e) {
        fprintf(stderr, "\n[FATAL] XRT/AIE error: %s\n", e.what()); std::exit(1);
    } catch (const xrt::run::command_error& e) {
        fprintf(stderr, "\n[FATAL] XRT command error: %s\n", e.what()); std::exit(1);
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[FATAL] %s\n", e.what()); std::exit(1);
    }
}

void FSTEngine::generate(const std::vector<int>& token_ids, int mt, float temperature, float top_p,
                         void(*cb)(int,const char*,void*), void* u) {
    if (config_.arch == ARCH_HY3) {                 /* HY3 has its own host-first path */
        generate_hy3(token_ids, mt, temperature, top_p, cb, u);
        return;
    }
    int N = mt > 0 ? mt : 3;

    std::vector<bf16_t, AlignedAllocator<bf16_t>> hidden;
    int M = (int)token_ids.size();
    int prefill_M = M;

    auto t0 = std::chrono::steady_clock::now();
    try {
        hidden = forward_embeddings(token_ids);   /* [M, hd] */
        /* Expand to the 4-stream HC residual [M, N_HC, hd] (each stream starts
         * as a copy of the plain embedding — ds4.c hc_from_plain_embedding). */
        {
            const int hd = config_.hidden_dim;
            std::vector<bf16_t, AlignedAllocator<bf16_t>> hc((size_t)M * N_HC * hd);
            for (int m = 0; m < M; m++)
                for (int s = 0; s < N_HC; s++)
                    memcpy(hc.data() + ((size_t)m * N_HC + s) * hd,
                           hidden.data() + (size_t)m * hd, (size_t)hd * sizeof(bf16_t));
            hidden = std::move(hc);
        }
        {float mx=-1e30f,mn=1e30f; int big=0; for(size_t i=0;i<(size_t)M*N_HC*config_.hidden_dim;i++){float v=bf16f(hidden[i]); if(std::isnan(v))continue; if(v>mx&&v<1e30f)mx=v; if(v<mn&&v>-1e30f)mn=v; if(std::fabs(v)>=1e29f)big++;} fprintf(stderr,"[emb] after HC expand: %zu vals, mn=%.4f mx=%.4f big(>=1e29)=%d\n",(size_t)M*N_HC*config_.hidden_dim,mn,mx,big); fflush(stderr);}

        is_prefill_ = true;
        prefill_tiled(hidden.data(), M, token_ids.data());

        int tid = npu_sample_token(hidden.data() + (size_t)(M - 1) * N_HC * config_.hidden_dim, 1, temperature, top_p);
        fprintf(stderr, "[gen-tok] prefill-last=%d\n", tid); fflush(stderr);
        if (cb) cb(tid, "", u);

        auto t1 = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        seq_pos_ += M;

        int tokens_generated = 1;
        for (int tok = 1; tok < N; tok++) {
            M = 1;
            const int hd = config_.hidden_dim;
            std::vector<bf16_t, AlignedAllocator<bf16_t>> dec_hidden((size_t)N_HC * hd, f2bf(0.0f));
            if (tid >= 0 && tid < config_.vocab_size)
                for (int s = 0; s < N_HC; s++)
                    memcpy(dec_hidden.data() + (size_t)s * hd,
                           host_embedding_table_.data() + (size_t)tid * hd,
                           (size_t)hd * sizeof(bf16_t));

            bool step_time = std::getenv("FST_STEP_TIME");
            double step_t0 = step_time ? now() : 0.0;
            for (int l = 0; l < config_.n_layers; l++)
                process_layer(l, dec_hidden.data(), 1, &tid);
            if (step_time)
                fprintf(stderr, "[step-time] %d-layer forward=%.3f ms\n",
                        config_.n_layers, (now() - step_t0) * 1000.0);

            tid = npu_sample_token(dec_hidden.data(), 1, temperature, top_p);
            fprintf(stderr, "[gen-tok] decode=%d\n", tid); fflush(stderr);
            if (cb) cb(tid, "", u);
            if (tid == eos_token_id()) break;
            tokens_generated++;
            seq_pos_ += 1;
            if (seq_pos_ >= config_.max_seq - 1) seq_pos_ = 0;
        }

        auto t2 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
        double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double tok_per_sec = (tokens_generated > 1) ? (double)(tokens_generated - 1) / (decode_ms / 1000.0) : 0.0;

        /* Peak RSS (resident set) — the hard 64 GB RAM budget metric. */
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        long peak_rss_kb = ru.ru_maxrss;

        fprintf(stderr, "\n");
        fprintf(stderr, "  Prefill Time: %.1f ms (%d tokens)\n", prefill_ms, prefill_M);
        fprintf(stderr, "  Decode Tokens/sec: %.2f\n", tok_per_sec);
        fprintf(stderr, "  Peak RAM (RSS): %.1f GB\n", (double)peak_rss_kb / 1048576.0);
        if (pager_) {
            fprintf(stderr, "  Expert Pager: gets=%ld hits=%ld misses=%ld hit_rate=%.3f prefetched=%ld cache=%zuMB/%zuMB\n",
                    pager_->gets(), pager_->hits(), pager_->misses(),
                    pager_->hit_rate(), pager_->prefetched(),
                    pager_->cache_size() >> 20, (size_t)8192);
        }
        auto cs = kernel_cache_->stats();
        fprintf(stderr, "  NPU Contexts: creates=%d evictions=%d hits=%d active=%zu\n",
                cs.creates, cs.evictions, cs.hits, kernel_cache_->active_contexts());

    } catch (const xrt::run::aie_error& e) {
        fprintf(stderr, "\n[FATAL] XRT/AIE error: %s\n", e.what()); std::exit(1);
    } catch (const xrt::run::command_error& e) {
        fprintf(stderr, "\n[FATAL] XRT command error: %s\n", e.what()); std::exit(1);
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[FATAL] %s\n", e.what()); std::exit(1);
    }
}
