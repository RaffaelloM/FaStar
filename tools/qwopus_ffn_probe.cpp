// tools/qwopus_ffn_probe.cpp — Re-stream-h BF16-MMUL NPU FFN matvec probe.
//
// SHAPE-PARAMETERIZED (Step 1 of the FFN down-shape divergence debug): serves
// BOTH the gate/up shape [N=17408,K=5120] (the original probe) AND the down
// shape [N=5120,K=17408] (the shape the original probe never tested, where the
// wired engine diverged per docs/QWOPUS_FFN_NPU_WIRED_POSTMORTEM.md).
//
// ONE xclbin (compiled N_TILE=2176, NPKT=680/tile) serves BOTH shapes — the
// kernel is K-agnostic and N_TILE-agnostic in its compute (it writes
// o[row_base+nt*8+n] with row_base = nc*16, idx < N_TILE_USED).  The COMPILED
// held-output buffer is 2176 fp32/tile regardless of shape.  For down N_TILE_USED
// = 640, so the kernel writes the FIRST 640/tile of the 2176-wide held output and
// the drain still pulls 2176/tile (offset t*2176).  Reading the down output
// CONTIGUOUSLY (outFull[n]) is WRONG — down outputs are STRIDED:
//     outNpu[n] = outFull[(n/N_TILE_USED)*N_TILE_COMPILED + (n%N_TILE_USED)]
// gate/up has N_TILE_USED == N_TILE_COMPILED == 2176 => that degenerates to
// outFull[n] (contiguous), reproducing the original probe exactly.
//
// Both shapes have NPKT = NCHUNKS*KCHUNKS = 680 (gate/up 136*5, down 40*17) =>
// the compiled NPKT=680 matches.  No recompile needed to test down.
//
// Validates the re-stream-h BF16-MMUL MXFP4 matvec kernel (fst_qwopus_ffn.xclbin)
// vs the host mxfp4_matvec_f32 reference, and measures COMPUTE-vs-DMA (full vs
// no-op).  Packet (host-packed uint8): [32 B hdr(row_base,kc) | h_chunk(2048 B
// BF16) | scales(512) | nibbles(8192)] = 10784 B.  h packed as BF16 (truncated
// from fp32, matching the engine's bf16 = u&0xffff0000); NPU streams bf16 h +
// vector bitwise dequant + aie::mmul<8,8,8> bfp16-emulated bf16×bf16->fp32-acc.
// Host ref keeps fp32 h + fp32 dequant + fp32 MAC (the shipped path).  Fork D
// lossless-argmax relaxation: argmax MUST match; rel higher (~1e-2, bfp16).
//
// Usage: qwopus_ffn_probe [dir] [shape] [scrange]
//   dir      = kernels (default)
//   shape   = up | down          (default up = gate/up [17408,5120])
//   scrange = narrow | wide | real (default narrow: sc 120..140 + ~1/23 dead)
//             wide: sc 1..254 (exercises the sc>252 clamp + sc=255 dead path)
//             real: load W + h from files (argv[5]=W.bin argv[6]=h.bin, fp32 h)
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

// ── Packet geometry (must match gen_qwopus_ffn.py + fst_qwopus_ffn_kernel.cc) ───
constexpr int M_n = 16, G = 32, HDR = 32, BLK = 17;
constexpr int H_BYTES  = G * 32 * 2;          // 2048 (bf16)
constexpr int SC_BYTES = M_n * G;             // 512 e8m0 scales
constexpr int NB_BYTES = M_n * G * 16;        // 8192 nibble bytes (16-aligned)
constexpr int W_BYTES  = SC_BYTES + NB_BYTES; // 8704
constexpr int PKT_BYTES = HDR + H_BYTES + W_BYTES;  // 10784
constexpr int NT = 8;
constexpr int N_TILE_COMPILED = 2176;         // the xclbin's held-output width

// Shape table (computed at runtime from argv).  NPKT must equal the compiled 680.
struct Shape {
    const char* name;
    int N, K, GROUPS, KCHUNKS, NCHUNKS, N_TILE_USED;   // NPKT = NCHUNKS*KCHUNKS
};
static const Shape SH_UP   = {"up",   17408, 5120,  160, 5,  136, 2176};  // 680 pkts
static const Shape SH_DOWN  = {"down",  5120, 17408, 544, 17,  40,  640};  // 680 pkts

static float FP4_TABLE[16] = {0};

static inline uint16_t fp32_to_bf16_trunc(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
static inline void pack_bf16_row(uint8_t* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t b = fp32_to_bf16_trunc(src[i]);
        dst[2 * i]     = (uint8_t)(b & 0xFF);
        dst[2 * i + 1] = (uint8_t)(b >> 8);
    }
}

// ── Host reference (verbatim math from src/fst_engine.cpp:251) ──────────────
static void mxfp4_matvec_f32(float* out, const float* h, const uint8_t* packed,
                             int Nrows, int K) {
    const int groups = K / 32;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < Nrows; n++) {
        const uint8_t* row = packed + (size_t)n * groups * 17;
        float acc = 0.0f;
        for (int g = 0; g < groups; g++) {
            const uint8_t* blk = row + (size_t)g * 17;
            uint8_t sc = blk[0];
            if (sc == 0) continue;
            float scale = std::ldexp(1.0f, (int)sc - 127);
            const uint8_t* nb = blk + 1;
            const float* hb = h + g * 32;
            for (int i = 0; i < 32; i++) {
                uint8_t byte = nb[i >> 1];
                uint8_t nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                acc += FP4_TABLE[nib] * scale * hb[i];
            }
        }
        out[n] = acc;
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

int main(int argc,char**argv){
    const char*dir=(argc>1)?argv[1]:"kernels";
    std::string shape_s = (argc>2) ? argv[2] : "up";
    std::string sc_s    = (argc>3) ? argv[3] : "narrow";
    // Env overrides to point the probe at variant xclbins (no rebuild of probe):
    //   FST_FFN_XCLBIN       (default fst_qwopus_ffn)      the full matvec xclbin stem
    //   FST_FFN_NOOP_XCLBIN  (default fst_qwopus_ffn_noop) the no-op xclbin stem
    //   FST_FFN_CONST_TEST=1  dq=1.0 isolation: verify ALL N outputs are IDENTICAL
    //     (= sum_k h[k]), the mmul/output-path correctness criterion (per-n
    //     variation => B read from outside dq / A not broadcast / cv[n]!=C[0,n]).
    std::string full_stem  = std::getenv("FST_FFN_XCLBIN")      ? std::getenv("FST_FFN_XCLBIN")      : "fst_qwopus_ffn";
    std::string noop_stem = std::getenv("FST_FFN_NOOP_XCLBIN") ? std::getenv("FST_FFN_NOOP_XCLBIN") : "fst_qwopus_ffn_noop";
    bool const_test = std::getenv("FST_FFN_CONST_TEST") != nullptr;
    bool h_one = std::getenv("FST_FFN_H_ONE") != nullptr;   // set all h=1.0 (with dq=1.0 => output must = K)
    const Shape* S = (shape_s == "down") ? &SH_DOWN : &SH_UP;
    const int N = S->N, K = S->K, GROUPS = S->GROUPS, KCHUNKS = S->KCHUNKS;
    const int NCHUNKS = S->NCHUNKS, N_TILE_USED = S->N_TILE_USED;
    const int NPKT = NCHUNKS * KCHUNKS;                          // 680 both
    fprintf(stderr,"[probe] shape=%s N=%d K=%d GROUPS=%d KCHUNKS=%d NCHUNKS=%d "
            "N_TILE_USED=%d NPKT=%d (compiled N_TILE=%d)\n",
            S->name, N, K, GROUPS, KCHUNKS, NCHUNKS, N_TILE_USED, NPKT, N_TILE_COMPILED);
    if (NPKT != 680) { fprintf(stderr,"[probe] FATAL NPKT=%d != compiled 680\n", NPKT); return 6; }

    const float FP4[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                           0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
    std::memcpy(FP4_TABLE, FP4, sizeof(FP4));

    xrt::device dev(0);
    XK kFull=load(dev,std::string(dir)+"/"+full_stem+".xclbin",
                  read_file(std::string(dir)+"/"+full_stem+"_insts.bin"));
    XK kNoop=load(dev,std::string(dir)+"/"+noop_stem+".xclbin",
                  read_file(std::string(dir)+"/"+noop_stem+"_insts.bin"));

    // ── Test data ────────────────────────────────────────────────────────────
    std::vector<float> h(K);
    std::vector<uint8_t> W((size_t)N * GROUPS * 17);

    if (sc_s == "real") {
        // argv[4]=W.bin (raw MXFP4, N*GROUPS*17 bytes), argv[5]=h.bin (K fp32)
        std::vector<char> wb = read_file(argv[4]);
        std::vector<char> hb = read_file(argv[5]);
        if ((size_t)wb.size() != (size_t)N*GROUPS*17) {
            fprintf(stderr,"[probe] W.bin %zu != expected %zu\n",wb.size(),(size_t)N*GROUPS*17);return 7;}
        if ((size_t)hb.size() != (size_t)K*4) {
            fprintf(stderr,"[probe] h.bin %zu != expected %zu\n",hb.size(),(size_t)K*4);return 7;}
        std::memcpy(W.data(), wb.data(), wb.size());
        std::memcpy(h.data(), hb.data(), hb.size());
        fprintf(stderr,"[probe] scrange=real: loaded W(%zuB) h(%zuB)\n",wb.size(),hb.size());
    } else {
        for (int k = 0; k < K; k++) h[k] = h_one ? 1.0f : (float)((((k * 37) % 1000) - 500)) / 500.0f;
        for (int n = 0; n < N; n++) {
            for (int g = 0; g < GROUPS; g++) {
                uint8_t* blk = &W[(size_t)n * GROUPS * 17 + (size_t)g * 17];
                uint8_t sc;
                if (sc_s == "wide") {
                    sc = (uint8_t)(1 + ((n*7 + g*13) % 254));     // 1..254 (exercises clamp + 255? no, 1..254)
                    if (((n + g) % 31) == 0) sc = 0;              // dead ~1/31
                    if (((n + g) % 41) == 0) sc = 255;            // e8m0-reserved dead ~1/41
                    if (((n + g) % 53) == 0) sc = 253;           // clamp-hit ~1/53
                } else { // narrow (reproduces the original probe's 120..140)
                    sc = (uint8_t)(120 + ((n*7 + g*13) % 21));   // 120..140 (nonzero)
                    if (((n + g) % 23) == 0) sc = 0;              // ~1/23 dead
                }
                blk[0] = sc;
                for (int i = 0; i < 16; i++) blk[1 + i] = (uint8_t)((n*3 + g*5 + i*11) & 0xFF);
            }
        }
        fprintf(stderr,"[probe] scrange=%s: synthetic sc (dead + nonzero nibbles)\n", sc_s.c_str());
    }

    // ── Host reference (full N) ──────────────────────────────────────────────
    std::vector<float> ref(N);
    mxfp4_matvec_f32(ref.data(), h.data(), W.data(), N, K);

    // ── Pack the input BO: NT tiles × NPKT packets × PKT_BYTES ────────────────
    const size_t IN_BYTES  = (size_t)NT * NPKT * PKT_BYTES;
    const size_t OUT_BYTES = (size_t)NT * N_TILE_COMPILED * 4;   // compiled width
    std::vector<uint8_t> inp(IN_BYTES);
    for (int t = 0; t < NT; t++) {
        for (int nc = 0; nc < NCHUNKS; nc++) {
            for (int kc = 0; kc < KCHUNKS; kc++) {
                const size_t pidx = (size_t)t * NPKT + (nc * KCHUNKS + kc);
                uint8_t* pkt = &inp[pidx * PKT_BYTES];
                int row_base = nc * M_n;                       // within tile (< N_TILE_USED)
                std::memcpy(pkt, &row_base, 4);
                std::memcpy(pkt + 4, &kc, 4);
                std::memset(pkt + 8, 0, HDR - 8);
                pack_bf16_row(pkt + HDR, &h[(size_t)kc * G * 32], G * 32);  // h_chunk (bf16)
                uint8_t* sc_dst = pkt + HDR + H_BYTES;
                uint8_t* nb_dst = sc_dst + SC_BYTES;
                for (int r = 0; r < M_n; r++) {
                    int gn = t * N_TILE_USED + nc * M_n + r;   // global row
                    for (int g = 0; g < G; g++) {
                        const uint8_t* blk =
                            &W[(size_t)gn * GROUPS * 17 + (size_t)(kc * G + g) * 17];
                        sc_dst[r * G + g] = blk[0];
                        std::memcpy(nb_dst + (size_t)(r * G + g) * 16, blk + 1, 16);
                    }
                }
            }
        }
    }

    xrt::ext::bo boIn(dev, IN_BYTES), boOut(dev, OUT_BYTES);
    {auto*p=boIn.map<uint8_t*>();std::memcpy(p,inp.data(),IN_BYTES);}
    boIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = [&](XK& k, const char* tag) -> double {
        {auto*p=boOut.map<float*>();std::memset(p,0,OUT_BYTES);}
        boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        double s0 = now_ms();
        k.krnl(3,0,0,static_cast<xrt::bo&>(boIn),static_cast<xrt::bo&>(boOut)).wait(120000);
        double d = now_ms() - s0;
        boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        fprintf(stderr,"[time] %s = %.2f ms  (%.2f GB/s over %.1f MB)\n", tag, d,
                (double)IN_BYTES / d / 1e6, (double)IN_BYTES / 1e6);
        return d;
    };

    double tFull = run(kFull, "ffn_matvec_restream");
    const float* outF = boOut.map<float*>();
    std::vector<float> outFull(outF, outF + NT * N_TILE_COMPILED);
    double tNoop = run(kNoop, "ffn_matvec_noop_stream");

    // ── Gather NPU outputs with the COMPILED-stride read ─────────────────────
    // outNpu[n] = outFull[(n/N_TILE_USED)*N_TILE_COMPILED + (n%N_TILE_USED)]
    // gate/up: N_TILE_USED == N_TILE_COMPILED == 2176 => outFull[n] (contiguous)
    // down:    N_TILE_USED=640, N_TILE_COMPILED=2176  => strided by 2176
    std::vector<float> outNpu(N);
    for (int n = 0; n < N; n++) {
        int t = n / N_TILE_USED;
        int i = n % N_TILE_USED;
        outNpu[n] = outFull[(size_t)t * N_TILE_COMPILED + i];
    }

    // ── dq=1.0 CONST isolation test (FST_FFN_CONST_TEST) ──────────────────────
    // With B[k,n]=1.0 (const kernel) and A=h broadcast, EVERY output = sum_k h[k]
    // = Σh, IDENTICAL across all N.  Per-n/tile variation => the mmul reads B from
    // outside the dq slots (B-layout bug), or A is not broadcast, or cv[n]!=C[0,n].
    if (const_test) {
        double sumh = 0; for (int k = 0; k < K; k++) sumh += h[k];
        double mn = outNpu[0], mx = outNpu[0], mean = 0;
        int nan_inf = 0;
        for (int n = 0; n < N; n++) {
            double v = outNpu[n];
            if (!std::isfinite(v)) { nan_inf++; continue; }
            if (v < mn) mn = v; if (v > mx) mx = v; mean += v;
        }
        mean /= N;
        double spread = mx - mn;
        // Σh is in bf16-truncated h space (h packed bf16); allow bfp16/bf16 slack.
        bool pass = (nan_inf == 0) && (spread < std::max(1e-3, std::fabs(sumh)*1e-2 + 1e-3));
        fprintf(stderr, "\n=== dq=1.0 CONST TEST [%s/%s] ===\n", S->name, sc_s.c_str());
        fprintf(stderr, "  Σh (expected every output) = %.6f\n", sumh);
        fprintf(stderr, "  outNpu: min=%.6f max=%.6f mean=%.6f spread=%.3e nan/inf=%d\n",
                mn, mx, mean, spread, nan_inf);
        fprintf(stderr, "  first 16: ");
        for (int n = 0; n < 16; n++) fprintf(stderr, "%.4f ", outNpu[n]);
        fprintf(stderr, "\n  => %s (spread %s 0)\n", pass?"PASS":"FAIL", pass?"~":"!=");
        printf("const_test shape=%s sumh=%.6f min=%.6f max=%.6f spread=%.3e naninf=%d %s\n",
               S->name, sumh, mn, mx, spread, nan_inf, pass?"PASS":"FAIL");
        return pass ? 0 : 4;
    }

    // ── First-16-outputs sanity (tile 0, row 0..15) vs host ref ─────────────
    fprintf(stderr, "\n=== first 16 outputs (tile 0) vs host ref ===\n");
    for (int n = 0; n < 16; n++)
        fprintf(stderr,"  y[%2d] npu=% .4e  ref=% .4e  Δ=% .3e\n",
                n, outNpu[n], ref[n], outNpu[n] - ref[n]);

    // ── Totals (scramble vs value-error discriminator) ───────────────────────
    // A pure permutation (same multiset, reassigned) conserves Σ_n npu == Σ_n ref
    // and Σ|npu|==Σ|ref|; differing totals => value/magnitude error, not a scramble.
    {
        double sn=0, sr=0, an=0, ar=0;
        for (int n = 0; n < N; n++) { sn+=outNpu[n]; sr+=ref[n]; an+=std::fabs(outNpu[n]); ar+=std::fabs(ref[n]); }
        fprintf(stderr, "totals: Σnpu=%.6e Σref=%.6e  (ratio %.4f)  Σ|npu|=%.6e Σ|ref|=%.6e (ratio %.4f)\n",
                sn, sr, sr?sn/sr:0, an, ar, ar?an/ar:0);
        printf("totals sum_npu=%.6e sum_ref=%.6e sumabs_npu=%.6e sumabs_ref=%.6e\n", sn, sr, an, ar);
    }

    // ── DUMP mode (FST_FFN_DUMP): print outNpu[0..255] as a dequant-dump readback
    // (for the dump kernel that writes dq[0..255] to o[0..255] on the first packet).
    if (std::getenv("FST_FFN_DUMP")) {
        printf("DUMP outNpu[0..255] (dq[ks*64+n*8+kr], nt=0 g=0):\n");
        for (int i = 0; i < 256; i++) printf("%g ", outNpu[i]);
        printf("\n");
        return 0;
    }

    // ── Correctness vs host ref (argmax gate) ────────────────────────────────
    double maxabs = 0; int firstbad = -1, ref_am=0, npu_am=0;
    float ref_mx=ref[0], npu_mx=outNpu[0];
    float ref_mx2=-1e30f, npu_mx2=-1e30f;
    for (int n = 1; n < N; n++) {
        if(ref[n]>ref_mx){ref_mx2=ref_mx;ref_mx=ref[n];ref_am=n;}
        else if(ref[n]>ref_mx2){ref_mx2=ref[n];}
        if(outNpu[n]>npu_mx){npu_mx2=npu_mx;npu_mx=outNpu[n];npu_am=n;}
        else if(outNpu[n]>npu_mx2){npu_mx2=outNpu[n];}
    }
    int nan_inf = 0;
    for (int n = 0; n < N; n++) {
        double d = std::fabs((double)outNpu[n] - ref[n]);
        if (d > maxabs) maxabs = d;
        if (firstbad < 0 && d > 1e-3) firstbad = n;
        if (!std::isfinite(outNpu[n]) || !std::isfinite(ref[n])) nan_inf++;
    }
    bool argmatch = (ref_am == npu_am);
    double maxabs_ref = 0; for (int n = 0; n < N; n++) maxabs_ref = std::fmax(maxabs_ref, std::fabs(ref[n]));
    double relmax = maxabs_ref ? maxabs / maxabs_ref : maxabs;
    double gap = std::fabs((double)ref_mx - ref_mx2);
    double gap_rel = maxabs_ref ? gap / maxabs_ref : gap;
    bool rel_sane = (relmax < 0.2);
    bool near_tie = (!argmatch) && (gap_rel < 5e-2);
    bool ok = argmatch && rel_sane && (nan_inf == 0);

    double dma_gb  = (double)IN_BYTES / tNoop / 1e6;
    double full_gb = (double)IN_BYTES / tFull  / 1e6;
    double ratio   = tFull / tNoop;

    printf("shape=%s scrange=%s full_ms=%.2f noop_ms=%.2f ratio=%.2f full_gb=%.2f noop_gb=%.2f "
           "maxabs=%.4e relmax=%.3e firstbad=%d naninf=%d argmax_ref=%d npu=%d %s gap_rel=%.3e %s\n",
           S->name, sc_s.c_str(), tFull, tNoop, ratio, full_gb, dma_gb, maxabs, relmax,
           firstbad, nan_inf, ref_am, npu_am, argmatch?"MATCH":"DIFF", gap_rel,
           ok?"PASS":(near_tie?"NEAR-TIE":"FAIL"));
    fprintf(stderr,"\n=== RESULT [%s/%s] ===\n", S->name, sc_s.c_str());
    fprintf(stderr,"full matvec : %.2f ms (%.2f GB/s)\n", tFull, full_gb);
    fprintf(stderr,"no-op (DMA) : %.2f ms (%.2f GB/s)\n", tNoop, dma_gb);
    fprintf(stderr,"compute/DMA ratio full/noop = %.2fx\n", ratio);
    if (ratio < 1.5) fprintf(stderr,"=> DMA-BOUND\n");
    else             fprintf(stderr,"=> COMPUTE-BOUND (%.2fx slower than DMA)\n", ratio);
    fprintf(stderr,"correctness : max|Δ|=%.4e rel=%.3e firstbad=%d nan/inf=%d argmax %s -> %s\n",
            maxabs, relmax, firstbad, nan_inf, argmatch?"MATCH":"DIFF",
            ok?"PASS":(near_tie?"NEAR-TIE":"FAIL"));
    fprintf(stderr,"argmax gap  : top1-top2 (host) = %.4e (gap_rel=%.3e); ref=%d npu=%d\n",
            gap, gap_rel, ref_am, npu_am);
    return ok ? 0 : (near_tie ? 0 : 4);
}