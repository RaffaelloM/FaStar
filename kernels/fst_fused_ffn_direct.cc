// FUSED MXFP4-dequant + bf16 GEMM, INTERLEAVED (no static bbuf).
// Canonical K-major pattern: dequant each B tile (kt,nt) into a STACK btmp[32]
// (stack memory is coherent with load_v; the static .bss bbuf is NOT — scalar
// and even store_v writes to static .bss are read wrong by load_v, while stack
// atile/tmp work).  btmp+atile = 128 B, fits the 1 KB core stack.
#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

static const bfloat16 FP4_LUT[16] = {
    (bfloat16)0.0f,(bfloat16)0.5f,(bfloat16)1.0f,(bfloat16)1.5f,
    (bfloat16)2.0f,(bfloat16)3.0f,(bfloat16)4.0f,(bfloat16)6.0f,
    (bfloat16)0.0f,(bfloat16)-0.5f,(bfloat16)-1.0f,(bfloat16)-1.5f,
    (bfloat16)-2.0f,(bfloat16)-3.0f,(bfloat16)-4.0f,(bfloat16)-6.0f
};
static inline bfloat16 e8m0_to_bf16(uint8_t s){
    if (s==0||s==255) return (bfloat16)0.0f;
    union{uint16_t u;bfloat16 f;}c; c.u=(uint16_t)s<<7; return c.f;
}
constexpr int M=16,K=64,N=64,BLK=32,BLK_BYTES=17,VR=4,VS=8,VT=4;
constexpr int K_DIV_S=K/VS,N_DIV_T=N/VT,M_DIV_R=M/VR;
using MMUL=aie::mmul<VR,VS,VT,bfloat16,bfloat16,accauto>;

static inline bfloat16 dequant_elem(const uint8_t *__restrict W,int k,int n){
    int blk_id=k/BLK, blk_off=k%BLK;
    const uint8_t* blk = W + (size_t)n*(K/BLK)*BLK_BYTES + (size_t)blk_id*BLK_BYTES;
    bfloat16 scale = e8m0_to_bf16(blk[0]);
    uint8_t byte = blk[1 + blk_off/2];
    bfloat16 v = (blk_off%2==0) ? FP4_LUT[byte & 0x0F] : FP4_LUT[(byte>>4)&0x0F];
    float sf=(float)scale, vf=(float)v;
    return (bfloat16)(sf*vf);
}

extern "C" void fst_fused_ffn_direct(uint8_t *__restrict W_q4, bfloat16 *__restrict A, bfloat16 *__restrict C_out)
{
    event0();
    alignas(32) bfloat16 btmp[32];       // B tile [8,4] row-major (stack)
    alignas(32) bfloat16 atile[MMUL::size_A];
    alignas(16) bfloat16 ctile[MMUL::size_C];
    for (int mt=0;mt<M_DIV_R;mt++)
        for (int nt=0;nt<N_DIV_T;nt++){
            MMUL C_acc; int first=1;
            for (int kt=0;kt<K_DIV_S;kt++){
                for (int k=0;k<VS;k++) for (int n=0;n<VT;n++)
                    btmp[k*VT+n]=dequant_elem(W_q4, kt*VS+k, nt*VT+n);
                for (int mr=0;mr<VR;mr++) for (int kr=0;kr<VS;kr++)
                    atile[mr*VS+kr]=A[((mt*VR+mr)*K)+kt*VS+kr];
                auto Av=aie::load_v<MMUL::size_A>((const bfloat16*)atile);
                auto Bv=aie::load_v<MMUL::size_B>((const bfloat16*)btmp);
                if(first){C_acc.mul(Av,Bv);first=0;}else{C_acc.mac(Av,Bv);}
            }
            aie::store_v((bfloat16*)ctile,C_acc.template to_vector<bfloat16>());
            for (int mr=0;mr<VR;mr++) for (int nc=0;nc<VT;nc++)
                C_out[(mt*VR+mr)*N+nt*VT+nc]=ctile[mr*VT+nc];
        }
    event1();
}