// fst_gdn_8kstack_micro_kernel.cc — single-call 8 KB-stack INTERNAL-LOOP test.
// Structure = the full chunkwise kernel: ONE dispatch, ONE call to the C
// function, which loops the 48 v-heads INTERNALLY (no repeated external
// calls — the worker core calls k() once and the 8 KB stack frame is entered
// only once, so SP does not need to restore between v-heads).
//
//   in[384]  = 48 v-heads x [gdec=0.5, rowoff, pad x6]
//   out[384] = 48 v-heads x [partial, pad x7]
//   k() loops v=0..47: fill 8 KB stack S[32*128] (rows rowoff+1..+32), RMW K=8
//   (gdec), sum -> out[v*8+0].  Expected partial = 264.0 for all 48 (bit-exact).
#include <aie_api/aie.hpp>
#include <stdint.h>
constexpr int HV=128, QUART=32, VB=16, NVB=HV/VB, KST=8, NVH=48, PKT=8;

extern "C" void gdn_8k_vhead(const float *__restrict in, float *__restrict out) {
    for (int v = 0; v < NVH; ++v) {
        const float *iv = in + v * PKT;
        float *ov = out + v * PKT;
        const float gdec = iv[0];
        const int row_offset = (int)iv[1];
        bfloat16 S[QUART * HV];                 // 8 KB stack — one frame for all 48 (no re-call)
        for (int r = 0; r < QUART; ++r) {
            bfloat16 *row = S + r * HV;
            bfloat16 val_bf = (bfloat16)(float)(row_offset + r + 1);
            aie::vector<bfloat16, VB> bv = aie::broadcast<bfloat16, VB>(val_bf);
            for (int n = 0; n < NVB; ++n) aie::store_v(row + n * VB, bv);
        }
        {
            const bfloat16 gdec_bf = (bfloat16)gdec;
            aie::vector<bfloat16, VB> gv = aie::broadcast<bfloat16, VB>(gdec_bf);
            for (int k = 0; k < KST; ++k)
                for (int r = 0; r < QUART; ++r) {
                    bfloat16 *row = S + r * HV;
                    for (int n = 0; n < NVB; ++n) {
                        aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
                        aie::accum<accfloat, VB> acc = aie::mul(sv, gv);
                        aie::store_v(row + n * VB, acc.to_vector<bfloat16>());
                    }
                }
        }
        float sum = 0.0f;
        aie::vector<bfloat16, VB> ones = aie::broadcast<bfloat16, VB>((bfloat16)1.0f);
        for (int r = 0; r < QUART; ++r) {
            bfloat16 *row = S + r * HV;
            for (int n = 0; n < NVB; ++n) {
                aie::vector<bfloat16, VB> sv = aie::load_v<VB>(row + n * VB);
                aie::accum<accfloat, VB> acc = aie::mul(sv, ones);
                sum += aie::reduce_add(acc.to_vector<float>());
            }
        }
        ov[0] = sum;
    }
}