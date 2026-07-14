// fst_gdn_8kslab_canary_kernel.cc — stack-S persistence diagnostic.
// Same shim/fifo contract + same function name as fst_gdn_8kslab_kernel.cc, but
// the recurrence body is replaced by a CANARY: each of the K=8 steps does
// S[i,c] += 1.0 on the 8 KB stack S.  Reading back snew tells us how the stack
// S behaves across the on-tile K-loop:
//   snew == S0 + 8.0   ⇒ stack S PERSISTS across the K-loop + passB writes OK
//                       (bug is in passA/delta/par-reading, not S lifetime)
//   snew == S0 + 1.0   ⇒ only the LAST step's write landed (S does NOT persist
//                       between recur_core calls — each sees a fresh/stale S)
//   snew == S0         ⇒ passB never writes the stack S at all
// The par/yd/s0/snew DMA is exercised exactly as the real kernel (same fifos,
// same depth, same TAPs) so any placement/DMA-pressure effect is reproduced.
#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int HV = 128, COLS = 32, K = 8, V = 8, NCV = COLS / 8;
constexpr int PAR_SZ = 2 * HV + COLS + 3, YD_SZ = 2 * COLS;

static inline void vcopy_bf16(const bfloat16 *__restrict in, bfloat16 *__restrict out, int nvec){
    for (int n = 0; n < nvec; ++n) aie::store_v(out + n * V, aie::load_v<V>(in + n * V));
}

// CANARY recurrence: S[i,c] += 1.0 per step.  par/yd are read/written just to
// exercise the DMA fifos (values ignored on read; a fixed pattern on yd write).
static inline void recur_core(bfloat16 *__restrict S, const bfloat16 *__restrict par,
                              bfloat16 *__restrict yd){
    const aie::vector<bfloat16, V> one_bf = aie::broadcast<bfloat16, V>((bfloat16)1.0f);
    // touch par so the DMA read isn't dead-coded (load + store into yd)
    for (int n = 0; n < (PAR_SZ + V - 1) / V; ++n)
        aie::store_v(yd + n * V, aie::load_v<V>(par + n * V));   // par -> yd (ignore semantics)
    for (int i = 0; i < HV; ++i){
        bfloat16 *srow = S + (size_t)i * COLS;
        for (int n = 0; n < NCV; ++n){
            aie::vector<bfloat16, V> sv = aie::load_v<V>(srow + n * V);
            // S += 1.0 in bf16 (broadcast-add)
            aie::vector<bfloat16, V> nv = aie::add(sv, one_bf);
            aie::store_v(srow + n * V, nv);
        }
    }
}

extern "C" void gdn_8kslab_vhead(const bfloat16 *__restrict s0, const bfloat16 *__restrict par,
                                 bfloat16 *__restrict snew, bfloat16 *__restrict yd){
    bfloat16 S[HV * COLS] __attribute__((aligned(32)));
    for (int i = 0; i < HV; ++i) vcopy_bf16(s0 + (size_t)i * COLS, S + (size_t)i * COLS, NCV);
    for (int t = 0; t < K; ++t) recur_core(S, par + (size_t)t * PAR_SZ, yd + (size_t)t * YD_SZ);
    for (int i = 0; i < HV; ++i) vcopy_bf16(S + (size_t)i * COLS, snew + (size_t)i * COLS, NCV);
}