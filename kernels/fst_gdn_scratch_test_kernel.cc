// fst_gdn_scratch_test_kernel.cc — Stage 1.1d: persistent held-scratchpad test.
//
// Tests the two properties the full chunkwise kernel needs from S storage that
// the STACK array (proven fast, pattern 3) CANNOT provide: PERSISTENCE across
// per-token C-fn re-calls.  The stack frame is freed when the C fn returns, so
// a stack S does NOT carry state across re-calls (the toy fill re-init each
// call masked this).  The full kernel's recurrence needs S to persist across
// the K=8 per-token calls.
//
// This test uses a NAMED on-tile memory region (aie.iron.Buffer) as S, passed
// to the C fn as a POINTER.  The C fn RMWs it with load_v/store_v (same as the
// stack array — plain tile memory, NOT the Python-level Buffer RMW that was
// 600x slow in the M=1 fused attempt).  S persists across re-calls because the
// named region is fixed for the whole dispatch.
//
//   S (persistent buffer, 2048 fp32 = 8 KB), initial S[0] = 0.
//   Per call: S[0] = S[0] * gdec + in[0]   (recurrence; gdec = in[1])
//              out[0] = S[0]               (drain the running accumulator)
//   After N re-calls with in[0] = 1, gdec = 1.0: S[0] = N  (persistence).
//   If S did NOT persist (re-zeroed each call), out[0] = 1 for every call.
//
//   The probe checks the LAST call's out[0] == N  (persistence) and measures
//   latency (fast = the C-kernel-direct-access Buffer is NOT the 600x slow path;
//   the 600x was Python-level Buffer RMW, not a C pointer to tile memory).

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int SLEN = 2048;          // 8 KB fp32 persistent scratchpad
constexpr int VB   = 8;

// RMW the persistent S: S[0] = S[0]*gdec + addend ; out[0] = S[0].
// Also RMW the full 8 KB (S[1..2047] *= gdec each call) so the whole buffer is
// live (defeats DCE; the speed test is real, not just S[0]).
extern "C" void gdn_scratch_step(float *__restrict S,
                                  const float *__restrict in,
                                  float *__restrict out) {
    const float addend = in[0];
    const float gdec   = in[1];
    // RMW the whole 8 KB buffer (vector loop) + accumulate S[0].
    auto gv = aie::broadcast<float, VB>(gdec);
    for (int n = 0; n < SLEN / VB; ++n) {
        auto sv = aie::load_v<VB>(S + n * VB);
        aie::store_v(S + n * VB, aie::mul(sv, gv).template to_vector<float>());
    }
    // S[0] += addend  (the recurrence accumulator; persists across calls)
    S[0] += addend;
    out[0] = S[0];
}

// Init the persistent S to zero before the re-call loop.
extern "C" void gdn_scratch_zero(float *__restrict S) {
    auto z = aie::zeros<float, VB>();
    for (int n = 0; n < SLEN / VB; ++n) aie::store_v(S + n * VB, z);
}