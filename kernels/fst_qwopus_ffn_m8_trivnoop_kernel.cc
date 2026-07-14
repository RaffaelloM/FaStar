// fst_qwopus_ffn_m8_trivnoop_kernel.cc — M=8 TRIVIAL noop diagnostic.
// Isolates the M=8 dispatch+DMA+output floor from the 128-load_v/packet noop.
// Trivial noop: 1 scalar read of pkt[0] (force input DMA), 1 output write, no
// load_v loop, no 8-h strided reads.  Built via FST_FFN_FN=ffn_matvec_noop_stream.
// If this is ~2 ms (like the M=1 noop), the 48 ms floor is the 128 load_v/packet
// (8 strided h-vector reads); if ~48 ms, it's dispatch/output RMW.
#include <aie_api/aie.hpp>
#include <stdint.h>
constexpr int M_n = 16, HDR = 32;
extern "C" void ffn_matvec_noop_stream(const uint8_t *__restrict pkt, float *__restrict o){
    const int row_base=*reinterpret_cast<const int*>(pkt);
    volatile uint8_t sink = pkt[HDR];          // 1 scalar read (force input DMA)
    o[row_base] = (float)sink;                  // 1 output write (force output DMA)
    (void)sink;
}