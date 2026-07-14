#include <aie_api/aie.hpp>
#include <stdint.h>
// Read bytes at key offsets (clamped to pkt_bytes) + a vector load at offset 0.
// Reports whether the uint8 packet is correctly DMA'd end-to-end and whether
// bytes beyond 16384 B (4096 words, the BD-length cap) arrive — i.e. whether
// IRON BD-chains >4096-word packets correctly.
extern "C" void diag_read(const uint8_t *__restrict p, float *__restrict out, int pkt_bytes){
    auto rd=[&](int off)->float{ return (off < pkt_bytes) ? (float)p[off] : -1.0f; };
    out[0]=rd(0);
    out[1]=rd(4096);
    out[2]=rd(8192);
    out[3]=rd(12000);
    out[4]=rd(20000);          // beyond 16384 B (4096 words) — tests BD chain
    const float *h=reinterpret_cast<const float*>(p);
    auto v=aie::load_v<8>(h);
    out[5]=v.get(0); out[6]=v.get(1); out[7]=v.get(7);
}
