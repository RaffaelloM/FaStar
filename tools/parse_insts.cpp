// tools/parse_insts.cpp — raw decoder for FaStar _insts.bin (blob_instr_transaction).
//
// FastFlowLM's npu_sequence::seq2cmds does not decode FaStar's op words, so this
// is a standalone raw parser using the bit layouts from the npu_cmd_* headers.
// Blob = 4-word header + ops. Op opcodes (op_headers): WRITE=0, BLOCKWRITE=1,
// BLOCKSET=2, MASKWRITE=3(=issue_token,7 words), ..., TCT/WAIT=0x80(4 words),
// DDR_PATCH=0x81(12 words). Op line counts: BLOCKWRITE=12, DDR_PATCH=12,
// WAIT=4, ISSUE_TOKEN(MASKWRITE)=7, WRITE=6.
//
// Build:
//   g++ -std=c++17 -O2 -o tools/parse_insts tools/parse_insts.cpp
// Usage:
//   tools/parse_insts fst_mla_wqb_insts.bin [...]

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

static uint32_t rd(const std::vector<uint32_t>& v, size_t i) { return i < v.size() ? v[i] : 0; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s insts.bin [...]\n", argv[0]); return 1; }
    for (int a = 1; a < argc; ++a) {
        std::ifstream f(argv[a], std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[a]); continue; }
        size_t sz = (size_t)f.tellg();
        if (sz % 4) { std::fprintf(stderr, "%s: not word-aligned (%zu bytes)\n", argv[a], sz); continue; }
        std::vector<uint32_t> w(sz / 4);
        f.seekg(0); f.read(reinterpret_cast<char*>(w.data()), sz);
        std::printf("=== %s  (%zu bytes, %zu words; header cmds=%u lines=%u) ===\n",
                    argv[a], sz, w.size(), rd(w, 2), rd(w, 3) / 4);
        size_t i = 4;
        int idx = 0;
        while (i < w.size()) {
            uint32_t op = w[i];
            std::printf("[%2d] @%-4zu op=0x%02x ", idx++, i, op);
            if (op == 1) { // BLOCKWRITE (dma_block_cmd), 12 words
                uint32_t loc = rd(w, i + 2);
                uint32_t col = (loc >> 25) & 0x7f, row = (loc >> 20) & 0x1f, bdid = (loc >> 5) & 0xf;
                uint32_t len = rd(w, i + 4), off = rd(w, i + 5);
                uint32_t d0 = rd(w, i + 7), d1 = rd(w, i + 8), d2 = rd(w, i + 9), it = rd(w, i + 10), nxt = rd(w, i + 11);
                uint32_t d0s = (d0 >> 20) & 0x3ff, d0st = (d0 & 0xfffff) + 1;
                uint32_t d1s = (d1 >> 20) & 0x3ff, d1st = (d1 & 0xfffff) + 1;
                uint32_t d2st = (d2 & 0xfffff) + 1;
                uint32_t its = ((it >> 20) & 0x3ff) + 1, itst = (it & 0xfffff) + 1;
                uint32_t next_bd = (nxt >> 27) & 0xf, valid = (nxt >> 25) & 0x1;
                uint32_t lacq = nxt & 0xf, lacqv = (nxt >> 5) & 0xef, lrel = (nxt >> 13) & 0xf;
                bool linear = (d0 == 0);
                std::printf("DMA_BD tile(r=%u,c=%u) bd=%u len=%uB off=%u %s d0(s=%u,st=%u) d1(s=%u,st=%u) d2(st=%u) iter(s=%u,st=%u) next_bd=%u valid=%u lock_acq(id=%u,v=%u) lock_rel(id=%u)\n",
                            row, col, bdid, len, off, linear ? "LINEAR" : "2D",
                            d0s, d0st, d1s, d1st, d2st, its, itst, next_bd, valid, lacq, lacqv, lrel);
                i += 12;
            } else if (op == 0x81) { // DDR_PATCH, 12 words
                uint32_t loc = rd(w, i + 6);
                uint32_t col = (loc >> 25) & 0x7f, row = (loc >> 20) & 0x1f, bdid = ((loc - 0x04) >> 5) & 0x1f;
                uint32_t arg_idx = rd(w, i + 8), arg_off = rd(w, i + 10);
                std::printf("DDR_PATCH tile(r=%u,c=%u) bd=%u arg_idx=%u arg_off=%uB\n",
                            row, col, bdid, arg_idx, arg_off);
                i += 12;
            } else if (op == 0x80) { // WAIT/TCT, 4 words
                uint32_t x = rd(w, i + 2), ch = (rd(w, i + 3) >> 24) & 0xff;
                uint32_t wr = (x >> 8) & 0xff, wc = (x >> 16) & 0xff, dir = x & 0x1;
                std::printf("WAIT tile(r=%u,c=%u) ch=%u dir=%s\n", wr, wc, ch, dir ? "MM2S" : "S2MM");
                i += 4;
            } else if (op == 3) { // MASKWRITE = issue_token, 7 words
                uint32_t loc = rd(w, i + 2);
                uint32_t col = (loc >> 25) & 0x7f, row = (loc >> 20) & 0x1f;
                uint32_t dir = (loc & 0x10) ? 1 : 0, chid = (loc >> 3) & 0x1;
                uint32_t pkt = rd(w, i + 4) >> 8;
                std::printf("ISSUE_TOK tile(r=%u,c=%u) ch=%u dir=%s pkt=%u\n",
                            row, col, chid, dir ? "MM2S" : "S2MM", pkt);
                i += 7;
            } else if (op == 0) { // WRITE, 6 words
                uint32_t loc = rd(w, i + 2), val = rd(w, i + 4);
                uint32_t col = (loc >> 25) & 0x7f, row = (loc >> 20) & 0x1f, reg = loc & 0xfffff;
                std::printf("WRITE tile(r=%u,c=%u) reg=0x%x val=0x%x\n", row, col, reg, val);
                i += 6;
            } else {
                std::printf("UNKNOWN op=0x%02x (stopping)\n", op);
                break;
            }
        }
        std::printf("--- end: parsed %d ops, consumed %zu/%zu words ---\n", idx, i, w.size());
    }
    return 0;
}