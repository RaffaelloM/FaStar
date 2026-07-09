// Build an NPU shim-DMA control sequence (insts.bin) via the FastFlowLM
// npu_sequence API (npu_dma_memcpy_nd + npu_dma_wait + cmds2seq).  This is the
// hand-built AIEBU control-seq path for kernels whose insts.bin aiecc cannot
// generate (raw shim-DMA / fused kernels).
//
// Validation: mode "rmsnorm" reproduces fst_ew_rmsnorm_insts.bin byte-for-byte
// (diff against the shipped file proves the API's BD encoding matches the
// IRON-compiled xclbins).  mode "fused" builds the 3-BD one-shot seq for
// fst_fused_ffn_direct (uint8 W MM2S ch0, bf16 A MM2S ch1, bf16 C S2MM ch0).
#include <climits>
#include "npu_utils/npu_instr_utils.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace std;

static void rmsnorm(npu_sequence& s){
    // [8,4096] rmsnorm: in(MM2S ch0,bd0,arg0), weight(MM2S ch1,bd1,arg1),
    // out(S2MM ch0,bd2,arg2).  Each buffer 8*4096=32768 bf16; size[0]=8 -> repeat=7.
    const npu_tiles shim = get_tile(0,0);
    const vector<uint32_t> off{0,0,0,0}, size{8,1,1,32768}, str{1,1,1,1};
    s.npu_dma_memcpy_nd(2,0,MM2S,shim,bd_0,it_channel_0,off,size,str);
    s.npu_dma_memcpy_nd(2,1,MM2S,shim,bd_1,it_channel_1,off,size,str);
    s.npu_dma_memcpy_nd(2,2,S2MM,shim,bd_2,it_channel_0,off,size,str);
    s.npu_dma_wait(shim,S2MM,it_channel_0);
}

static void fused(npu_sequence& s){
    // fst_fused_ffn_direct probe: W 2176 i8 (MM2S ch0,bd0,arg0),
    // A 1024 bf16 (MM2S ch1,bd1,arg1), C 1024 bf16 (S2MM ch0,bd2,arg2).  One-shot.
    const npu_tiles shim = get_tile(0,0);
    const vector<uint32_t> off{0,0,0,0}, str{1,1,1,1};
    s.npu_dma_memcpy_nd(1,0,MM2S,shim,bd_0,it_channel_0,off,{1,1,1,2176},str);  // uint8
    s.npu_dma_memcpy_nd(2,1,MM2S,shim,bd_1,it_channel_1,off,{1,1,1,1024},str);   // bf16
    s.npu_dma_memcpy_nd(2,2,S2MM,shim,bd_2,it_channel_0,off,{1,1,1,1024},str);  // bf16
    s.npu_dma_wait(shim,S2MM,it_channel_0);
}

int main(int argc, char** argv){
    if(argc<3){ fprintf(stderr,"usage: %s <rmsnorm|fused> <out_insts.bin>\n",argv[0]); return 1; }
    string mode = argv[1];
    npu_sequence s(device_npu2);
    if(mode=="rmsnorm") rmsnorm(s);
    else if(mode=="fused") fused(s);
    else { fprintf(stderr,"unknown mode %s\n",mode.c_str()); return 2; }
    s.cmds2seq();
    s.write_out_sequence(argv[2]);
    fprintf(stderr,"[gen] %s -> %s\n",mode.c_str(),argv[2]);
    return 0;
}