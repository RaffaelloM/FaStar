// Standalone insts.bin decoder (FastFlowLM npu_sequence::seq2cmds has a library bug:
// it clear_cmds() — which clears npu_seq — BEFORE parsing, so it reads empty memory.
// This tool reads the file into a local buffer and walks it with the correct per-op
// sizes, printing each command's raw words so the BD/lock/DDR-patch pattern of a
// working single-tile kernel can be reverse-engineered for hand-building control seqs.)
#include <climits>
#include "npu_utils/instr_utils/npu_cmd.hpp"
#include <fstream>
#include <iostream>
#include <vector>
static const char* nm(uint32_t o){
    switch(o){
      case 0: return "WRITE"; case 1: return "BLOCKWRITE(DMA BD)"; case 2: return "BLOCKSET";
      case 3: return "ISSUE_TOKEN"; case 4: return "MASKPOLL"; case 5: return "NOOP";
      case 0x80: return "TCT/WAIT"; case 0x81: return "DDR_PATCH";
      default: return "UNKNOWN";
    }
}
static int opsize(uint32_t o){
    switch(o){
      case 1: return 12; case 0x81: return 12; case 0: return 6; case 0x80: return 4; case 3: return 7;
      default: return 1;
    }
}
int main(int argc, char** argv){
    if(argc<2){ std::cerr<<"usage: "<<argv[0]<<" <insts.bin>\n"; return 1; }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    if(raw.size()%4){ std::cerr<<"bad size\n"; return 2; }
    std::vector<uint32_t> s(raw.size()/4);
    memcpy(s.data(), raw.data(), raw.size());
    std::cout<<"words="<<s.size()<<" hdr_cmds="<<s[2]<<" hdr_lines="<<(s[3]/4)
             <<" npu2 gen="<<((s[0]>>16)&0xff)<<" rows="<<((s[0]>>24)&0xff)
             <<" cols="<<(s[1]&0xff)<<"\n";
    int i=4, c=0;
    while(i < (int)s.size()){
        uint32_t op=s[i]; int n=opsize(op);
        if(op==0 || op==1 || op==0x80 || op==0x81 || op==3 || op==4){
            std::cout<<"["<<i<<"] CMD#"<<c<<" OP="<<op<<" ("<<nm(op)<<") size="<<n<<"\n";
            for(int k=1;k<n && i+k<(int)s.size();k++)
                std::cout<<"    w"<<k<<" 0x"<<std::hex<<s[i+k]<<std::dec<<"\n";
            c++;
        } else {
            std::cout<<"["<<i<<"] OP="<<op<<" (unknown/inline) — stopping\n";
            break;
        }
        i+=n;
    }
    std::cout<<"parsed "<<c<<" cmds, ended at word "<<i<<" of "<<s.size()<<"\n";
    return 0;
}
