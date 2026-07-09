/// \file fst_npu_executor.cpp
/// \brief Low-latency NPU dequant executor using extracted IRON topology.
///
/// Builds a DMA instruction sequence that matches the AIE graph compiled
/// inside dequant_mxfp4.xclbin, using the EXACT topology extracted from
/// FastFlowLM's captured dequant_seq.bin:
///
///   - 8 Shim columns (row=0, col=0..7)
///   - Compute tiles: rows 2,3,4,5 × cols 0..7 (RTP write 0x8A00 = 1)
///   - MM2S ch0: BD 2, 2D (32×40×23 = 29440 words = 117760 bytes)
///   - MM2S ch1: BD 10, same 2D pattern
///   - S2MM ch0: BD 1 (ch0 output) + BD 9 (ch1 output), linear 188416 words
///   - DDR patches: arg_idx=0 → bo_in, arg_idx=1 → bo_out
///   - Wait: S2MM ch0
///
/// Compile:
///   g++ -std=c++17 -O2 -I. -I Source/FastFlowLM-main/src/include \
///       -o fst_npu_executor fst_npu_executor.cpp \
///       -lxrt_coreutil -lxrt_core -luuid -laiebu -lpthread

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <climits>
#include <fcntl.h>
#include <unistd.h>
#include <thread>

#include "npu_utils/npu_instr_utils.hpp"
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include "aiebu/aiebu.h"

// ---------------------------------------------------------------------------
// Topology constants (extracted from dequant_seq.bin via analyze_dequant_seq.py)
// ---------------------------------------------------------------------------

static constexpr int NUM_COLUMNS = 8;

// MM2S BD: 2D pattern 32×40×23 = 29440 AIE words = 117760 bytes per BD
static constexpr uint32_t MM2S_BUF_WORDS  = 29440;   // AIE 32-bit words
static constexpr uint32_t MM2S_BUF_BYTES  = MM2S_BUF_WORDS * 4;  // 117760
static constexpr uint32_t MM2S_DIM0       = 32;      // dim0_size (AIE words)
static constexpr uint32_t MM2S_DIM1       = 40;      // dim1_size
static constexpr uint32_t MM2S_DIM2       = 23;      // dim2_size (inferred)
// API sizes (before elem_size=1 /4 conversion): dim0 = 32*4 = 128 bytes
static constexpr uint32_t MM2S_DIM0_API   = MM2S_DIM0 * 4;  // 128
// API strides (before /4): dim1 = 32*4=128, dim2 = 1280*4=5120
static constexpr uint32_t MM2S_ST_DIM1    = MM2S_DIM0 * 4;      // 128
static constexpr uint32_t MM2S_ST_DIM2    = MM2S_DIM0 * MM2S_DIM1 * 4; // 5120

// S2MM BD: linear 188416 AIE words = 753664 bytes per BD
static constexpr uint32_t S2MM_BUF_WORDS  = 188416;
static constexpr uint32_t S2MM_BUF_BYTES  = S2MM_BUF_WORDS * 4;  // 753664
// API size (before elem_size=2 /2 conversion): 188416*2 = 376832 BF16 elements
static constexpr uint32_t S2MM_SIZE_API   = S2MM_BUF_WORDS * 2;  // 376832

// Expert data: [32, 69, 23, 4, 2560] per layer, 1 expert = 69*23*4*2560
static constexpr size_t EXPERT_BYTES = 69 * 23 * 4 * 2560;  // 16,250,880

// Expert 0 in model.q4nx (from safetensors header)
// model.layers.0.ffn_gate_up_down_exps.weight offset = 1982086272
static constexpr uint64_t EXPERT0_TENSOR_OFFSET = 1982086272ULL;

// BD IDs
static constexpr npu_bd_id BD_MM2S_CH0 = bd_2;
static constexpr npu_bd_id BD_MM2S_CH1 = bd_10;
static constexpr npu_bd_id BD_S2MM_CH0 = bd_1;
static constexpr npu_bd_id BD_S2MM_CH1 = bd_9;

// Compute tile rows that receive RTP writes
static constexpr int COMPUTE_ROWS[] = {2, 3, 4, 5};
static constexpr uint32_t RTP_ADDR = 0x8A00;
static constexpr uint32_t RTP_VALUE = 1;

// ---------------------------------------------------------------------------
// Build the NPU sequence for a single expert dequant
// ---------------------------------------------------------------------------
// Build sequence using EXACT topology extracted from dequant_seq.bin
// 138 individual MM2S BDs + 23 S2MM BDs (repeat=3) + 32 RTP writes
// ---------------------------------------------------------------------------

struct InBd { int col; int bd_id; uint32_t offset; };
struct OutBd { int col; int bd_id; uint32_t offset; int repeat; };

void build_sequence(npu_sequence& seq) {
    // RTP writes: enable compute tiles (rows 2-5, cols 0-7)
    for (int r : {2, 3, 4, 5})
        for (int c = 0; c < 8; c++)
            seq.rtp_write(get_tile(r, c), 0x8A00, 1);

    // 138 individual MM2S BDs with exact offsets from captured sequence
    static const InBd in_bds[] = {
        {0, 0, 0},        {0, 2, 5120},
        {1, 0, 471040},   {1, 2, 476160},
        {2, 0, 942080},   {2, 2, 947200},
        {3, 0, 1413120},  {3, 2, 1418240},
        {4, 0, 1884160},  {4, 2, 1889280},
        {5, 0, 2355200},  {5, 2, 2360320},
        {6, 0, 2826240},  {6, 2, 2831360},
        {7, 0, 3297280},  {7, 2, 3302400},
        {0, 8, 3768320},  {0, 10, 3773440},
        {1, 8, 4239360},  {1, 10, 4244480},
        {2, 8, 4710400},  {2, 10, 4715520},
        {3, 8, 5181440},  {3, 10, 5186560},
        {4, 8, 5652480},  {4, 10, 5657600},
        {5, 8, 6123520},  {5, 10, 6128640},
        {6, 8, 6594560},  {6, 10, 6599680},
        {7, 8, 7065600},  {7, 10, 7070720},
        {0, 0, 7536640},  {0, 2, 7541760},
        {1, 0, 8007680},  {1, 2, 8012800},
        {2, 0, 8478720},  {2, 2, 8483840},
        {3, 0, 8949760},  {3, 2, 8954880},
        {4, 0, 9420800},  {4, 2, 9425920},
        {5, 0, 9891840},  {5, 2, 9896960},
        {6, 0, 10362880}, {6, 2, 10368000},
        {0, 0, 235520},   {0, 2, 240640},
        {1, 0, 706560},   {1, 2, 711680},
        {2, 0, 1177600},  {2, 2, 1182720},
        {3, 0, 1648640},  {3, 2, 1653760},
        {4, 0, 2119680},  {4, 2, 2124800},
        {5, 0, 2590720},  {5, 2, 2595840},
        {6, 0, 3061760},  {6, 2, 3066880},
        {7, 0, 3532800},  {7, 2, 3537920},
        {0, 8, 4003840},  {0, 10, 4008960},
        {1, 8, 4474880},  {1, 10, 4480000},
        {2, 8, 4945920},  {2, 10, 4951040},
        {3, 8, 5416960},  {3, 10, 5422080},
        {4, 8, 5888000},  {4, 10, 5893120},
        {5, 8, 6359040},  {5, 10, 6364160},
        {6, 8, 6830080},  {6, 10, 6835200},
        {7, 8, 7301120},  {7, 10, 7306240},
        {0, 0, 7772160},  {0, 2, 7777280},
        {1, 0, 8243200},  {1, 2, 8248320},
        {2, 0, 8714240},  {2, 2, 8719360},
        {3, 0, 9185280},  {3, 2, 9190400},
        {4, 0, 9656320},  {4, 2, 9661440},
        {5, 0, 10127360}, {5, 2, 10132480},
        {6, 0, 10598400}, {6, 2, 10603520},
        {0, 0, 10833920}, {0, 2, 10839040},
        {1, 0, 11069440}, {1, 2, 11074560},
        {2, 0, 11304960}, {2, 2, 11310080},
        {3, 0, 11540480}, {3, 2, 11545600},
        {4, 0, 11776000}, {4, 2, 11781120},
        {5, 0, 12011520}, {5, 2, 12016640},
        {6, 0, 12247040}, {6, 2, 12252160},
        {7, 0, 12482560}, {7, 2, 12487680},
        {0, 8, 12718080}, {0, 10, 12723200},
        {1, 8, 12953600}, {1, 10, 12958720},
        {2, 8, 13189120}, {2, 10, 13194240},
        {3, 8, 13424640}, {3, 10, 13429760},
        {4, 8, 13660160}, {4, 10, 13665280},
        {5, 8, 13895680}, {5, 10, 13900800},
        {6, 8, 14131200}, {6, 10, 14136320},
        {7, 8, 14366720}, {7, 10, 14371840},
        {0, 0, 14602240}, {0, 2, 14607360},
        {1, 0, 14837760}, {1, 2, 14842880},
        {2, 0, 15073280}, {2, 2, 15078400},
        {3, 0, 15308800}, {3, 2, 15313920},
        {4, 0, 15544320}, {4, 2, 15549440},
        {5, 0, 15779840}, {5, 2, 15784960},
        {6, 0, 16015360}, {6, 2, 16020480},
    };

    // 23 S2MM BDs with repeat=3 (1 expert = 96/32 of full sequence)
    static const OutBd out_bds[] = {
        {0, 1, 0, 3},        {1, 1, 753664, 3},
        {2, 1, 1507328, 3},  {3, 1, 2260992, 3},
        {4, 1, 3014656, 3},  {5, 1, 3768320, 3},
        {6, 1, 4521984, 3},  {7, 1, 5275648, 3},
        {0, 9, 6029312, 3},  {1, 9, 6782976, 3},
        {2, 9, 7536640, 3},  {3, 9, 8290304, 3},
        {4, 9, 9043968, 3},  {5, 9, 9797632, 3},
        {6, 9, 10551296, 3}, {7, 9, 11304960, 3},
        {0, 1, 12058624, 3}, {1, 1, 12812288, 3},
        {2, 1, 13565952, 3}, {3, 1, 14319616, 3},
        {4, 1, 15073280, 3}, {5, 1, 15826944, 3},
        {6, 1, 16580608, 3},
    };

    // Submit all 138 MM2S BDs (individual, iter=1, 2D pattern 32×40×23)
    for (const auto& b : in_bds) {
        npu_tiles shim = get_tile(0, b.col);
        npu_it_channel ch = (b.bd_id >= 8) ? it_channel_1 : it_channel_0;
        seq.npu_dma_memcpy_nd(
            1, 0, MM2S, shim, (npu_bd_id)b.bd_id, ch,
            {0, 0, 0, b.offset},
            {1, 23, 40, 128},
            {117760, 5120, 128, 1},
            -1, 0, false, normal_cache, 15, true);
    }

    // Submit 23 S2MM BDs (with repeat=3, linear 188416 words)
    for (const auto& b : out_bds) {
        npu_tiles shim = get_tile(0, b.col);
        seq.npu_dma_memcpy_nd(
            2, 1, S2MM, shim, (npu_bd_id)b.bd_id, it_channel_0,
            {0, 0, 0, b.offset / 2},
            {(uint32_t)b.repeat, 1, 1, 376832},
            {376832, 0, 0, 1},
            -1, 0, true, normal_cache, 15, true);
    }

    // Wait for S2MM ch0 on all columns
    for (int c = 0; c < 8; c++)
        seq.npu_dma_wait(get_tile(0, c), S2MM, it_channel_0);
}

// ---------------------------------------------------------------------------
// XRT initialization
// ---------------------------------------------------------------------------

struct XrtCtx {
    xrt::device device;
    std::unique_ptr<xrt::hw_context> hw_ctx;
    std::string kernel_name;

    XrtCtx(const std::string& xclbin_path)
        : device(0)
        , kernel_name()
    {
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        hw_ctx = std::make_unique<xrt::hw_context>(device, xclbin.get_uuid());

        auto kernels = xclbin.get_kernels();
        auto it = std::find_if(kernels.begin(), kernels.end(),
            [](xrt::xclbin::kernel& k) {
                return k.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (it == kernels.end())
            throw std::runtime_error("MLIR_AIE kernel not found in " + xclbin_path);
        kernel_name = it->get_name();

        std::cout << "[xrt] xclbin: " << xclbin_path << "\n";
        std::cout << "[xrt] kernel: " << kernel_name << "\n";
    }
};

// ---------------------------------------------------------------------------
// Generate ELF from sequence via aiebu
// ---------------------------------------------------------------------------

xrt::ext::kernel make_kernel(XrtCtx& ctx, npu_sequence& seq) {
    auto [instr_ptr, instr_size] = seq.dump();
    if (!instr_ptr || instr_size == 0)
        throw std::runtime_error("Empty instruction sequence");

    std::cout << "[aiebu] Instruction words: " << instr_size
              << " (" << instr_size * 4 << " bytes)\n";

    char* elf_buf = nullptr;
    uint32_t elf_size = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        (char*)instr_ptr, instr_size * sizeof(uint32_t),
        NULL, 0, (void**)&elf_buf, NULL, 0, "", "", NULL, 0);

    if (elf_size == 0)
        throw std::runtime_error("aiebu_assembler_get_elf failed");

    std::cout << "[aiebu] ELF size: " << elf_size << " bytes\n";

    xrt::elf elf(elf_buf, elf_size);
    xrt::module mod(elf);
    xrt::ext::kernel kernel(*ctx.hw_ctx, mod, ctx.kernel_name);
    free(elf_buf);
    return kernel;
}

// ---------------------------------------------------------------------------
// Load expert 0 data from model.q4nx via pread
// ---------------------------------------------------------------------------

size_t load_expert0(xrt::ext::bo& bo, const std::string& q4nx_path) {
    int fd = open(q4nx_path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("Cannot open " + q4nx_path);

    // Read safetensors header length (first 8 bytes, uint64 LE)
    uint64_t header_len = 0;
    if (pread(fd, &header_len, 8, 0) != 8) {
        close(fd);
        throw std::runtime_error("Failed to read header length");
    }
    std::cout << "[q4nx] Header JSON length: " << header_len << "\n";

    // Expert 0 absolute file offset = 8 + header_len + tensor_data_offset
    uint64_t file_offset = 8 + header_len + EXPERT0_TENSOR_OFFSET;
    std::cout << "[q4nx] Expert 0 file offset: " << file_offset
              << " (0x" << std::hex << file_offset << std::dec << ")\n";

    auto* ptr = bo.map<uint8_t*>();
    ssize_t n = pread(fd, ptr, EXPERT_BYTES, file_offset);
    close(fd);

    if (n != (ssize_t)EXPERT_BYTES)
        throw std::runtime_error("pread got " + std::to_string(n) +
                                 " bytes, expected " + std::to_string(EXPERT_BYTES));

    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::cout << "[q4nx] Loaded " << n << " bytes of expert 0 data\n";
    return n;
}

// ---------------------------------------------------------------------------
// BF16 helpers
// ---------------------------------------------------------------------------

static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string xclbin_path = "Source/FastFlowLM-main/src/xclbins/GPT-OSS-20B-NPU2/dequant.xclbin";
    std::string q4nx_path   = "Source/GPT OSS 20B Fastflow/model.q4nx";

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--xclbin") && i+1 < argc)
            xclbin_path = argv[++i];
        else if (!std::strcmp(argv[i], "--q4nx") && i+1 < argc)
            q4nx_path = argv[++i];
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            std::cout << "Usage: " << argv[0] << " [--xclbin PATH] [--q4nx PATH]\n";
            return 0;
        }
    }

    std::cout << "============================================================\n";
    std::cout << "FaStar NPU Dequant Executor (IRON topology-matched)\n";
    std::cout << "============================================================\n";

    // --- Step 1: Plan the expert distribution ---
    std::cout << "\n[1] Expert dequant: 138 individual MM2S BDs + 23 S2MM BDs (repeat=3)\n";

    // --- Step 2: XRT init ---
    std::cout << "\n[2] XRT initialization\n";
    XrtCtx ctx(xclbin_path);

    // --- Step 3: Build sequence ---
    std::cout << "\n[3] Build NPU sequence\n";
    npu_sequence seq(device_npu2);
    build_sequence(seq);

    auto [instr_ptr, instr_size] = seq.dump();
    std::cout << "   Sequence: " << instr_size << " words"
              << " (" << instr_size * 4 << " bytes)\n";

    // Write sequence for debugging
    seq.write_out_sequence("/tmp/fst_executor_seq.bin");
    std::cout << "   Written to /tmp/fst_executor_seq.bin\n";

    // --- Step 4: Generate ELF + kernel ---
    std::cout << "\n[4] Generate ELF via aiebu\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto kernel = make_kernel(ctx, seq);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elf_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "   ELF generation: " << elf_us << " us\n";

    // Print group IDs
    std::cout << "   group_ids:";
    for (int i = 0; i < 8; i++) {
        try { std::cout << " [" << i << "]=" << kernel.group_id(i); }
        catch (...) { break; }
    }
    std::cout << "\n";

    // --- Step 5: Allocate BOs ---
    std::cout << "\n[5] Allocate BOs\n";
    size_t bo_in_sz  = EXPERT_BYTES;                        // 16,250,880 bytes
    // Output: 138 BDs × 753,664 = 104,005,632 bytes → round up to 128 MB
    size_t bo_out_sz = 128ULL * 1024 * 1024;

    // Use xrt::ext::bo (like fst_npu_dynamic_seq.cpp) and pass 5 BOs
    // The ext::kernel has group_ids 3-7, needs all 5 BO arguments
    xrt::ext::bo bo_in (ctx.device, bo_in_sz);
    xrt::ext::bo bo_out(ctx.device, bo_out_sz);
    xrt::ext::bo bo2(ctx.device, 1024 * 1024);  // dummy BOs for args 5-7
    xrt::ext::bo bo3(ctx.device, 1024 * 1024);
    xrt::ext::bo bo4(ctx.device, 1024 * 1024);
    std::cout << "   bo_in:  " << bo_in_sz  << " bytes (" << bo_in_sz / (1024.0*1024.0) << " MB)\n";
    std::cout << "   bo_out: " << bo_out_sz << " bytes (" << bo_out_sz / (1024.0*1024.0) << " MB)\n";

    // --- Step 6: Load expert 0 data ---
    std::cout << "\n[6] Load expert 0 data\n";
    load_expert0(bo_in, q4nx_path);

    // Clear output buffer
    auto* out_ptr = bo_out.map<uint8_t*>();
    std::memset(out_ptr, 0, bo_out_sz);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // --- Step 7: Execute and measure ---
    std::cout << "\n[7] Execute NPU dequant\n" << std::flush;

    ert_cmd_state state = ERT_CMD_STATE_MAX;
    auto start = std::chrono::high_resolution_clock::now();

    try {
        auto run = kernel(3, 0, 0,
            static_cast<xrt::bo&>(bo_in),
            static_cast<xrt::bo&>(bo_out),
            static_cast<xrt::bo&>(bo2),
            static_cast<xrt::bo&>(bo3),
            static_cast<xrt::bo&>(bo4));
        // Use wait with 5s timeout - throws on error/abnormal state
        state = run.wait(5000);
    } catch (const std::exception& e) {
        // wait() throws on ERROR/ABORT state - but kernel may have partially executed
        std::cerr << "   wait() exception: " << e.what() << "\n";
        state = ERT_CMD_STATE_ERROR;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "NPU Execution Time: " << ms << " ms (State: " << (int)state << ")\n";
    std::cout << "   State: " << state << " (1=NEW 2=QUEUED 3=RUNNING 4=COMPLETED 5=ERROR 6=ABORT 7=SUBMITTED 8=TIMEOUT)\n";

    // --- Step 8: Validate output ---
    std::cout << "\n[8] Validate output\n";
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    uint16_t* bf16_out = reinterpret_cast<uint16_t*>(out_ptr);
    std::cout << "   First 4 BF16 output values:\n";
    for (int i = 0; i < 4; i++) {
        uint16_t raw = bf16_out[i];
        float fval = bf16_to_f32(raw);
        std::cout << "     [" << i << "] raw=0x" << std::hex << raw << std::dec
                  << "  float=" << fval << "\n";
    }

    // Check if output is non-zero
    bool nonzero = false;
    for (size_t i = 0; i < bo_out_sz; i++) {
        if (out_ptr[i] != 0) { nonzero = true; break; }
    }
    std::cout << "   Output " << (nonzero ? "NON-ZERO (kernel executed)" : "ALL ZEROS (kernel did not run)") << "\n";

    // --- Summary ---
    std::cout << "\n============================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "Topology:  8 shim columns, rows 2-5 compute tiles\n";
    std::cout << "MM2S:      BD 2 (ch0) + BD 10 (ch1), 2D 32×40×23 = 29440 words/BD\n";
    std::cout << "S2MM:      BD 1 (ch0) + BD 9 (ch1), linear 188416 words/BD\n";
    std::cout << "DDR patch: arg_idx=0→bo_in, arg_idx=1→bo_out\n";
    std::cout << "Total BDs: 138 input (individual) + 23 output (repeat=3) = 69 output chunks\n";
    std::cout << "Latency:   " << ms << " ms (" << us << " us)\n";
    std::cout << "State:     " << (int)state;
    if (state == ERT_CMD_STATE_COMPLETED)
        std::cout << " (COMPLETED)";
    else if (state == ERT_CMD_STATE_ERROR)
        std::cout << " (ERROR)";
    else if (state == ERT_CMD_STATE_TIMEOUT)
        std::cout << " (TIMEOUT)";
    std::cout << "\n";
    std::cout << "============================================================\n";

    // Use _exit to bypass BO/kernel destructors that can hang on NPU
    std::cout << std::flush;
    _exit((state == ERT_CMD_STATE_COMPLETED) ? 0 : 1);
}
