// fst_xclbin_inspect.cpp - Inspect kernel metadata in mm.xclbin and expert.xclbin
#include <iostream>
#include <vector>
#include <string>
#include <xrt/xrt_device.h>
#include <xrt/experimental/xrt_xclbin.h>

int main(int argc, char** argv) {
    std::vector<std::string> paths = {
        "Source/GPT OSS 20B Fastflow/mm.xclbin",
        "Source/GPT OSS 20B Fastflow/expert.xclbin",
        "Source/FastFlowLM-main/src/xclbins/GPT-OSS-20B-NPU2/mm.xclbin",
        "Source/FastFlowLM-main/src/xclbins/GPT-OSS-20B-NPU2/expert.xclbin",
    };

    xrt::device device(0);

    for (const auto& path : paths) {
        std::cout << "========================================\n";
        std::cout << "File: " << path << "\n";

        try {
            xrt::xclbin xclbin(path);

            auto kernels = xclbin.get_kernels();
            std::cout << "  UUID: " << xclbin.get_uuid().to_string() << "\n";
            std::cout << "  Kernels: " << kernels.size() << "\n";

            for (auto& k : kernels) {
                std::cout << "    Kernel: " << k.get_name() << "\n";

                auto args = k.get_args();
                std::cout << "      Args: " << args.size() << "\n";

                for (size_t i = 0; i < args.size(); i++) {
                    const auto& a = args[i];
                    std::cout << "        [" << i << "] " << a.get_name()
                              << " size=" << a.get_size()
                              << " group_id=" << a.get_index();
                    auto mems = a.get_mems();
                    if (!mems.empty()) {
                        std::cout << " mems={";
                        for (size_t m = 0; m < mems.size(); m++) {
                            if (m) std::cout << ",";
                            std::cout << (int)mems[m].get_type();
                        }
                        std::cout << "}";
                    }
                    std::cout << "\n";
                }
            }
        } catch (const std::exception& e) {
            std::cout << "  ERROR: " << e.what() << "\n";
        }
    }

    return 0;
}
