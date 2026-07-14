// tools/gdn_uuids.cpp — print xclbin UUIDs to detect collisions.
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    xrt::device dev(0);
    for (int i = 1; i < argc; i++) {
        try {
            xrt::xclbin xb(std::string(argv[i]));
            printf("%s: %s\n", argv[i], xb.get_uuid().to_string().c_str());
        } catch (const std::exception& e) {
            printf("%s: EXC %s\n", argv[i], e.what());
        }
    }
    return 0;
}