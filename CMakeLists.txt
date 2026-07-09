cmake_minimum_required(VERSION 3.16)
project(ds4_npu_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# FastFlowLM ships no root CMakeLists.txt, so we fetch it download-only via
# FetchContent_Populate (FetchContent_MakeAvailable would try to configure its
# tree).  Allow that pattern explicitly on CMake >= 3.30 where it is otherwise
# a deprecation warning.
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# ── System dependencies (installed via apt / Ryzen AI SW stack) ───────────
# XRT (AMD Xilinx Runtime) and AIEBU are NOT vendored — they are expected to be
# installed on the host (on Ubuntu their CMake config files live under
# /usr/share/cmake/{XRT,AIEBU}). find_package resolves them automatically.
find_package(XRT REQUIRED)
find_package(AIEBU REQUIRED)

# ── Header-only libraries fetched at configure time via FetchContent ──────
# The repo ships ZERO third_party/ vendored folders. cpp-httplib (HTTP server)
# and nlohmann/json (JSON) are pulled from GitHub on the first `cmake` run and
# cached under build/_deps for subsequent offline reconfigures.
include(FetchContent)

# cpp-httplib: HTTP-only build (disable optional OpenSSL / zlib / brotli so the
# header has no system crypto dependency).
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE   OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_OPENSSL          OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_ZLIB             OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.3
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(cpp_httplib)

# nlohmann/json: header-only JSON.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(nlohmann_json)

# FastFlowLM: provides the NPU instruction-sequence builder headers
# (npu_utils/npu_instr_utils.hpp + npu_utils/instr_utils/* + buffer.hpp +
# utils/debug_utils.hpp) that FaStar's NpuSequenceBuilder wraps for in-process
# run_blob dispatch.  Pinned to v0.9.44, whose npu_instr_utils.hpp is
# byte-identical to the version FaStar was developed and verified against.
# Download-only: FastFlowLM ships no root CMakeLists.txt at this tag, so we
# FetchContent_Populate (no add_subdirectory / no configure of its tree) and
# just point the compiler at its src/include.
FetchContent_Declare(
    fastflowlm
    GIT_REPOSITORY https://github.com/FastFlowLM/FastFlowLM.git
    GIT_TAG        v0.9.44
    GIT_SHALLOW    TRUE)
FetchContent_GetProperties(fastflowlm)
if(NOT fastflowlm_POPULATED)
    FetchContent_Populate(fastflowlm)
endif()

# ── Engine executable ──────────────────────────────────────────────────────
add_executable(ds4_npu_engine
    src/fst_engine.cpp
    src/fst_main.cpp
    src/expert_pager.cpp
)

target_compile_options(ds4_npu_engine PRIVATE -O3 -march=native)

target_include_directories(ds4_npu_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # Our FetchContent'd nlohmann/json FIRST, so FastFlowLM's own vendored
    # nlohmann/ (under its src/include) is shadowed — FaStar uses v3.11.3.
    ${nlohmann_json_SOURCE_DIR}/include
    ${fastflowlm_SOURCE_DIR}/src/include
    ${XRT_INCLUDE_DIRS}
    ${AIEBU_INCLUDE_DIRS}
)

target_link_directories(ds4_npu_engine PRIVATE
    ${XRT_LINK_DIRS}
    ${AIEBU_LINK_DIRS}
)

target_link_libraries(ds4_npu_engine PRIVATE
    ${XRT_CORE_LIBRARIES}
    AIEBU::aiebu_static
    httplib::httplib
    nlohmann_json::nlohmann_json
    uuid
    pthread
    dl
)