// SPDX-License-Identifier: Apache-2.0
// fst_sv_scale.cc -- Identity scale for SV (no-op)

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void fst_sv_scale(bfloat16 *__restrict data, int32_t D)
{
    (void)data; (void)D;
}
