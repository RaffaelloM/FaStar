// SPDX-License-Identifier: Apache-2.0
//
// fst_sample_kernel.cc -- Vectorized token sampling
//
// AIE2P (NPU2) target.
// Processes TILE bf16 logits at a time using aie::vector operations.
// params[0]=inv_temp, params[1]=global_max, params[2]=inv_sum, params[3]=pass
//   pass=0: temp-scale + find max (vectorized)
//   pass=1: exp(x - global_max) + accumulate sum (vectorized)
//   pass=2: normalize + find argmax (vectorized)

#include <aie_api/aie.hpp>
#include <stdint.h>

#define SM_VEC 64
#define log2e 1.4453125f

extern "C" void fst_sample_kernel(
    bfloat16 *__restrict in,
    bfloat16 *__restrict out,
    float *__restrict aux,
    const float *__restrict params,
    int32_t tile_size)
{
    event0();

    float inv_temp  = params[0];
    float global_max = params[1];
    float inv_sum   = params[2];
    int32_t pass    = (int32_t)params[3];

    const int elem_iters = tile_size / SM_VEC;

    if (pass == 0) {
        // Vectorized temp scale + find max
        // Pattern from softmax.cc: mul bf16 by bf16 scale, convert to float, reduce_max
        aie::vector<bfloat16, SM_VEC> log2e_bf16 = aie::broadcast<bfloat16, SM_VEC>((bfloat16)log2e);
        aie::vector<bfloat16, SM_VEC> inv_temp_bf16 = aie::broadcast<bfloat16, SM_VEC>((bfloat16)inv_temp);
        float max_val = -1e30f;

        for (int i = 0; i < elem_iters; i++) {
            aie::vector<bfloat16, SM_VEC> v_in = aie::load_v<SM_VEC>(in + i * SM_VEC);
            // Scale by inv_temp in bf16
            aie::accum<accfloat, SM_VEC> scaled = aie::mul(v_in, inv_temp_bf16);
            aie::vector<bfloat16, SM_VEC> v_scaled = scaled.to_vector<bfloat16>();
            // Store scaled output
            aie::store_v(out + i * SM_VEC, v_scaled);
            // Find max via reduce
            aie::accum<accfloat, SM_VEC> to_float = aie::mul(v_scaled,
                aie::broadcast<bfloat16, SM_VEC>((bfloat16)1.0f));
            float running_max = aie::reduce_max(to_float.to_vector<bfloat16>());
            if (running_max > max_val) max_val = running_max;
        }
        // Scalar tail
        for (int i = elem_iters * SM_VEC; i < tile_size; i++) {
            float x = (float)in[i] * inv_temp;
            out[i] = (bfloat16)x;
            if (x > max_val) max_val = x;
        }
        aux[0] = max_val;

    } else if (pass == 1) {
        // Vectorized exp(x - global_max) + accumulate sum
        // Pattern from softmax.cc: mul by log2e, sub max, exp2
        aie::vector<bfloat16, SM_VEC> log2e_bf16 = aie::broadcast<bfloat16, SM_VEC>((bfloat16)log2e);
        aie::vector<bfloat16, SM_VEC> gmax_bf16 = aie::broadcast<bfloat16, SM_VEC>((bfloat16)global_max);
        aie::accum<accfloat, SM_VEC> exp_val_accum = aie::zeros<accfloat, SM_VEC>();

        for (int i = 0; i < elem_iters; i++) {
            aie::vector<bfloat16, SM_VEC> v_in = aie::load_v<SM_VEC>(in + i * SM_VEC);
            // Scale by log2e (like softmax.cc)
            aie::accum<accfloat, SM_VEC> scaled_accum = aie::mul(v_in, log2e_bf16);
            // Subtract global_max (in bf16 space)
            aie::accum<accfloat, SM_VEC> exp_in_accum = aie::sub(scaled_accum, gmax_bf16);
            // Compute exp2
            aie::vector<bfloat16, SM_VEC> exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
            // Accumulate sum
            exp_val_accum = aie::add(exp_val_accum, exp_val);
            // Store
            aie::store_v(out + i * SM_VEC, exp_val);
        }
        // Horizontal sum
        aie::vector<float, SM_VEC> reduce = exp_val_accum.to_vector<float>();
        float tile_sum = aie::reduce_add(reduce);
        // Scalar tail
        for (int i = elem_iters * SM_VEC; i < tile_size; i++) {
            float x = (float)in[i];
            float x_log2e = x * log2e;
            float gmax_log2e = global_max * log2e;
            // Use LUT exp for scalar tail
            float diff = x_log2e - gmax_log2e;
            float e;
            if (diff < -10.0f) e = 0.0f;
            else if (diff > 0.0f) e = 1.0f;
            else {
                float idx_f = -diff;
                int i0 = (int)(idx_f * 8.0f);
                if (i0 > 80) i0 = 80;
                int i1 = i0 + 1;
                if (i1 > 80) i1 = 80;
                static const float LUT[83] = {
                    1.0f, 0.8825f, 0.7788f, 0.6873f, 0.6065f, 0.5353f, 0.4724f, 0.4169f,
                    0.3679f, 0.3247f, 0.2865f, 0.2528f, 0.2231f, 0.1969f, 0.1738f, 0.1534f,
                    0.1353f, 0.1194f, 0.1054f, 0.0930f, 0.0821f, 0.0724f, 0.0639f, 0.0564f,
                    0.0498f, 0.0439f, 0.0388f, 0.0342f, 0.0302f, 0.0266f, 0.0235f, 0.0208f,
                    0.0183f, 0.0162f, 0.0143f, 0.0126f, 0.0111f, 0.0098f, 0.0087f, 0.0076f,
                    0.0067f, 0.0059f, 0.0052f, 0.0046f, 0.0041f, 0.0036f, 0.0032f, 0.0028f,
                    0.0025f, 0.0022f, 0.0019f, 0.0017f, 0.0015f, 0.0013f, 0.0012f, 0.0010f,
                    0.0009f, 0.0008f, 0.0007f, 0.0006f, 0.0006f, 0.0005f, 0.0004f, 0.0004f,
                    0.0003f, 0.0003f, 0.0003f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0001f,
                    0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0000f, 0.0000f, 0.0000f,
                    0.0000f
                };
                float frac = (idx_f * 8.0f) - (float)i0;
                e = LUT[i0] + frac * (LUT[i1] - LUT[i0]);
            }
            tile_sum += e;
            out[i] = (bfloat16)e;
        }
        aux[0] = tile_sum;

    } else if (pass == 2) {
        // Vectorized normalize + find argmax
        aie::vector<bfloat16, SM_VEC> inv_sum_bf16 = aie::broadcast<bfloat16, SM_VEC>((bfloat16)inv_sum);
        int32_t tile_argmax = 0;
        float tile_max_p = -1.0f;

        for (int i = 0; i < elem_iters; i++) {
            aie::vector<bfloat16, SM_VEC> v_in = aie::load_v<SM_VEC>(in + i * SM_VEC);
            // Normalize: multiply by inv_sum
            aie::accum<accfloat, SM_VEC> normalized = aie::mul(v_in, inv_sum_bf16);
            aie::vector<bfloat16, SM_VEC> v_norm = normalized.to_vector<bfloat16>();
            // Store
            aie::store_v(out + i * SM_VEC, v_norm);
            // Find max prob and index (scalar scan of vector)
            for (int j = 0; j < SM_VEC; j++) {
                float p = (float)v_norm[j];
                if (p > tile_max_p) {
                    tile_max_p = p;
                    tile_argmax = i * SM_VEC + j;
                }
            }
        }
        // Scalar tail
        for (int i = elem_iters * SM_VEC; i < tile_size; i++) {
            float p = (float)in[i] * inv_sum;
            out[i] = (bfloat16)p;
            if (p > tile_max_p) {
                tile_max_p = p;
                tile_argmax = i;
            }
        }
        aux[0] = (float)tile_argmax;
        aux[1] = tile_max_p;
    }

    event1();
}
