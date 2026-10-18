/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON block means (picture analysis), bit-exact with EbComputeMean_C.c (exact integer sums).

#include "EbNeonCommon.h"
#include "EbComputeMean_NEON.h"
#include "EbComputeMean_C.h"

#define VARIANCE_PRECISION 16

// Sums of pixels and of squared pixels over rows 0, step, 2 * step, ... (width multiple of 8)
ALWAYS_INLINE void block_sums(const uint8_t *p, uint32_t stride, uint32_t w, uint32_t h, uint32_t step,
                              uint64_t *sum, uint64_t *sum_sq, int want_sq) {
    uint64x2_t s = vdupq_n_u64(0), q = vdupq_n_u64(0);
    for (uint32_t y = 0; y < h; y += step, p += step * stride) {
        // one row: < 2^16 * 255 per 32-bit lane for any width below 2^15
        uint32x4_t rs = vdupq_n_u32(0), rq = vdupq_n_u32(0);
        for (uint32_t x = 0; x < w; x += 8) {
            const uint8x8_t v = vld1_u8(p + x);
            rs                = vaddw_u16(rs, vget_low_u16(vmovl_u8(v)));
            rs                = vaddw_high_u16(rs, vmovl_u8(v));
            if (want_sq)
                rq = vpadalq_u16(rq, vmull_u8(v, v));
        }
        s = vpadalq_u32(s, rs);
        q = vpadalq_u32(q, rq);
    }
    *sum    = vaddvq_u64(s);
    *sum_sq = vaddvq_u64(q);
}

uint64_t eb_vp9_compute_mean_neon(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width,
                                  uint32_t input_area_height) {
    if (input_area_width & 7)
        return compute_mean(input_samples, input_stride, input_area_width, input_area_height);
    uint64_t sum, sq;
    block_sums(input_samples, input_stride, input_area_width, input_area_height, 1, &sum, &sq, 0);
    return (sum << (VARIANCE_PRECISION >> 1)) / (input_area_width * input_area_height);
}

uint64_t eb_vp9_compute_mean_of_squared_values_neon(uint8_t *input_samples, uint32_t input_stride,
                                                    uint32_t input_area_width, uint32_t input_area_height) {
    if (input_area_width & 7)
        return compute_mean_of_squared_values(input_samples, input_stride, input_area_width, input_area_height);
    uint64_t sum, sq;
    block_sums(input_samples, input_stride, input_area_width, input_area_height, 1, &sum, &sq, 1);
    return (sq << VARIANCE_PRECISION) / (input_area_width * input_area_height);
}

uint64_t eb_vp9_compute_sub_mean8x8_neon(uint8_t *input_samples, uint16_t input_stride) {
    uint64_t sum, sq;
    block_sums(input_samples, input_stride, 8, 8, 2, &sum, &sq, 0);
    return (uint64_t)(uint32_t)sum << 3;
}

uint64_t eb_vp9_compute_subd_mean_of_squared_values8x8_neon(uint8_t *input_samples, uint16_t input_stride) {
    uint64_t sum, sq;
    block_sums(input_samples, input_stride, 8, 8, 2, &sum, &sq, 1);
    return (uint64_t)(uint32_t)sq << 11;
}
