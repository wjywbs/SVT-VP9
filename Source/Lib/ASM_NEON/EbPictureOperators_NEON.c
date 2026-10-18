/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON picture operators, bit-exact with EbPictureOperators_C.c.
// The residual kernel is not here: clang auto-vectorizes the C loop to the same code.

#include <string.h>
#include "EbNeonCommon.h"
#include "EbPictureOperators_NEON.h"

void eb_vp9_picture_addition_kernel_neon(uint8_t *pred_ptr, uint32_t pred_stride, int16_t *residual_ptr,
                                         uint32_t residual_stride, uint8_t *recon_ptr, uint32_t recon_stride,
                                         uint32_t width, uint32_t height) {
    for (uint32_t y = 0; y < height; ++y) {
        uint32_t x = 0;
        // saturating 16-bit add: a sum above 32767 still clips to 255, and residual + pred >= -32768
        for (; x + 16 <= width; x += 16) {
            const uint8x16_t p  = vld1q_u8(pred_ptr + x);
            const int16x8_t  lo = vqaddq_s16(vld1q_s16(residual_ptr + x), vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p))));
            const int16x8_t  hi = vqaddq_s16(vld1q_s16(residual_ptr + x + 8), vreinterpretq_s16_u16(vmovl_high_u8(p)));
            vst1q_u8(recon_ptr + x, vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi)));
        }
        if (x + 8 <= width) {
            const int16x8_t v = vqaddq_s16(vld1q_s16(residual_ptr + x), vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pred_ptr + x))));
            vst1_u8(recon_ptr + x, vqmovun_s16(v));
            x += 8;
        }
        if (x + 4 <= width) {
            const int16x4_t v = vqadd_s16(vld1_s16(residual_ptr + x),
                                          vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(load_u8_4(pred_ptr + x)))));
            store_u8_4(recon_ptr + x, vqmovun_s16(vcombine_s16(v, v)));
            x += 4;
        }
        for (; x < width; ++x) {
            const int32_t v = residual_ptr[x] + pred_ptr[x];
            recon_ptr[x]    = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
        residual_ptr += residual_stride;
        pred_ptr += pred_stride;
        recon_ptr += recon_stride;
    }
}

void eb_vp9_copy_kernel8_bit_neon(EbByte src, uint32_t src_stride, EbByte dst, uint32_t dst_stride,
                                  uint32_t area_width, uint32_t area_height) {
    switch (area_width) {
    case 4:
        for (uint32_t y = 0; y < area_height; ++y, src += src_stride, dst += dst_stride) memcpy(dst, src, 4);
        break;
    case 8:
        for (uint32_t y = 0; y < area_height; ++y, src += src_stride, dst += dst_stride) vst1_u8(dst, vld1_u8(src));
        break;
    case 16:
        for (uint32_t y = 0; y < area_height; ++y, src += src_stride, dst += dst_stride)
            vst1q_u8(dst, vld1q_u8(src));
        break;
    case 32:
        for (uint32_t y = 0; y < area_height; ++y, src += src_stride, dst += dst_stride)
            vst1q_u8_x2(dst, vld1q_u8_x2(src));
        break;
    case 64:
        for (uint32_t y = 0; y < area_height; ++y, src += src_stride, dst += dst_stride)
            vst1q_u8_x4(dst, vld1q_u8_x4(src));
        break;
    default:
        for (uint32_t y = 0; y < area_height; ++y, src += src_stride, dst += dst_stride) memcpy(dst, src, area_width);
        break;
    }
}

// mode bit 0: residual distortion (int16_t)(coeff - recon)^2, bit 1: prediction distortion coeff^2.
// Squares fit in 32 bits; the sums wrap in 32-bit lanes like the C uint32_t accumulators.
ALWAYS_INLINE void full_distortion(const int16_t *coeff, uint32_t coeff_stride, const int16_t *recon_coeff,
                                   uint32_t recon_coeff_stride, uint32_t area_width, uint32_t area_height,
                                   uint32_t *res_out, uint32_t *pred_out, int mode) {
    int32x4_t res0 = vdupq_n_s32(0), res1 = vdupq_n_s32(0), prd0 = vdupq_n_s32(0), prd1 = vdupq_n_s32(0);
    uint32_t  res_s = 0, prd_s = 0;
    for (uint32_t y = 0; y < area_height; ++y) {
        uint32_t x = 0;
        for (; x + 8 <= area_width; x += 8) {
            const int16x8_t c = vld1q_s16(coeff + x);
            if (mode & 1) {
                const int16x8_t d = vsubq_s16(c, vld1q_s16(recon_coeff + x));
                res0              = vmlal_s16(res0, vget_low_s16(d), vget_low_s16(d));
                res1              = vmlal_high_s16(res1, d, d);
            }
            if (mode & 2) {
                prd0 = vmlal_s16(prd0, vget_low_s16(c), vget_low_s16(c));
                prd1 = vmlal_high_s16(prd1, c, c);
            }
        }
        if (x + 4 <= area_width) {
            const int16x4_t c = vld1_s16(coeff + x);
            if (mode & 1) {
                const int16x4_t d = vsub_s16(c, vld1_s16(recon_coeff + x));
                res0              = vmlal_s16(res0, d, d);
            }
            if (mode & 2)
                prd0 = vmlal_s16(prd0, c, c);
            x += 4;
        }
        for (; x < area_width; ++x) {
            const int16_t d = (int16_t)(coeff[x] - recon_coeff[x]);
            res_s += (uint32_t)(d * d);
            prd_s += (uint32_t)(coeff[x] * coeff[x]);
        }
        coeff += coeff_stride;
        recon_coeff += recon_coeff_stride;
    }
    *res_out  = vaddvq_u32(vreinterpretq_u32_s32(vaddq_s32(res0, res1))) + res_s;
    *pred_out = vaddvq_u32(vreinterpretq_u32_s32(vaddq_s32(prd0, prd1))) + prd_s;
}

void eb_vp9_full_distortion_kernel32bit_neon(int16_t *coeff, uint32_t coeff_stride, int16_t *recon_coeff,
                                             uint32_t recon_coeff_stride, uint64_t distortion_result[2],
                                             uint32_t area_width, uint32_t area_height) {
    uint32_t res, prd;
    full_distortion(coeff, coeff_stride, recon_coeff, recon_coeff_stride, area_width, area_height, &res, &prd, 3);
    distortion_result[0] = res;
    distortion_result[1] = prd;
}

void eb_vp9_full_distortion_kernel_eob_zero32bit_neon(int16_t *coeff, uint32_t coeff_stride, int16_t *recon_coeff,
                                                      uint32_t recon_coeff_stride, uint64_t distortion_result[2],
                                                      uint32_t area_width, uint32_t area_height) {
    uint32_t res, prd;
    full_distortion(coeff, coeff_stride, recon_coeff, recon_coeff_stride, area_width, area_height, &res, &prd, 2);
    distortion_result[0] = prd;
    distortion_result[1] = prd;
}

void eb_vp9_full_distortion_kernel_intra32bit_neon(int16_t *coeff, uint32_t coeff_stride, int16_t *recon_coeff,
                                                   uint32_t recon_coeff_stride, uint64_t distortion_result[2],
                                                   uint32_t area_width, uint32_t area_height) {
    uint32_t res, prd;
    full_distortion(coeff, coeff_stride, recon_coeff, recon_coeff_stride, area_width, area_height, &res, &prd, 1);
    distortion_result[0] = res;
    distortion_result[1] = res;
}

uint64_t eb_vp9_spatial_full_distortion_kernel_neon(uint8_t *input, uint32_t input_stride, uint8_t *recon,
                                                    uint32_t recon_stride, uint32_t area_width, uint32_t area_height) {
    uint64x2_t sum   = vdupq_n_u64(0);
    uint64_t   sum_s = 0;
    for (uint32_t y = 0; y < area_height; ++y) {
        // per row each 32-bit lane collects width / 4 squares (< 2^32 for any width below 2^18)
        uint32x4_t row = vdupq_n_u32(0);
        uint32_t   x   = 0;
        for (; x + 16 <= area_width; x += 16) {
            const uint8x16_t d = vabdq_u8(vld1q_u8(input + x), vld1q_u8(recon + x));
            row                = vpadalq_u16(row, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
            row                = vpadalq_u16(row, vmull_high_u8(d, d));
        }
        if (x + 8 <= area_width) {
            const uint8x8_t d = vabd_u8(vld1_u8(input + x), vld1_u8(recon + x));
            row               = vpadalq_u16(row, vmull_u8(d, d));
            x += 8;
        }
        if (x + 4 <= area_width) {
            const uint8x8_t d = vabd_u8(load_u8_4(input + x), load_u8_4(recon + x));
            row               = vpadalq_u16(row, vmull_u8(d, d)); // upper 4 lanes are 0
            x += 4;
        }
        for (; x < area_width; ++x) {
            const int32_t d = input[x] - recon[x];
            sum_s += (uint64_t)(d * d);
        }
        sum = vpadalq_u32(sum, row);
        input += input_stride;
        recon += recon_stride;
    }
    return vaddvq_u64(sum) + sum_s;
}
