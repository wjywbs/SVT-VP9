/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON inverse transforms, bit-exact with inv_txfm.c / vp9_idct.c (for any input).
// - idct8 / idct16 / idct32 store every intermediate in int16_t (wrapping): they run in int16x8 lanes (8 rows or
//   8 columns at a time) with wrapping 16-bit adds; the products are formed in 32 bits (vmull / vmlal, sums of
//   products are distributed (a + b) * c = a * c + b * c, exact in 32 bits), and vrshrn_n_s32(x, 14) gives
//   (x + 8192) >> 14 wrapped to 16 bits as in C.
// - iadst8 / iadst16 keep 32-bit intermediates: they run in int32x4 lanes (as the forward transforms).
// The 1-D transforms are generated from the C code by a translator.

#include "EbNeonCommon.h"
#include "vpx_dsp_rtcd.h"
#include "vp9_rtcd.h"
#include "txfm_common.h"
#include "mem.h"

// (int32_t)(int16_t)x
ALWAYS_INLINE int32x4_t wrap16_s32(const int32x4_t x) { return vmovl_s16(vmovn_s32(x)); }

/*******************************************
 * 1-D transforms (generated from the C code)
 *******************************************/
static void idct8_x8(const int16x8_t *input, int16x8_t *output) {
    int16x8_t step1_0, step1_2, step1_1, step1_3, step1_4, step1_7, step1_5, step1_6, step2_0, step2_1, step2_2,
        step2_3, step2_4, step2_5, step2_6, step2_7;
    int32x4_t temp1_lo, temp1_hi, temp2_lo, temp2_hi;
    step1_0 = input[0];
    step1_2 = input[4];
    step1_1 = input[2];
    step1_3 = input[6];
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[1]), cospi_28_64), vget_low_s16(input[7]), -cospi_4_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[1], cospi_28_64), input[7], -cospi_4_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[1]), cospi_4_64), vget_low_s16(input[7]), cospi_28_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[1], cospi_4_64), input[7], cospi_28_64);
    step1_4 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_7 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[5]), cospi_12_64), vget_low_s16(input[3]), -cospi_20_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[5], cospi_12_64), input[3], -cospi_20_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[5]), cospi_20_64), vget_low_s16(input[3]), cospi_12_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[5], cospi_20_64), input[3], cospi_12_64);
    step1_5 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_6 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_0), cospi_16_64), vget_low_s16(step1_2), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_0, cospi_16_64), step1_2, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_0), cospi_16_64), vget_low_s16(step1_2), -cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_0, cospi_16_64), step1_2, -cospi_16_64);
    step2_0 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_1 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_1), cospi_24_64), vget_low_s16(step1_3), -cospi_8_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_1, cospi_24_64), step1_3, -cospi_8_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_1), cospi_8_64), vget_low_s16(step1_3), cospi_24_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_1, cospi_8_64), step1_3, cospi_24_64);
    step2_2 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_3 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_4 = vaddq_s16(step1_4, step1_5);
    step2_5 = vsubq_s16(step1_4, step1_5);
    step2_6 = vsubq_s16(step1_7, step1_6);
    step2_7 = vaddq_s16(step1_6, step1_7);
    step1_0 = vaddq_s16(step2_0, step2_3);
    step1_1 = vaddq_s16(step2_1, step2_2);
    step1_2 = vsubq_s16(step2_1, step2_2);
    step1_3 = vsubq_s16(step2_0, step2_3);
    step1_4 = step2_4;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_6), cospi_16_64), vget_low_s16(step2_5), -cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_6, cospi_16_64), step2_5, -cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_5), cospi_16_64), vget_low_s16(step2_6), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_5, cospi_16_64), step2_6, cospi_16_64);
    step1_5 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_6 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_7 = step2_7;
    output[0] = vaddq_s16(step1_0, step1_7);
    output[1] = vaddq_s16(step1_1, step1_6);
    output[2] = vaddq_s16(step1_2, step1_5);
    output[3] = vaddq_s16(step1_3, step1_4);
    output[4] = vsubq_s16(step1_3, step1_4);
    output[5] = vsubq_s16(step1_2, step1_5);
    output[6] = vsubq_s16(step1_1, step1_6);
    output[7] = vsubq_s16(step1_0, step1_7);
}

static void idct16_x8(const int16x8_t *input, int16x8_t *output) {
    int16x8_t step1_0, step1_1, step1_2, step1_3, step1_4, step1_5, step1_6, step1_7, step1_8, step1_9, step1_10,
        step1_11, step1_12, step1_13, step1_14, step1_15, step2_0, step2_1, step2_2, step2_3, step2_4, step2_5,
        step2_6, step2_7, step2_8, step2_15, step2_9, step2_14, step2_10, step2_13, step2_11, step2_12;
    int32x4_t temp1_lo, temp1_hi, temp2_lo, temp2_hi;
    step1_0 = input[0];
    step1_1 = input[8];
    step1_2 = input[4];
    step1_3 = input[12];
    step1_4 = input[2];
    step1_5 = input[10];
    step1_6 = input[6];
    step1_7 = input[14];
    step1_8 = input[1];
    step1_9 = input[9];
    step1_10 = input[5];
    step1_11 = input[13];
    step1_12 = input[3];
    step1_13 = input[11];
    step1_14 = input[7];
    step1_15 = input[15];
    step2_0 = step1_0;
    step2_1 = step1_1;
    step2_2 = step1_2;
    step2_3 = step1_3;
    step2_4 = step1_4;
    step2_5 = step1_5;
    step2_6 = step1_6;
    step2_7 = step1_7;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_8), cospi_30_64), vget_low_s16(step1_15), -cospi_2_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_8, cospi_30_64), step1_15, -cospi_2_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_8), cospi_2_64), vget_low_s16(step1_15), cospi_30_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_8, cospi_2_64), step1_15, cospi_30_64);
    step2_8 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_15 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), cospi_14_64), vget_low_s16(step1_14), -cospi_18_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, cospi_14_64), step1_14, -cospi_18_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), cospi_18_64), vget_low_s16(step1_14), cospi_14_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, cospi_18_64), step1_14, cospi_14_64);
    step2_9 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_14 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), cospi_22_64), vget_low_s16(step1_13), -cospi_10_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, cospi_22_64), step1_13, -cospi_10_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), cospi_10_64), vget_low_s16(step1_13), cospi_22_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, cospi_10_64), step1_13, cospi_22_64);
    step2_10 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_13 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), cospi_6_64), vget_low_s16(step1_12), -cospi_26_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, cospi_6_64), step1_12, -cospi_26_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), cospi_26_64), vget_low_s16(step1_12), cospi_6_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, cospi_26_64), step1_12, cospi_6_64);
    step2_11 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_12 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_0 = step2_0;
    step1_1 = step2_1;
    step1_2 = step2_2;
    step1_3 = step2_3;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_4), cospi_28_64), vget_low_s16(step2_7), -cospi_4_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_4, cospi_28_64), step2_7, -cospi_4_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_4), cospi_4_64), vget_low_s16(step2_7), cospi_28_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_4, cospi_4_64), step2_7, cospi_28_64);
    step1_4 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_7 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_5), cospi_12_64), vget_low_s16(step2_6), -cospi_20_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_5, cospi_12_64), step2_6, -cospi_20_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_5), cospi_20_64), vget_low_s16(step2_6), cospi_12_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_5, cospi_20_64), step2_6, cospi_12_64);
    step1_5 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_6 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_8 = vaddq_s16(step2_8, step2_9);
    step1_9 = vsubq_s16(step2_8, step2_9);
    step1_10 = vsubq_s16(step2_11, step2_10);
    step1_11 = vaddq_s16(step2_10, step2_11);
    step1_12 = vaddq_s16(step2_12, step2_13);
    step1_13 = vsubq_s16(step2_12, step2_13);
    step1_14 = vsubq_s16(step2_15, step2_14);
    step1_15 = vaddq_s16(step2_14, step2_15);
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_0), cospi_16_64), vget_low_s16(step1_1), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_0, cospi_16_64), step1_1, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_0), cospi_16_64), vget_low_s16(step1_1), -cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_0, cospi_16_64), step1_1, -cospi_16_64);
    step2_0 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_1 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_2), cospi_24_64), vget_low_s16(step1_3), -cospi_8_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_2, cospi_24_64), step1_3, -cospi_8_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_2), cospi_8_64), vget_low_s16(step1_3), cospi_24_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_2, cospi_8_64), step1_3, cospi_24_64);
    step2_2 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_3 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_4 = vaddq_s16(step1_4, step1_5);
    step2_5 = vsubq_s16(step1_4, step1_5);
    step2_6 = vsubq_s16(step1_7, step1_6);
    step2_7 = vaddq_s16(step1_6, step1_7);
    step2_8 = step1_8;
    step2_15 = step1_15;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), -cospi_8_64), vget_low_s16(step1_14), cospi_24_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, -cospi_8_64), step1_14, cospi_24_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), cospi_24_64), vget_low_s16(step1_14), cospi_8_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, cospi_24_64), step1_14, cospi_8_64);
    step2_9 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_14 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), -cospi_24_64), vget_low_s16(step1_13), -cospi_8_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, -cospi_24_64), step1_13, -cospi_8_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), -cospi_8_64), vget_low_s16(step1_13), cospi_24_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, -cospi_8_64), step1_13, cospi_24_64);
    step2_10 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_13 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_11 = step1_11;
    step2_12 = step1_12;
    step1_0 = vaddq_s16(step2_0, step2_3);
    step1_1 = vaddq_s16(step2_1, step2_2);
    step1_2 = vsubq_s16(step2_1, step2_2);
    step1_3 = vsubq_s16(step2_0, step2_3);
    step1_4 = step2_4;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_6), cospi_16_64), vget_low_s16(step2_5), -cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_6, cospi_16_64), step2_5, -cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_5), cospi_16_64), vget_low_s16(step2_6), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_5, cospi_16_64), step2_6, cospi_16_64);
    step1_5 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_6 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_7 = step2_7;
    step1_8 = vaddq_s16(step2_8, step2_11);
    step1_9 = vaddq_s16(step2_9, step2_10);
    step1_10 = vsubq_s16(step2_9, step2_10);
    step1_11 = vsubq_s16(step2_8, step2_11);
    step1_12 = vsubq_s16(step2_15, step2_12);
    step1_13 = vsubq_s16(step2_14, step2_13);
    step1_14 = vaddq_s16(step2_13, step2_14);
    step1_15 = vaddq_s16(step2_12, step2_15);
    step2_0 = vaddq_s16(step1_0, step1_7);
    step2_1 = vaddq_s16(step1_1, step1_6);
    step2_2 = vaddq_s16(step1_2, step1_5);
    step2_3 = vaddq_s16(step1_3, step1_4);
    step2_4 = vsubq_s16(step1_3, step1_4);
    step2_5 = vsubq_s16(step1_2, step1_5);
    step2_6 = vsubq_s16(step1_1, step1_6);
    step2_7 = vsubq_s16(step1_0, step1_7);
    step2_8 = step1_8;
    step2_9 = step1_9;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), -cospi_16_64), vget_low_s16(step1_13), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, -cospi_16_64), step1_13, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), cospi_16_64), vget_low_s16(step1_13), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, cospi_16_64), step1_13, cospi_16_64);
    step2_10 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_13 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), -cospi_16_64), vget_low_s16(step1_12), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, -cospi_16_64), step1_12, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), cospi_16_64), vget_low_s16(step1_12), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, cospi_16_64), step1_12, cospi_16_64);
    step2_11 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_12 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_14 = step1_14;
    step2_15 = step1_15;
    output[0] = vaddq_s16(step2_0, step2_15);
    output[1] = vaddq_s16(step2_1, step2_14);
    output[2] = vaddq_s16(step2_2, step2_13);
    output[3] = vaddq_s16(step2_3, step2_12);
    output[4] = vaddq_s16(step2_4, step2_11);
    output[5] = vaddq_s16(step2_5, step2_10);
    output[6] = vaddq_s16(step2_6, step2_9);
    output[7] = vaddq_s16(step2_7, step2_8);
    output[8] = vsubq_s16(step2_7, step2_8);
    output[9] = vsubq_s16(step2_6, step2_9);
    output[10] = vsubq_s16(step2_5, step2_10);
    output[11] = vsubq_s16(step2_4, step2_11);
    output[12] = vsubq_s16(step2_3, step2_12);
    output[13] = vsubq_s16(step2_2, step2_13);
    output[14] = vsubq_s16(step2_1, step2_14);
    output[15] = vsubq_s16(step2_0, step2_15);
}

static void idct32_x8(const int16x8_t *input, int16x8_t *output) {
    int16x8_t step1_0, step1_1, step1_2, step1_3, step1_4, step1_5, step1_6, step1_7, step1_8, step1_9, step1_10,
        step1_11, step1_12, step1_13, step1_14, step1_15, step1_16, step1_31, step1_17, step1_30, step1_18,
        step1_29, step1_19, step1_28, step1_20, step1_27, step1_21, step1_26, step1_22, step1_25, step1_23,
        step1_24, step2_0, step2_1, step2_2, step2_3, step2_4, step2_5, step2_6, step2_7, step2_8, step2_15,
        step2_9, step2_14, step2_10, step2_13, step2_11, step2_12, step2_16, step2_17, step2_18, step2_19, step2_20,
        step2_21, step2_22, step2_23, step2_24, step2_25, step2_26, step2_27, step2_28, step2_29, step2_30,
        step2_31;
    int32x4_t temp1_lo, temp1_hi, temp2_lo, temp2_hi;
    step1_0 = input[0];
    step1_1 = input[16];
    step1_2 = input[8];
    step1_3 = input[24];
    step1_4 = input[4];
    step1_5 = input[20];
    step1_6 = input[12];
    step1_7 = input[28];
    step1_8 = input[2];
    step1_9 = input[18];
    step1_10 = input[10];
    step1_11 = input[26];
    step1_12 = input[6];
    step1_13 = input[22];
    step1_14 = input[14];
    step1_15 = input[30];
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[1]), cospi_31_64), vget_low_s16(input[31]), -cospi_1_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[1], cospi_31_64), input[31], -cospi_1_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[1]), cospi_1_64), vget_low_s16(input[31]), cospi_31_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[1], cospi_1_64), input[31], cospi_31_64);
    step1_16 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_31 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[17]), cospi_15_64), vget_low_s16(input[15]), -cospi_17_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[17], cospi_15_64), input[15], -cospi_17_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[17]), cospi_17_64), vget_low_s16(input[15]), cospi_15_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[17], cospi_17_64), input[15], cospi_15_64);
    step1_17 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_30 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[9]), cospi_23_64), vget_low_s16(input[23]), -cospi_9_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[9], cospi_23_64), input[23], -cospi_9_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[9]), cospi_9_64), vget_low_s16(input[23]), cospi_23_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[9], cospi_9_64), input[23], cospi_23_64);
    step1_18 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_29 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[25]), cospi_7_64), vget_low_s16(input[7]), -cospi_25_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[25], cospi_7_64), input[7], -cospi_25_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[25]), cospi_25_64), vget_low_s16(input[7]), cospi_7_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[25], cospi_25_64), input[7], cospi_7_64);
    step1_19 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_28 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[5]), cospi_27_64), vget_low_s16(input[27]), -cospi_5_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[5], cospi_27_64), input[27], -cospi_5_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[5]), cospi_5_64), vget_low_s16(input[27]), cospi_27_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[5], cospi_5_64), input[27], cospi_27_64);
    step1_20 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_27 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[21]), cospi_11_64), vget_low_s16(input[11]), -cospi_21_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[21], cospi_11_64), input[11], -cospi_21_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[21]), cospi_21_64), vget_low_s16(input[11]), cospi_11_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[21], cospi_21_64), input[11], cospi_11_64);
    step1_21 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_26 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[13]), cospi_19_64), vget_low_s16(input[19]), -cospi_13_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[13], cospi_19_64), input[19], -cospi_13_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[13]), cospi_13_64), vget_low_s16(input[19]), cospi_19_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[13], cospi_13_64), input[19], cospi_19_64);
    step1_22 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_25 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[29]), cospi_3_64), vget_low_s16(input[3]), -cospi_29_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(input[29], cospi_3_64), input[3], -cospi_29_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(input[29]), cospi_29_64), vget_low_s16(input[3]), cospi_3_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(input[29], cospi_29_64), input[3], cospi_3_64);
    step1_23 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_24 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_0 = step1_0;
    step2_1 = step1_1;
    step2_2 = step1_2;
    step2_3 = step1_3;
    step2_4 = step1_4;
    step2_5 = step1_5;
    step2_6 = step1_6;
    step2_7 = step1_7;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_8), cospi_30_64), vget_low_s16(step1_15), -cospi_2_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_8, cospi_30_64), step1_15, -cospi_2_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_8), cospi_2_64), vget_low_s16(step1_15), cospi_30_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_8, cospi_2_64), step1_15, cospi_30_64);
    step2_8 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_15 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), cospi_14_64), vget_low_s16(step1_14), -cospi_18_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, cospi_14_64), step1_14, -cospi_18_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), cospi_18_64), vget_low_s16(step1_14), cospi_14_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, cospi_18_64), step1_14, cospi_14_64);
    step2_9 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_14 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), cospi_22_64), vget_low_s16(step1_13), -cospi_10_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, cospi_22_64), step1_13, -cospi_10_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), cospi_10_64), vget_low_s16(step1_13), cospi_22_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, cospi_10_64), step1_13, cospi_22_64);
    step2_10 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_13 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), cospi_6_64), vget_low_s16(step1_12), -cospi_26_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, cospi_6_64), step1_12, -cospi_26_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), cospi_26_64), vget_low_s16(step1_12), cospi_6_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, cospi_26_64), step1_12, cospi_6_64);
    step2_11 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_12 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_16 = vaddq_s16(step1_16, step1_17);
    step2_17 = vsubq_s16(step1_16, step1_17);
    step2_18 = vsubq_s16(step1_19, step1_18);
    step2_19 = vaddq_s16(step1_18, step1_19);
    step2_20 = vaddq_s16(step1_20, step1_21);
    step2_21 = vsubq_s16(step1_20, step1_21);
    step2_22 = vsubq_s16(step1_23, step1_22);
    step2_23 = vaddq_s16(step1_22, step1_23);
    step2_24 = vaddq_s16(step1_24, step1_25);
    step2_25 = vsubq_s16(step1_24, step1_25);
    step2_26 = vsubq_s16(step1_27, step1_26);
    step2_27 = vaddq_s16(step1_26, step1_27);
    step2_28 = vaddq_s16(step1_28, step1_29);
    step2_29 = vsubq_s16(step1_28, step1_29);
    step2_30 = vsubq_s16(step1_31, step1_30);
    step2_31 = vaddq_s16(step1_30, step1_31);
    step1_0 = step2_0;
    step1_1 = step2_1;
    step1_2 = step2_2;
    step1_3 = step2_3;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_4), cospi_28_64), vget_low_s16(step2_7), -cospi_4_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_4, cospi_28_64), step2_7, -cospi_4_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_4), cospi_4_64), vget_low_s16(step2_7), cospi_28_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_4, cospi_4_64), step2_7, cospi_28_64);
    step1_4 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_7 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_5), cospi_12_64), vget_low_s16(step2_6), -cospi_20_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_5, cospi_12_64), step2_6, -cospi_20_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_5), cospi_20_64), vget_low_s16(step2_6), cospi_12_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_5, cospi_20_64), step2_6, cospi_12_64);
    step1_5 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_6 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_8 = vaddq_s16(step2_8, step2_9);
    step1_9 = vsubq_s16(step2_8, step2_9);
    step1_10 = vsubq_s16(step2_11, step2_10);
    step1_11 = vaddq_s16(step2_10, step2_11);
    step1_12 = vaddq_s16(step2_12, step2_13);
    step1_13 = vsubq_s16(step2_12, step2_13);
    step1_14 = vsubq_s16(step2_15, step2_14);
    step1_15 = vaddq_s16(step2_14, step2_15);
    step1_16 = step2_16;
    step1_31 = step2_31;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_17), -cospi_4_64), vget_low_s16(step2_30), cospi_28_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_17, -cospi_4_64), step2_30, cospi_28_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_17), cospi_28_64), vget_low_s16(step2_30), cospi_4_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_17, cospi_28_64), step2_30, cospi_4_64);
    step1_17 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_30 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_18), -cospi_28_64), vget_low_s16(step2_29), -cospi_4_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_18, -cospi_28_64), step2_29, -cospi_4_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_18), -cospi_4_64), vget_low_s16(step2_29), cospi_28_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_18, -cospi_4_64), step2_29, cospi_28_64);
    step1_18 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_29 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_19 = step2_19;
    step1_20 = step2_20;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_21), -cospi_20_64), vget_low_s16(step2_26), cospi_12_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_21, -cospi_20_64), step2_26, cospi_12_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_21), cospi_12_64), vget_low_s16(step2_26), cospi_20_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_21, cospi_12_64), step2_26, cospi_20_64);
    step1_21 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_26 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_22), -cospi_12_64), vget_low_s16(step2_25), -cospi_20_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_22, -cospi_12_64), step2_25, -cospi_20_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_22), -cospi_20_64), vget_low_s16(step2_25), cospi_12_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_22, -cospi_20_64), step2_25, cospi_12_64);
    step1_22 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_25 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_23 = step2_23;
    step1_24 = step2_24;
    step1_27 = step2_27;
    step1_28 = step2_28;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_0), cospi_16_64), vget_low_s16(step1_1), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_0, cospi_16_64), step1_1, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_0), cospi_16_64), vget_low_s16(step1_1), -cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_0, cospi_16_64), step1_1, -cospi_16_64);
    step2_0 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_1 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_2), cospi_24_64), vget_low_s16(step1_3), -cospi_8_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_2, cospi_24_64), step1_3, -cospi_8_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_2), cospi_8_64), vget_low_s16(step1_3), cospi_24_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_2, cospi_8_64), step1_3, cospi_24_64);
    step2_2 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_3 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_4 = vaddq_s16(step1_4, step1_5);
    step2_5 = vsubq_s16(step1_4, step1_5);
    step2_6 = vsubq_s16(step1_7, step1_6);
    step2_7 = vaddq_s16(step1_6, step1_7);
    step2_8 = step1_8;
    step2_15 = step1_15;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), -cospi_8_64), vget_low_s16(step1_14), cospi_24_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, -cospi_8_64), step1_14, cospi_24_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_9), cospi_24_64), vget_low_s16(step1_14), cospi_8_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_9, cospi_24_64), step1_14, cospi_8_64);
    step2_9 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_14 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), -cospi_24_64), vget_low_s16(step1_13), -cospi_8_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, -cospi_24_64), step1_13, -cospi_8_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), -cospi_8_64), vget_low_s16(step1_13), cospi_24_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, -cospi_8_64), step1_13, cospi_24_64);
    step2_10 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_13 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_11 = step1_11;
    step2_12 = step1_12;
    step2_16 = vaddq_s16(step1_16, step1_19);
    step2_17 = vaddq_s16(step1_17, step1_18);
    step2_18 = vsubq_s16(step1_17, step1_18);
    step2_19 = vsubq_s16(step1_16, step1_19);
    step2_20 = vsubq_s16(step1_23, step1_20);
    step2_21 = vsubq_s16(step1_22, step1_21);
    step2_22 = vaddq_s16(step1_21, step1_22);
    step2_23 = vaddq_s16(step1_20, step1_23);
    step2_24 = vaddq_s16(step1_24, step1_27);
    step2_25 = vaddq_s16(step1_25, step1_26);
    step2_26 = vsubq_s16(step1_25, step1_26);
    step2_27 = vsubq_s16(step1_24, step1_27);
    step2_28 = vsubq_s16(step1_31, step1_28);
    step2_29 = vsubq_s16(step1_30, step1_29);
    step2_30 = vaddq_s16(step1_29, step1_30);
    step2_31 = vaddq_s16(step1_28, step1_31);
    step1_0 = vaddq_s16(step2_0, step2_3);
    step1_1 = vaddq_s16(step2_1, step2_2);
    step1_2 = vsubq_s16(step2_1, step2_2);
    step1_3 = vsubq_s16(step2_0, step2_3);
    step1_4 = step2_4;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_6), cospi_16_64), vget_low_s16(step2_5), -cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_6, cospi_16_64), step2_5, -cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_5), cospi_16_64), vget_low_s16(step2_6), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_5, cospi_16_64), step2_6, cospi_16_64);
    step1_5 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_6 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_7 = step2_7;
    step1_8 = vaddq_s16(step2_8, step2_11);
    step1_9 = vaddq_s16(step2_9, step2_10);
    step1_10 = vsubq_s16(step2_9, step2_10);
    step1_11 = vsubq_s16(step2_8, step2_11);
    step1_12 = vsubq_s16(step2_15, step2_12);
    step1_13 = vsubq_s16(step2_14, step2_13);
    step1_14 = vaddq_s16(step2_13, step2_14);
    step1_15 = vaddq_s16(step2_12, step2_15);
    step1_16 = step2_16;
    step1_17 = step2_17;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_18), -cospi_8_64), vget_low_s16(step2_29), cospi_24_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_18, -cospi_8_64), step2_29, cospi_24_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_18), cospi_24_64), vget_low_s16(step2_29), cospi_8_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_18, cospi_24_64), step2_29, cospi_8_64);
    step1_18 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_29 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_19), -cospi_8_64), vget_low_s16(step2_28), cospi_24_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_19, -cospi_8_64), step2_28, cospi_24_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_19), cospi_24_64), vget_low_s16(step2_28), cospi_8_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_19, cospi_24_64), step2_28, cospi_8_64);
    step1_19 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_28 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_20), -cospi_24_64), vget_low_s16(step2_27), -cospi_8_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_20, -cospi_24_64), step2_27, -cospi_8_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_20), -cospi_8_64), vget_low_s16(step2_27), cospi_24_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_20, -cospi_8_64), step2_27, cospi_24_64);
    step1_20 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_27 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_21), -cospi_24_64), vget_low_s16(step2_26), -cospi_8_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_21, -cospi_24_64), step2_26, -cospi_8_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_21), -cospi_8_64), vget_low_s16(step2_26), cospi_24_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_21, -cospi_8_64), step2_26, cospi_24_64);
    step1_21 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_26 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_22 = step2_22;
    step1_23 = step2_23;
    step1_24 = step2_24;
    step1_25 = step2_25;
    step1_30 = step2_30;
    step1_31 = step2_31;
    step2_0 = vaddq_s16(step1_0, step1_7);
    step2_1 = vaddq_s16(step1_1, step1_6);
    step2_2 = vaddq_s16(step1_2, step1_5);
    step2_3 = vaddq_s16(step1_3, step1_4);
    step2_4 = vsubq_s16(step1_3, step1_4);
    step2_5 = vsubq_s16(step1_2, step1_5);
    step2_6 = vsubq_s16(step1_1, step1_6);
    step2_7 = vsubq_s16(step1_0, step1_7);
    step2_8 = step1_8;
    step2_9 = step1_9;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), -cospi_16_64), vget_low_s16(step1_13), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, -cospi_16_64), step1_13, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_10), cospi_16_64), vget_low_s16(step1_13), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_10, cospi_16_64), step1_13, cospi_16_64);
    step2_10 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_13 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), -cospi_16_64), vget_low_s16(step1_12), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, -cospi_16_64), step1_12, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step1_11), cospi_16_64), vget_low_s16(step1_12), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step1_11, cospi_16_64), step1_12, cospi_16_64);
    step2_11 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step2_12 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step2_14 = step1_14;
    step2_15 = step1_15;
    step2_16 = vaddq_s16(step1_16, step1_23);
    step2_17 = vaddq_s16(step1_17, step1_22);
    step2_18 = vaddq_s16(step1_18, step1_21);
    step2_19 = vaddq_s16(step1_19, step1_20);
    step2_20 = vsubq_s16(step1_19, step1_20);
    step2_21 = vsubq_s16(step1_18, step1_21);
    step2_22 = vsubq_s16(step1_17, step1_22);
    step2_23 = vsubq_s16(step1_16, step1_23);
    step2_24 = vsubq_s16(step1_31, step1_24);
    step2_25 = vsubq_s16(step1_30, step1_25);
    step2_26 = vsubq_s16(step1_29, step1_26);
    step2_27 = vsubq_s16(step1_28, step1_27);
    step2_28 = vaddq_s16(step1_27, step1_28);
    step2_29 = vaddq_s16(step1_26, step1_29);
    step2_30 = vaddq_s16(step1_25, step1_30);
    step2_31 = vaddq_s16(step1_24, step1_31);
    step1_0 = vaddq_s16(step2_0, step2_15);
    step1_1 = vaddq_s16(step2_1, step2_14);
    step1_2 = vaddq_s16(step2_2, step2_13);
    step1_3 = vaddq_s16(step2_3, step2_12);
    step1_4 = vaddq_s16(step2_4, step2_11);
    step1_5 = vaddq_s16(step2_5, step2_10);
    step1_6 = vaddq_s16(step2_6, step2_9);
    step1_7 = vaddq_s16(step2_7, step2_8);
    step1_8 = vsubq_s16(step2_7, step2_8);
    step1_9 = vsubq_s16(step2_6, step2_9);
    step1_10 = vsubq_s16(step2_5, step2_10);
    step1_11 = vsubq_s16(step2_4, step2_11);
    step1_12 = vsubq_s16(step2_3, step2_12);
    step1_13 = vsubq_s16(step2_2, step2_13);
    step1_14 = vsubq_s16(step2_1, step2_14);
    step1_15 = vsubq_s16(step2_0, step2_15);
    step1_16 = step2_16;
    step1_17 = step2_17;
    step1_18 = step2_18;
    step1_19 = step2_19;
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_20), -cospi_16_64), vget_low_s16(step2_27), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_20, -cospi_16_64), step2_27, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_20), cospi_16_64), vget_low_s16(step2_27), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_20, cospi_16_64), step2_27, cospi_16_64);
    step1_20 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_27 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_21), -cospi_16_64), vget_low_s16(step2_26), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_21, -cospi_16_64), step2_26, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_21), cospi_16_64), vget_low_s16(step2_26), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_21, cospi_16_64), step2_26, cospi_16_64);
    step1_21 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_26 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_22), -cospi_16_64), vget_low_s16(step2_25), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_22, -cospi_16_64), step2_25, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_22), cospi_16_64), vget_low_s16(step2_25), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_22, cospi_16_64), step2_25, cospi_16_64);
    step1_22 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_25 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    temp1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_23), -cospi_16_64), vget_low_s16(step2_24), cospi_16_64);
    temp1_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_23, -cospi_16_64), step2_24, cospi_16_64);
    temp2_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(step2_23), cospi_16_64), vget_low_s16(step2_24), cospi_16_64);
    temp2_hi = vmlal_high_n_s16(vmull_high_n_s16(step2_23, cospi_16_64), step2_24, cospi_16_64);
    step1_23 = vcombine_s16(vrshrn_n_s32(temp1_lo, DCT_CONST_BITS), vrshrn_n_s32(temp1_hi, DCT_CONST_BITS));
    step1_24 = vcombine_s16(vrshrn_n_s32(temp2_lo, DCT_CONST_BITS), vrshrn_n_s32(temp2_hi, DCT_CONST_BITS));
    step1_28 = step2_28;
    step1_29 = step2_29;
    step1_30 = step2_30;
    step1_31 = step2_31;
    output[0] = vaddq_s16(step1_0, step1_31);
    output[1] = vaddq_s16(step1_1, step1_30);
    output[2] = vaddq_s16(step1_2, step1_29);
    output[3] = vaddq_s16(step1_3, step1_28);
    output[4] = vaddq_s16(step1_4, step1_27);
    output[5] = vaddq_s16(step1_5, step1_26);
    output[6] = vaddq_s16(step1_6, step1_25);
    output[7] = vaddq_s16(step1_7, step1_24);
    output[8] = vaddq_s16(step1_8, step1_23);
    output[9] = vaddq_s16(step1_9, step1_22);
    output[10] = vaddq_s16(step1_10, step1_21);
    output[11] = vaddq_s16(step1_11, step1_20);
    output[12] = vaddq_s16(step1_12, step1_19);
    output[13] = vaddq_s16(step1_13, step1_18);
    output[14] = vaddq_s16(step1_14, step1_17);
    output[15] = vaddq_s16(step1_15, step1_16);
    output[16] = vsubq_s16(step1_15, step1_16);
    output[17] = vsubq_s16(step1_14, step1_17);
    output[18] = vsubq_s16(step1_13, step1_18);
    output[19] = vsubq_s16(step1_12, step1_19);
    output[20] = vsubq_s16(step1_11, step1_20);
    output[21] = vsubq_s16(step1_10, step1_21);
    output[22] = vsubq_s16(step1_9, step1_22);
    output[23] = vsubq_s16(step1_8, step1_23);
    output[24] = vsubq_s16(step1_7, step1_24);
    output[25] = vsubq_s16(step1_6, step1_25);
    output[26] = vsubq_s16(step1_5, step1_26);
    output[27] = vsubq_s16(step1_4, step1_27);
    output[28] = vsubq_s16(step1_3, step1_28);
    output[29] = vsubq_s16(step1_2, step1_29);
    output[30] = vsubq_s16(step1_1, step1_30);
    output[31] = vsubq_s16(step1_0, step1_31);
}

static void iadst8_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t input_7, x0, input_0, x1, input_5, x2, input_2, x3, input_3, x4, input_4, x5, input_1, x6, input_6,
        x7, s0, s1, s2, s3, s4, s5, s6, s7, output_0, output_1, output_2, output_3, output_4, output_5, output_6,
        output_7;
    input_0 = in[0];
    input_1 = in[1];
    input_2 = in[2];
    input_3 = in[3];
    input_4 = in[4];
    input_5 = in[5];
    input_6 = in[6];
    input_7 = in[7];
    x0 = input_7;
    x1 = input_0;
    x2 = input_5;
    x3 = input_2;
    x4 = input_3;
    x5 = input_4;
    x6 = input_1;
    x7 = input_6;
    s0 = vmlaq_n_s32(vmulq_n_s32(x0, cospi_2_64), x1, cospi_30_64);
    s1 = vmlsq_n_s32(vmulq_n_s32(x0, cospi_30_64), x1, cospi_2_64);
    s2 = vmlaq_n_s32(vmulq_n_s32(x2, cospi_10_64), x3, cospi_22_64);
    s3 = vmlsq_n_s32(vmulq_n_s32(x2, cospi_22_64), x3, cospi_10_64);
    s4 = vmlaq_n_s32(vmulq_n_s32(x4, cospi_18_64), x5, cospi_14_64);
    s5 = vmlsq_n_s32(vmulq_n_s32(x4, cospi_14_64), x5, cospi_18_64);
    s6 = vmlaq_n_s32(vmulq_n_s32(x6, cospi_26_64), x7, cospi_6_64);
    s7 = vmlsq_n_s32(vmulq_n_s32(x6, cospi_6_64), x7, cospi_26_64);
    x0 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s0), s4), DCT_CONST_BITS);
    x1 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s1), s5), DCT_CONST_BITS);
    x2 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s2), s6), DCT_CONST_BITS);
    x3 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s3), s7), DCT_CONST_BITS);
    x4 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s0), s4), DCT_CONST_BITS);
    x5 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s1), s5), DCT_CONST_BITS);
    x6 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s2), s6), DCT_CONST_BITS);
    x7 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s3), s7), DCT_CONST_BITS);
    s0 = x0;
    s1 = x1;
    s2 = x2;
    s3 = x3;
    s4 = vmlaq_n_s32(vmulq_n_s32(x4, cospi_8_64), x5, cospi_24_64);
    s5 = vmlsq_n_s32(vmulq_n_s32(x4, cospi_24_64), x5, cospi_8_64);
    s6 = vmlaq_n_s32(vmulq_n_s32(x6, -cospi_24_64), x7, cospi_8_64);
    s7 = vmlaq_n_s32(vmulq_n_s32(x6, cospi_8_64), x7, cospi_24_64);
    x0 = vaddq_s32(s0, s2);
    x1 = vaddq_s32(s1, s3);
    x2 = vsubq_s32(s0, s2);
    x3 = vsubq_s32(s1, s3);
    x4 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s4), s6), DCT_CONST_BITS);
    x5 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s5), s7), DCT_CONST_BITS);
    x6 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s4), s6), DCT_CONST_BITS);
    x7 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s5), s7), DCT_CONST_BITS);
    s2 = vmulq_n_s32(vaddq_s32(x2, x3), cospi_16_64);
    s3 = vmulq_n_s32(vsubq_s32(x2, x3), cospi_16_64);
    s6 = vmulq_n_s32(vaddq_s32(x6, x7), cospi_16_64);
    s7 = vmulq_n_s32(vsubq_s32(x6, x7), cospi_16_64);
    x2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s2), DCT_CONST_BITS);
    x3 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s3), DCT_CONST_BITS);
    x6 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s6), DCT_CONST_BITS);
    x7 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s7), DCT_CONST_BITS);
    output_0 = wrap16_s32(x0);
    output_1 = wrap16_s32(vnegq_s32(x4));
    output_2 = wrap16_s32(x6);
    output_3 = wrap16_s32(vnegq_s32(x2));
    output_4 = wrap16_s32(x3);
    output_5 = wrap16_s32(vnegq_s32(x7));
    output_6 = wrap16_s32(x5);
    output_7 = wrap16_s32(vnegq_s32(x1));
    out[0] = output_0;
    out[1] = output_1;
    out[2] = output_2;
    out[3] = output_3;
    out[4] = output_4;
    out[5] = output_5;
    out[6] = output_6;
    out[7] = output_7;
}

static void iadst16_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t input_15, x0, input_0, x1, input_13, x2, input_2, x3, input_11, x4, input_4, x5, input_9, x6, input_6,
        x7, input_7, x8, input_8, x9, input_5, x10, input_10, x11, input_3, x12, input_12, x13, input_1, x14,
        input_14, x15, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, output_0, output_1,
        output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9, output_10, output_11,
        output_12, output_13, output_14, output_15;
    input_0 = in[0];
    input_1 = in[1];
    input_2 = in[2];
    input_3 = in[3];
    input_4 = in[4];
    input_5 = in[5];
    input_6 = in[6];
    input_7 = in[7];
    input_8 = in[8];
    input_9 = in[9];
    input_10 = in[10];
    input_11 = in[11];
    input_12 = in[12];
    input_13 = in[13];
    input_14 = in[14];
    input_15 = in[15];
    x0 = input_15;
    x1 = input_0;
    x2 = input_13;
    x3 = input_2;
    x4 = input_11;
    x5 = input_4;
    x6 = input_9;
    x7 = input_6;
    x8 = input_7;
    x9 = input_8;
    x10 = input_5;
    x11 = input_10;
    x12 = input_3;
    x13 = input_12;
    x14 = input_1;
    x15 = input_14;
    s0 = vmlaq_n_s32(vmulq_n_s32(x0, cospi_1_64), x1, cospi_31_64);
    s1 = vmlsq_n_s32(vmulq_n_s32(x0, cospi_31_64), x1, cospi_1_64);
    s2 = vmlaq_n_s32(vmulq_n_s32(x2, cospi_5_64), x3, cospi_27_64);
    s3 = vmlsq_n_s32(vmulq_n_s32(x2, cospi_27_64), x3, cospi_5_64);
    s4 = vmlaq_n_s32(vmulq_n_s32(x4, cospi_9_64), x5, cospi_23_64);
    s5 = vmlsq_n_s32(vmulq_n_s32(x4, cospi_23_64), x5, cospi_9_64);
    s6 = vmlaq_n_s32(vmulq_n_s32(x6, cospi_13_64), x7, cospi_19_64);
    s7 = vmlsq_n_s32(vmulq_n_s32(x6, cospi_19_64), x7, cospi_13_64);
    s8 = vmlaq_n_s32(vmulq_n_s32(x8, cospi_17_64), x9, cospi_15_64);
    s9 = vmlsq_n_s32(vmulq_n_s32(x8, cospi_15_64), x9, cospi_17_64);
    s10 = vmlaq_n_s32(vmulq_n_s32(x10, cospi_21_64), x11, cospi_11_64);
    s11 = vmlsq_n_s32(vmulq_n_s32(x10, cospi_11_64), x11, cospi_21_64);
    s12 = vmlaq_n_s32(vmulq_n_s32(x12, cospi_25_64), x13, cospi_7_64);
    s13 = vmlsq_n_s32(vmulq_n_s32(x12, cospi_7_64), x13, cospi_25_64);
    s14 = vmlaq_n_s32(vmulq_n_s32(x14, cospi_29_64), x15, cospi_3_64);
    s15 = vmlsq_n_s32(vmulq_n_s32(x14, cospi_3_64), x15, cospi_29_64);
    x0 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s0), s8), DCT_CONST_BITS);
    x1 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s1), s9), DCT_CONST_BITS);
    x2 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s2), s10), DCT_CONST_BITS);
    x3 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s3), s11), DCT_CONST_BITS);
    x4 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s4), s12), DCT_CONST_BITS);
    x5 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s5), s13), DCT_CONST_BITS);
    x6 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s6), s14), DCT_CONST_BITS);
    x7 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s7), s15), DCT_CONST_BITS);
    x8 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s0), s8), DCT_CONST_BITS);
    x9 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s1), s9), DCT_CONST_BITS);
    x10 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s2), s10), DCT_CONST_BITS);
    x11 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s3), s11), DCT_CONST_BITS);
    x12 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s4), s12), DCT_CONST_BITS);
    x13 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s5), s13), DCT_CONST_BITS);
    x14 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s6), s14), DCT_CONST_BITS);
    x15 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s7), s15), DCT_CONST_BITS);
    s0 = x0;
    s1 = x1;
    s2 = x2;
    s3 = x3;
    s4 = x4;
    s5 = x5;
    s6 = x6;
    s7 = x7;
    s8 = vmlaq_n_s32(vmulq_n_s32(x8, cospi_4_64), x9, cospi_28_64);
    s9 = vmlsq_n_s32(vmulq_n_s32(x8, cospi_28_64), x9, cospi_4_64);
    s10 = vmlaq_n_s32(vmulq_n_s32(x10, cospi_20_64), x11, cospi_12_64);
    s11 = vmlsq_n_s32(vmulq_n_s32(x10, cospi_12_64), x11, cospi_20_64);
    s12 = vmlaq_n_s32(vmulq_n_s32(x12, -cospi_28_64), x13, cospi_4_64);
    s13 = vmlaq_n_s32(vmulq_n_s32(x12, cospi_4_64), x13, cospi_28_64);
    s14 = vmlaq_n_s32(vmulq_n_s32(x14, -cospi_12_64), x15, cospi_20_64);
    s15 = vmlaq_n_s32(vmulq_n_s32(x14, cospi_20_64), x15, cospi_12_64);
    x0 = vaddq_s32(s0, s4);
    x1 = vaddq_s32(s1, s5);
    x2 = vaddq_s32(s2, s6);
    x3 = vaddq_s32(s3, s7);
    x4 = vsubq_s32(s0, s4);
    x5 = vsubq_s32(s1, s5);
    x6 = vsubq_s32(s2, s6);
    x7 = vsubq_s32(s3, s7);
    x8 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s8), s12), DCT_CONST_BITS);
    x9 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s9), s13), DCT_CONST_BITS);
    x10 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s10), s14), DCT_CONST_BITS);
    x11 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s11), s15), DCT_CONST_BITS);
    x12 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s8), s12), DCT_CONST_BITS);
    x13 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s9), s13), DCT_CONST_BITS);
    x14 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s10), s14), DCT_CONST_BITS);
    x15 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s11), s15), DCT_CONST_BITS);
    s0 = x0;
    s1 = x1;
    s2 = x2;
    s3 = x3;
    s4 = vmlaq_n_s32(vmulq_n_s32(x4, cospi_8_64), x5, cospi_24_64);
    s5 = vmlsq_n_s32(vmulq_n_s32(x4, cospi_24_64), x5, cospi_8_64);
    s6 = vmlaq_n_s32(vmulq_n_s32(x6, -cospi_24_64), x7, cospi_8_64);
    s7 = vmlaq_n_s32(vmulq_n_s32(x6, cospi_8_64), x7, cospi_24_64);
    s8 = x8;
    s9 = x9;
    s10 = x10;
    s11 = x11;
    s12 = vmlaq_n_s32(vmulq_n_s32(x12, cospi_8_64), x13, cospi_24_64);
    s13 = vmlsq_n_s32(vmulq_n_s32(x12, cospi_24_64), x13, cospi_8_64);
    s14 = vmlaq_n_s32(vmulq_n_s32(x14, -cospi_24_64), x15, cospi_8_64);
    s15 = vmlaq_n_s32(vmulq_n_s32(x14, cospi_8_64), x15, cospi_24_64);
    x0 = vaddq_s32(s0, s2);
    x1 = vaddq_s32(s1, s3);
    x2 = vsubq_s32(s0, s2);
    x3 = vsubq_s32(s1, s3);
    x4 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s4), s6), DCT_CONST_BITS);
    x5 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s5), s7), DCT_CONST_BITS);
    x6 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s4), s6), DCT_CONST_BITS);
    x7 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s5), s7), DCT_CONST_BITS);
    x8 = vaddq_s32(s8, s10);
    x9 = vaddq_s32(s9, s11);
    x10 = vsubq_s32(s8, s10);
    x11 = vsubq_s32(s9, s11);
    x12 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s12), s14), DCT_CONST_BITS);
    x13 = vshrq_n_s32(vaddq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s13), s15), DCT_CONST_BITS);
    x14 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s12), s14), DCT_CONST_BITS);
    x15 = vshrq_n_s32(vsubq_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s13), s15), DCT_CONST_BITS);
    s2 = vmulq_n_s32(vaddq_s32(x2, x3), -cospi_16_64);
    s3 = vmulq_n_s32(vsubq_s32(x2, x3), cospi_16_64);
    s6 = vmulq_n_s32(vaddq_s32(x6, x7), cospi_16_64);
    s7 = vmulq_n_s32(vsubq_s32(x7, x6), cospi_16_64);
    s10 = vmulq_n_s32(vaddq_s32(x10, x11), cospi_16_64);
    s11 = vmulq_n_s32(vsubq_s32(x11, x10), cospi_16_64);
    s14 = vmulq_n_s32(vaddq_s32(x14, x15), -cospi_16_64);
    s15 = vmulq_n_s32(vsubq_s32(x14, x15), cospi_16_64);
    x2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s2), DCT_CONST_BITS);
    x3 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s3), DCT_CONST_BITS);
    x6 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s6), DCT_CONST_BITS);
    x7 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s7), DCT_CONST_BITS);
    x10 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s10), DCT_CONST_BITS);
    x11 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s11), DCT_CONST_BITS);
    x14 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s14), DCT_CONST_BITS);
    x15 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), s15), DCT_CONST_BITS);
    output_0 = wrap16_s32(x0);
    output_1 = wrap16_s32(vnegq_s32(x8));
    output_2 = wrap16_s32(x12);
    output_3 = wrap16_s32(vnegq_s32(x4));
    output_4 = wrap16_s32(x6);
    output_5 = wrap16_s32(x14);
    output_6 = wrap16_s32(x10);
    output_7 = wrap16_s32(x2);
    output_8 = wrap16_s32(x3);
    output_9 = wrap16_s32(x11);
    output_10 = wrap16_s32(x15);
    output_11 = wrap16_s32(x7);
    output_12 = wrap16_s32(x5);
    output_13 = wrap16_s32(vnegq_s32(x13));
    output_14 = wrap16_s32(x9);
    output_15 = wrap16_s32(vnegq_s32(x1));
    out[0] = output_0;
    out[1] = output_1;
    out[2] = output_2;
    out[3] = output_3;
    out[4] = output_4;
    out[5] = output_5;
    out[6] = output_6;
    out[7] = output_7;
    out[8] = output_8;
    out[9] = output_9;
    out[10] = output_10;
    out[11] = output_11;
    out[12] = output_12;
    out[13] = output_13;
    out[14] = output_14;
    out[15] = output_15;
}


/*******************************************
 * 2-D drivers
 *******************************************/
ALWAYS_INLINE void transpose8x8_s16(int16x8_t *v) {
    const int16x8_t a0 = vtrn1q_s16(v[0], v[1]), a1 = vtrn2q_s16(v[0], v[1]);
    const int16x8_t a2 = vtrn1q_s16(v[2], v[3]), a3 = vtrn2q_s16(v[2], v[3]);
    const int16x8_t a4 = vtrn1q_s16(v[4], v[5]), a5 = vtrn2q_s16(v[4], v[5]);
    const int16x8_t a6 = vtrn1q_s16(v[6], v[7]), a7 = vtrn2q_s16(v[6], v[7]);
    const int32x4_t b0 = vtrn1q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a2));
    const int32x4_t b1 = vtrn1q_s32(vreinterpretq_s32_s16(a1), vreinterpretq_s32_s16(a3));
    const int32x4_t b2 = vtrn2q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a2));
    const int32x4_t b3 = vtrn2q_s32(vreinterpretq_s32_s16(a1), vreinterpretq_s32_s16(a3));
    const int32x4_t b4 = vtrn1q_s32(vreinterpretq_s32_s16(a4), vreinterpretq_s32_s16(a6));
    const int32x4_t b5 = vtrn1q_s32(vreinterpretq_s32_s16(a5), vreinterpretq_s32_s16(a7));
    const int32x4_t b6 = vtrn2q_s32(vreinterpretq_s32_s16(a4), vreinterpretq_s32_s16(a6));
    const int32x4_t b7 = vtrn2q_s32(vreinterpretq_s32_s16(a5), vreinterpretq_s32_s16(a7));
    v[0] = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(b0), vreinterpretq_s64_s32(b4)));
    v[1] = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(b1), vreinterpretq_s64_s32(b5)));
    v[2] = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(b2), vreinterpretq_s64_s32(b6)));
    v[3] = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(b3), vreinterpretq_s64_s32(b7)));
    v[4] = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(b0), vreinterpretq_s64_s32(b4)));
    v[5] = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(b1), vreinterpretq_s64_s32(b5)));
    v[6] = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(b2), vreinterpretq_s64_s32(b6)));
    v[7] = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(b3), vreinterpretq_s64_s32(b7)));
}

// v[k] (k < n) = lanes l: src[l * stride + k], for 8 lines; lines >= rows are zero
ALWAYS_INLINE void load_transposed(const int16_t *src, int stride, int n, int rows, int16x8_t *v) {
    for (int k = 0; k < n; k += 8) {
        for (int l = 0; l < 8; l++) v[k + l] = l < rows ? vld1q_s16(src + l * stride + k) : vdupq_n_s16(0);
        transpose8x8_s16(v + k);
    }
}

// dest row j, 8 pixels = clip(dest + ROUND_POWER_OF_TWO(v[j], shift))
ALWAYS_INLINE void add_store(uint8_t *dest, int stride, const int16x8_t *v, int n, int shift) {
    for (int j = 0; j < n; j++) {
        const int16x8_t r = shift == 5 ? vrshrq_n_s16(v[j], 5) : vrshrq_n_s16(v[j], 6);
        const int16x8_t s = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r), vld1_u8(dest + j * stride)));
        vst1_u8(dest + j * stride, vqmovun_s16(s));
    }
}

typedef void (*Idct1dX8)(const int16x8_t *in, int16x8_t *out);

// Rows (only the first 'rows' input rows are used, the others count as zero, as in the C _NN_add versions),
// then columns. buf[k * n + r] = row-pass output k of row r.
ALWAYS_INLINE void inv_txfm_2d(const tran_low_t *input, uint8_t *dest, int stride, int n, int rows, Idct1dX8 row_fn,
                               Idct1dX8 col_fn, int shift) {
    DECLARE_ALIGNED(16, int16_t, buf[32 * 32]);
    int16x8_t in[32], out[32];
    for (int r = 0; r < n; r += 8) {
        if (r >= rows) {
            for (int k = 0; k < n; k++) vst1q_s16(buf + k * n + r, vdupq_n_s16(0));
            continue;
        }
        load_transposed(input + r * n, n, n, rows - r, in);
        row_fn(in, out);
        for (int k = 0; k < n; k++) vst1q_s16(buf + k * n + r, out[k]);
    }
    for (int c = 0; c < n; c += 8) {
        load_transposed(buf + c * n, n, n, 8, in);
        col_fn(in, out);
        add_store(dest + c, stride, out, n, shift);
    }
}

// DC only: every pixel gets the same value
ALWAYS_INLINE void dc_only_add(const tran_low_t *input, uint8_t *dest, int stride, int n, int shift) {
    tran_low_t out = (int16_t)ROUND_POWER_OF_TWO((int16_t)input[0] * cospi_16_64, DCT_CONST_BITS);
    out            = (int16_t)ROUND_POWER_OF_TWO(out * cospi_16_64, DCT_CONST_BITS);
    const int16x8_t a1 = vdupq_n_s16((int16_t)ROUND_POWER_OF_TWO(out, shift));
    for (int j = 0; j < n; j++, dest += stride) {
        for (int i = 0; i < n; i += 8) {
            const int16x8_t s = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(a1), vld1_u8(dest + i)));
            vst1_u8(dest + i, vqmovun_s16(s));
        }
    }
}

void eb_vp9_idct8x8_64_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 8, 8, idct8_x8, idct8_x8, 5);
}
void eb_vp9_idct8x8_12_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 8, 4, idct8_x8, idct8_x8, 5);
}
void eb_vp9_idct8x8_1_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    dc_only_add(input, dest, stride, 8, 5);
}
void eb_vp9_idct16x16_256_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 16, 16, idct16_x8, idct16_x8, 6);
}
void eb_vp9_idct16x16_38_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 16, 8, idct16_x8, idct16_x8, 6);
}
void eb_vp9_idct16x16_10_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 16, 4, idct16_x8, idct16_x8, 6);
}
void eb_vp9_idct16x16_1_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    dc_only_add(input, dest, stride, 16, 6);
}
void eb_vp9_idct32x32_1024_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 32, 32, idct32_x8, idct32_x8, 6);
}
void eb_vp9_idct32x32_135_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 32, 16, idct32_x8, idct32_x8, 6);
}
void eb_vp9_idct32x32_34_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    inv_txfm_2d(input, dest, stride, 32, 8, idct32_x8, idct32_x8, 6);
}
void eb_vp9_idct32x32_1_add_neon(const tran_low_t *input, uint8_t *dest, int stride) {
    dc_only_add(input, dest, stride, 32, 6);
}

// iadst on 8 lanes as two int32x4 halves
#define IADST_X8(n)                                                                                \
    static void iadst##n##_x8(const int16x8_t *in, int16x8_t *out) {                               \
        int32x4_t lo[n], hi[n], olo[n], ohi[n];                                                    \
        for (int k = 0; k < n; k++) {                                                              \
            lo[k] = vmovl_s16(vget_low_s16(in[k]));                                                \
            hi[k] = vmovl_high_s16(in[k]);                                                         \
        }                                                                                          \
        iadst##n##_x4(lo, olo);                                                                    \
        iadst##n##_x4(hi, ohi);                                                                    \
        for (int k = 0; k < n; k++) out[k] = vcombine_s16(vmovn_s32(olo[k]), vmovn_s32(ohi[k])); \
    }
IADST_X8(8)
IADST_X8(16)

void eb_vp9_iht8x8_64_add_neon(const tran_low_t *input, uint8_t *dest, int stride, int tx_type) {
    // tx_type: 0 DCT_DCT, 1 ADST_DCT (ADST columns), 2 DCT_ADST (ADST rows), 3 ADST_ADST
    inv_txfm_2d(input, dest, stride, 8, 8, (tx_type & 2) ? iadst8_x8 : idct8_x8, (tx_type & 1) ? iadst8_x8 : idct8_x8, 5);
}

void eb_vp9_iht16x16_256_add_neon(const tran_low_t *input, uint8_t *dest, int stride, int tx_type) {
    inv_txfm_2d(input, dest, stride, 16, 16, (tx_type & 2) ? iadst16_x8 : idct16_x8,
                (tx_type & 1) ? iadst16_x8 : idct16_x8, 6);
}
