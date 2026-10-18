/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON forward transforms, bit-exact with fwd_txfm.c / vp9_dct.c.
// The 1-D transforms are direct translations of the C code to int32x4 lanes (4 columns or 4 rows at a time):
// tran_high_t is int32, and wrapping 32-bit vector adds / multiplies give the same results as the C code.
// Rounding shifts (x + 8192) >> 14 fold the 8192 into the multiply-accumulate chain. Casts to tran_low_t wrap
// to 16 bits as in C.

#include "EbNeonCommon.h"
#include "vpx_dsp_rtcd.h"
#include "vp9_rtcd.h"
#include "txfm_common.h"
#include "mem.h"

// (x + 1 + (x < 0)) >> 2
ALWAYS_INLINE int32x4_t half_round_shift_s32(const int32x4_t x) {
    return vshrq_n_s32(vsubq_s32(vaddq_s32(x, vdupq_n_s32(1)), vreinterpretq_s32_u32(vcltzq_s32(x))), 2);
}

// (x + 1 + (x > 0)) >> 2
ALWAYS_INLINE int32x4_t half_round_shift_pos_s32(const int32x4_t x) {
    return vshrq_n_s32(vsubq_s32(vaddq_s32(x, vdupq_n_s32(1)), vreinterpretq_s32_u32(vcgtzq_s32(x))), 2);
}

// (int32_t)(tran_low_t)x
ALWAYS_INLINE int32x4_t wrap16_s32(const int32x4_t x) { return vmovl_s16(vmovn_s32(x)); }

ALWAYS_INLINE void transpose4x4_s32(int32x4_t *r0, int32x4_t *r1, int32x4_t *r2, int32x4_t *r3) {
    const int32x4_t a0 = vtrn1q_s32(*r0, *r1), a1 = vtrn2q_s32(*r0, *r1);
    const int32x4_t a2 = vtrn1q_s32(*r2, *r3), a3 = vtrn2q_s32(*r2, *r3);
    *r0 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a2)));
    *r1 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(a1), vreinterpretq_s64_s32(a3)));
    *r2 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a2)));
    *r3 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(a1), vreinterpretq_s64_s32(a3)));
}

// Stores n (multiple of 4) vectors v[k] (lanes = 4 consecutive lines l0..l0+3) transposed: dst[(l0 + l) * stride + k] = v[k][l]
ALWAYS_INLINE void store_transposed_s32(int32_t *dst, int stride, const int32x4_t *v, int n) {
    for (int k = 0; k < n; k += 4) {
        int32x4_t a = v[k], b = v[k + 1], c = v[k + 2], d = v[k + 3];
        transpose4x4_s32(&a, &b, &c, &d);
        vst1q_s32(dst + 0 * stride + k, a);
        vst1q_s32(dst + 1 * stride + k, b);
        vst1q_s32(dst + 2 * stride + k, c);
        vst1q_s32(dst + 3 * stride + k, d);
    }
}

// Same, wrapped to tran_low_t
ALWAYS_INLINE void store_transposed_s16(tran_low_t *dst, int stride, const int32x4_t *v, int n) {
    for (int k = 0; k < n; k += 4) {
        int32x4_t a = v[k], b = v[k + 1], c = v[k + 2], d = v[k + 3];
        transpose4x4_s32(&a, &b, &c, &d);
        vst1_s16(dst + 0 * stride + k, vmovn_s32(a));
        vst1_s16(dst + 1 * stride + k, vmovn_s32(b));
        vst1_s16(dst + 2 * stride + k, vmovn_s32(c));
        vst1_s16(dst + 3 * stride + k, vmovn_s32(d));
    }
}

/*******************************************
 * 1-D transforms (generated from the C code)
 *******************************************/
static void fdct32_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t input_0, input_31, step_0, input_1, input_30, step_1, input_2, input_29, step_2, input_3, input_28,
        step_3, input_4, input_27, step_4, input_5, input_26, step_5, input_6, input_25, step_6, input_7, input_24,
        step_7, input_8, input_23, step_8, input_9, input_22, step_9, input_10, input_21, step_10, input_11,
        input_20, step_11, input_12, input_19, step_12, input_13, input_18, step_13, input_14, input_17, step_14,
        input_15, input_16, step_15, step_16, step_17, step_18, step_19, step_20, step_21, step_22, step_23,
        step_24, step_25, step_26, step_27, step_28, step_29, step_30, step_31, output_0, output_1, output_2,
        output_3, output_4, output_5, output_6, output_7, output_8, output_9, output_10, output_11, output_12,
        output_13, output_14, output_15, output_16, output_17, output_18, output_19, output_20, output_21,
        output_22, output_23, output_24, output_25, output_26, output_27, output_28, output_29, output_30,
        output_31;
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
    input_16 = in[16];
    input_17 = in[17];
    input_18 = in[18];
    input_19 = in[19];
    input_20 = in[20];
    input_21 = in[21];
    input_22 = in[22];
    input_23 = in[23];
    input_24 = in[24];
    input_25 = in[25];
    input_26 = in[26];
    input_27 = in[27];
    input_28 = in[28];
    input_29 = in[29];
    input_30 = in[30];
    input_31 = in[31];
    step_0 = vaddq_s32(input_0, input_31);
    step_1 = vaddq_s32(input_1, input_30);
    step_2 = vaddq_s32(input_2, input_29);
    step_3 = vaddq_s32(input_3, input_28);
    step_4 = vaddq_s32(input_4, input_27);
    step_5 = vaddq_s32(input_5, input_26);
    step_6 = vaddq_s32(input_6, input_25);
    step_7 = vaddq_s32(input_7, input_24);
    step_8 = vaddq_s32(input_8, input_23);
    step_9 = vaddq_s32(input_9, input_22);
    step_10 = vaddq_s32(input_10, input_21);
    step_11 = vaddq_s32(input_11, input_20);
    step_12 = vaddq_s32(input_12, input_19);
    step_13 = vaddq_s32(input_13, input_18);
    step_14 = vaddq_s32(input_14, input_17);
    step_15 = vaddq_s32(input_15, input_16);
    step_16 = vsubq_s32(input_15, input_16);
    step_17 = vsubq_s32(input_14, input_17);
    step_18 = vsubq_s32(input_13, input_18);
    step_19 = vsubq_s32(input_12, input_19);
    step_20 = vsubq_s32(input_11, input_20);
    step_21 = vsubq_s32(input_10, input_21);
    step_22 = vsubq_s32(input_9, input_22);
    step_23 = vsubq_s32(input_8, input_23);
    step_24 = vsubq_s32(input_7, input_24);
    step_25 = vsubq_s32(input_6, input_25);
    step_26 = vsubq_s32(input_5, input_26);
    step_27 = vsubq_s32(input_4, input_27);
    step_28 = vsubq_s32(input_3, input_28);
    step_29 = vsubq_s32(input_2, input_29);
    step_30 = vsubq_s32(input_1, input_30);
    step_31 = vsubq_s32(input_0, input_31);
    output_0 = vaddq_s32(step_0, step_15);
    output_1 = vaddq_s32(step_1, step_14);
    output_2 = vaddq_s32(step_2, step_13);
    output_3 = vaddq_s32(step_3, step_12);
    output_4 = vaddq_s32(step_4, step_11);
    output_5 = vaddq_s32(step_5, step_10);
    output_6 = vaddq_s32(step_6, step_9);
    output_7 = vaddq_s32(step_7, step_8);
    output_8 = vsubq_s32(step_7, step_8);
    output_9 = vsubq_s32(step_6, step_9);
    output_10 = vsubq_s32(step_5, step_10);
    output_11 = vsubq_s32(step_4, step_11);
    output_12 = vsubq_s32(step_3, step_12);
    output_13 = vsubq_s32(step_2, step_13);
    output_14 = vsubq_s32(step_1, step_14);
    output_15 = vsubq_s32(step_0, step_15);
    output_16 = step_16;
    output_17 = step_17;
    output_18 = step_18;
    output_19 = step_19;
    output_20 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_27, step_20), cospi_16_64), DCT_CONST_BITS);
    output_21 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_26, step_21), cospi_16_64), DCT_CONST_BITS);
    output_22 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_25, step_22), cospi_16_64), DCT_CONST_BITS);
    output_23 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_24, step_23), cospi_16_64), DCT_CONST_BITS);
    output_24 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_24, step_23), cospi_16_64), DCT_CONST_BITS);
    output_25 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_25, step_22), cospi_16_64), DCT_CONST_BITS);
    output_26 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_26, step_21), cospi_16_64), DCT_CONST_BITS);
    output_27 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_27, step_20), cospi_16_64), DCT_CONST_BITS);
    output_28 = step_28;
    output_29 = step_29;
    output_30 = step_30;
    output_31 = step_31;
    step_0 = vaddq_s32(output_0, output_7);
    step_1 = vaddq_s32(output_1, output_6);
    step_2 = vaddq_s32(output_2, output_5);
    step_3 = vaddq_s32(output_3, output_4);
    step_4 = vsubq_s32(output_3, output_4);
    step_5 = vsubq_s32(output_2, output_5);
    step_6 = vsubq_s32(output_1, output_6);
    step_7 = vsubq_s32(output_0, output_7);
    step_8 = output_8;
    step_9 = output_9;
    step_10 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(output_13, output_10), cospi_16_64), DCT_CONST_BITS);
    step_11 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(output_12, output_11), cospi_16_64), DCT_CONST_BITS);
    step_12 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(output_12, output_11), cospi_16_64), DCT_CONST_BITS);
    step_13 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(output_13, output_10), cospi_16_64), DCT_CONST_BITS);
    step_14 = output_14;
    step_15 = output_15;
    step_16 = vaddq_s32(output_16, output_23);
    step_17 = vaddq_s32(output_17, output_22);
    step_18 = vaddq_s32(output_18, output_21);
    step_19 = vaddq_s32(output_19, output_20);
    step_20 = vsubq_s32(output_19, output_20);
    step_21 = vsubq_s32(output_18, output_21);
    step_22 = vsubq_s32(output_17, output_22);
    step_23 = vsubq_s32(output_16, output_23);
    step_24 = vsubq_s32(output_31, output_24);
    step_25 = vsubq_s32(output_30, output_25);
    step_26 = vsubq_s32(output_29, output_26);
    step_27 = vsubq_s32(output_28, output_27);
    step_28 = vaddq_s32(output_28, output_27);
    step_29 = vaddq_s32(output_29, output_26);
    step_30 = vaddq_s32(output_30, output_25);
    step_31 = vaddq_s32(output_31, output_24);
    output_0 = vaddq_s32(step_0, step_3);
    output_1 = vaddq_s32(step_1, step_2);
    output_2 = vsubq_s32(step_1, step_2);
    output_3 = vsubq_s32(step_0, step_3);
    output_4 = step_4;
    output_5 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_6, step_5), cospi_16_64), DCT_CONST_BITS);
    output_6 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_6, step_5), cospi_16_64), DCT_CONST_BITS);
    output_7 = step_7;
    output_8 = vaddq_s32(step_8, step_11);
    output_9 = vaddq_s32(step_9, step_10);
    output_10 = vsubq_s32(step_9, step_10);
    output_11 = vsubq_s32(step_8, step_11);
    output_12 = vsubq_s32(step_15, step_12);
    output_13 = vsubq_s32(step_14, step_13);
    output_14 = vaddq_s32(step_14, step_13);
    output_15 = vaddq_s32(step_15, step_12);
    output_16 = step_16;
    output_17 = step_17;
    output_18 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_18, cospi_8_64), step_29, cospi_24_64), DCT_CONST_BITS);
    output_19 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_19, cospi_8_64), step_28, cospi_24_64), DCT_CONST_BITS);
    output_20 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_20, cospi_24_64), step_27, cospi_8_64), DCT_CONST_BITS);
    output_21 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_21, cospi_24_64), step_26, cospi_8_64), DCT_CONST_BITS);
    output_22 = step_22;
    output_23 = step_23;
    output_24 = step_24;
    output_25 = step_25;
    output_26 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_26, cospi_24_64), step_21, cospi_8_64), DCT_CONST_BITS);
    output_27 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_27, cospi_24_64), step_20, cospi_8_64), DCT_CONST_BITS);
    output_28 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_28, cospi_8_64), step_19, cospi_24_64), DCT_CONST_BITS);
    output_29 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_29, cospi_8_64), step_18, cospi_24_64), DCT_CONST_BITS);
    output_30 = step_30;
    output_31 = step_31;
    step_0 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(output_0, output_1), cospi_16_64), DCT_CONST_BITS);
    step_1 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(output_0, output_1), cospi_16_64), DCT_CONST_BITS);
    step_2 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_2, cospi_24_64), output_3, cospi_8_64), DCT_CONST_BITS);
    step_3 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_3, cospi_24_64), output_2, cospi_8_64), DCT_CONST_BITS);
    step_4 = vaddq_s32(output_4, output_5);
    step_5 = vsubq_s32(output_4, output_5);
    step_6 = vsubq_s32(output_7, output_6);
    step_7 = vaddq_s32(output_7, output_6);
    step_8 = output_8;
    step_9 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_9, cospi_8_64), output_14, cospi_24_64), DCT_CONST_BITS);
    step_10 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_10, cospi_24_64), output_13, cospi_8_64), DCT_CONST_BITS);
    step_11 = output_11;
    step_12 = output_12;
    step_13 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_13, cospi_24_64), output_10, cospi_8_64), DCT_CONST_BITS);
    step_14 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_14, cospi_8_64), output_9, cospi_24_64), DCT_CONST_BITS);
    step_15 = output_15;
    step_16 = vaddq_s32(output_16, output_19);
    step_17 = vaddq_s32(output_17, output_18);
    step_18 = vsubq_s32(output_17, output_18);
    step_19 = vsubq_s32(output_16, output_19);
    step_20 = vsubq_s32(output_23, output_20);
    step_21 = vsubq_s32(output_22, output_21);
    step_22 = vaddq_s32(output_22, output_21);
    step_23 = vaddq_s32(output_23, output_20);
    step_24 = vaddq_s32(output_24, output_27);
    step_25 = vaddq_s32(output_25, output_26);
    step_26 = vsubq_s32(output_25, output_26);
    step_27 = vsubq_s32(output_24, output_27);
    step_28 = vsubq_s32(output_31, output_28);
    step_29 = vsubq_s32(output_30, output_29);
    step_30 = vaddq_s32(output_30, output_29);
    step_31 = vaddq_s32(output_31, output_28);
    output_0 = step_0;
    output_1 = step_1;
    output_2 = step_2;
    output_3 = step_3;
    output_4 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_4, cospi_28_64), step_7, cospi_4_64), DCT_CONST_BITS);
    output_5 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_5, cospi_12_64), step_6, cospi_20_64), DCT_CONST_BITS);
    output_6 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_6, cospi_12_64), step_5, cospi_20_64), DCT_CONST_BITS);
    output_7 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_7, cospi_28_64), step_4, cospi_4_64), DCT_CONST_BITS);
    output_8 = vaddq_s32(step_8, step_9);
    output_9 = vsubq_s32(step_8, step_9);
    output_10 = vsubq_s32(step_11, step_10);
    output_11 = vaddq_s32(step_11, step_10);
    output_12 = vaddq_s32(step_12, step_13);
    output_13 = vsubq_s32(step_12, step_13);
    output_14 = vsubq_s32(step_15, step_14);
    output_15 = vaddq_s32(step_15, step_14);
    output_16 = step_16;
    output_17 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_17, cospi_4_64), step_30, cospi_28_64), DCT_CONST_BITS);
    output_18 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_18, cospi_28_64), step_29, cospi_4_64), DCT_CONST_BITS);
    output_19 = step_19;
    output_20 = step_20;
    output_21 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_21, cospi_20_64), step_26, cospi_12_64), DCT_CONST_BITS);
    output_22 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_22, cospi_12_64), step_25, cospi_20_64), DCT_CONST_BITS);
    output_23 = step_23;
    output_24 = step_24;
    output_25 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_25, cospi_12_64), step_22, cospi_20_64), DCT_CONST_BITS);
    output_26 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_26, cospi_20_64), step_21, cospi_12_64), DCT_CONST_BITS);
    output_27 = step_27;
    output_28 = step_28;
    output_29 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_29, cospi_28_64), step_18, cospi_4_64), DCT_CONST_BITS);
    output_30 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_30, cospi_4_64), step_17, cospi_28_64), DCT_CONST_BITS);
    output_31 = step_31;
    step_0 = output_0;
    step_1 = output_1;
    step_2 = output_2;
    step_3 = output_3;
    step_4 = output_4;
    step_5 = output_5;
    step_6 = output_6;
    step_7 = output_7;
    step_8 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_8, cospi_30_64), output_15, cospi_2_64), DCT_CONST_BITS);
    step_9 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_9, cospi_14_64), output_14, cospi_18_64), DCT_CONST_BITS);
    step_10 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_10, cospi_22_64), output_13, cospi_10_64), DCT_CONST_BITS);
    step_11 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_11, cospi_6_64), output_12, cospi_26_64), DCT_CONST_BITS);
    step_12 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_12, cospi_6_64), output_11, cospi_26_64), DCT_CONST_BITS);
    step_13 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_13, cospi_22_64), output_10, cospi_10_64), DCT_CONST_BITS);
    step_14 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_14, cospi_14_64), output_9, cospi_18_64), DCT_CONST_BITS);
    step_15 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_15, cospi_30_64), output_8, cospi_2_64), DCT_CONST_BITS);
    step_16 = vaddq_s32(output_16, output_17);
    step_17 = vsubq_s32(output_16, output_17);
    step_18 = vsubq_s32(output_19, output_18);
    step_19 = vaddq_s32(output_19, output_18);
    step_20 = vaddq_s32(output_20, output_21);
    step_21 = vsubq_s32(output_20, output_21);
    step_22 = vsubq_s32(output_23, output_22);
    step_23 = vaddq_s32(output_23, output_22);
    step_24 = vaddq_s32(output_24, output_25);
    step_25 = vsubq_s32(output_24, output_25);
    step_26 = vsubq_s32(output_27, output_26);
    step_27 = vaddq_s32(output_27, output_26);
    step_28 = vaddq_s32(output_28, output_29);
    step_29 = vsubq_s32(output_28, output_29);
    step_30 = vsubq_s32(output_31, output_30);
    step_31 = vaddq_s32(output_31, output_30);
    output_0 = step_0;
    output_16 = step_1;
    output_8 = step_2;
    output_24 = step_3;
    output_4 = step_4;
    output_20 = step_5;
    output_12 = step_6;
    output_28 = step_7;
    output_2 = step_8;
    output_18 = step_9;
    output_10 = step_10;
    output_26 = step_11;
    output_6 = step_12;
    output_22 = step_13;
    output_14 = step_14;
    output_30 = step_15;
    output_1 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_16, cospi_31_64), step_31, cospi_1_64), DCT_CONST_BITS);
    output_17 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_17, cospi_15_64), step_30, cospi_17_64), DCT_CONST_BITS);
    output_9 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_18, cospi_23_64), step_29, cospi_9_64), DCT_CONST_BITS);
    output_25 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_19, cospi_7_64), step_28, cospi_25_64), DCT_CONST_BITS);
    output_5 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_20, cospi_27_64), step_27, cospi_5_64), DCT_CONST_BITS);
    output_21 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_21, cospi_11_64), step_26, cospi_21_64), DCT_CONST_BITS);
    output_13 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_22, cospi_19_64), step_25, cospi_13_64), DCT_CONST_BITS);
    output_29 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_23, cospi_3_64), step_24, cospi_29_64), DCT_CONST_BITS);
    output_3 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_24, cospi_3_64), step_23, cospi_29_64), DCT_CONST_BITS);
    output_19 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_25, cospi_19_64), step_22, cospi_13_64), DCT_CONST_BITS);
    output_11 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_26, cospi_11_64), step_21, cospi_21_64), DCT_CONST_BITS);
    output_27 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_27, cospi_27_64), step_20, cospi_5_64), DCT_CONST_BITS);
    output_7 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_28, cospi_7_64), step_19, cospi_25_64), DCT_CONST_BITS);
    output_23 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_29, cospi_23_64), step_18, cospi_9_64), DCT_CONST_BITS);
    output_15 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_30, cospi_15_64), step_17, cospi_17_64), DCT_CONST_BITS);
    output_31 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_31, cospi_31_64), step_16, cospi_1_64), DCT_CONST_BITS);
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
    out[16] = output_16;
    out[17] = output_17;
    out[18] = output_18;
    out[19] = output_19;
    out[20] = output_20;
    out[21] = output_21;
    out[22] = output_22;
    out[23] = output_23;
    out[24] = output_24;
    out[25] = output_25;
    out[26] = output_26;
    out[27] = output_27;
    out[28] = output_28;
    out[29] = output_29;
    out[30] = output_30;
    out[31] = output_31;
}

static void partial_fdct32_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t input_0, input_31, step_0, input_1, input_30, step_1, input_2, input_29, step_2, input_3, input_28,
        step_3, input_4, input_27, step_4, input_5, input_26, step_5, input_6, input_25, step_6, input_7, input_24,
        step_7, input_8, input_23, step_8, input_9, input_22, step_9, input_10, input_21, step_10, input_11,
        input_20, step_11, input_12, input_19, step_12, input_13, input_18, step_13, input_14, input_17, step_14,
        input_15, input_16, step_15, step_16, step_17, step_18, step_19, step_20, step_21, step_22, step_23,
        step_24, step_25, step_26, step_27, step_28, step_29, step_30, step_31, output_0, output_1, output_2,
        output_3, output_4, output_5, output_6, output_7, output_8, output_9, output_10, output_11, output_12,
        output_13, output_14, output_15, output_16, output_17, output_18, output_19, output_20, output_21,
        output_22, output_23, output_24, output_25, output_26, output_27, output_28, output_29, output_30,
        output_31;
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
    input_16 = in[16];
    input_17 = in[17];
    input_18 = in[18];
    input_19 = in[19];
    input_20 = in[20];
    input_21 = in[21];
    input_22 = in[22];
    input_23 = in[23];
    input_24 = in[24];
    input_25 = in[25];
    input_26 = in[26];
    input_27 = in[27];
    input_28 = in[28];
    input_29 = in[29];
    input_30 = in[30];
    input_31 = in[31];
    step_0 = vaddq_s32(input_0, input_31);
    step_1 = vaddq_s32(input_1, input_30);
    step_2 = vaddq_s32(input_2, input_29);
    step_3 = vaddq_s32(input_3, input_28);
    step_4 = vaddq_s32(input_4, input_27);
    step_5 = vaddq_s32(input_5, input_26);
    step_6 = vaddq_s32(input_6, input_25);
    step_7 = vaddq_s32(input_7, input_24);
    step_8 = vaddq_s32(input_8, input_23);
    step_9 = vaddq_s32(input_9, input_22);
    step_10 = vaddq_s32(input_10, input_21);
    step_11 = vaddq_s32(input_11, input_20);
    step_12 = vaddq_s32(input_12, input_19);
    step_13 = vaddq_s32(input_13, input_18);
    step_14 = vaddq_s32(input_14, input_17);
    step_15 = vaddq_s32(input_15, input_16);
    step_16 = vsubq_s32(input_15, input_16);
    step_17 = vsubq_s32(input_14, input_17);
    step_18 = vsubq_s32(input_13, input_18);
    step_19 = vsubq_s32(input_12, input_19);
    step_20 = vsubq_s32(input_11, input_20);
    step_21 = vsubq_s32(input_10, input_21);
    step_22 = vsubq_s32(input_9, input_22);
    step_23 = vsubq_s32(input_8, input_23);
    step_24 = vsubq_s32(input_7, input_24);
    step_25 = vsubq_s32(input_6, input_25);
    step_26 = vsubq_s32(input_5, input_26);
    step_27 = vsubq_s32(input_4, input_27);
    step_28 = vsubq_s32(input_3, input_28);
    step_29 = vsubq_s32(input_2, input_29);
    step_30 = vsubq_s32(input_1, input_30);
    step_31 = vsubq_s32(input_0, input_31);
    output_0 = vaddq_s32(step_0, step_15);
    output_1 = vaddq_s32(step_1, step_14);
    output_2 = vaddq_s32(step_2, step_13);
    output_3 = vaddq_s32(step_3, step_12);
    output_4 = vaddq_s32(step_4, step_11);
    output_5 = vaddq_s32(step_5, step_10);
    output_6 = vaddq_s32(step_6, step_9);
    output_7 = vaddq_s32(step_7, step_8);
    output_8 = vsubq_s32(step_7, step_8);
    output_9 = vsubq_s32(step_6, step_9);
    output_10 = vsubq_s32(step_5, step_10);
    output_11 = vsubq_s32(step_4, step_11);
    output_12 = vsubq_s32(step_3, step_12);
    output_13 = vsubq_s32(step_2, step_13);
    output_14 = vsubq_s32(step_1, step_14);
    output_15 = vsubq_s32(step_0, step_15);
    output_16 = step_16;
    output_17 = step_17;
    output_18 = step_18;
    output_19 = step_19;
    output_20 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_27, step_20), cospi_16_64), DCT_CONST_BITS);
    output_21 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_26, step_21), cospi_16_64), DCT_CONST_BITS);
    output_22 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_25, step_22), cospi_16_64), DCT_CONST_BITS);
    output_23 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_24, step_23), cospi_16_64), DCT_CONST_BITS);
    output_24 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_24, step_23), cospi_16_64), DCT_CONST_BITS);
    output_25 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_25, step_22), cospi_16_64), DCT_CONST_BITS);
    output_26 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_26, step_21), cospi_16_64), DCT_CONST_BITS);
    output_27 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_27, step_20), cospi_16_64), DCT_CONST_BITS);
    output_28 = step_28;
    output_29 = step_29;
    output_30 = step_30;
    output_31 = step_31;
    step_0 = vaddq_s32(output_0, output_7);
    step_1 = vaddq_s32(output_1, output_6);
    step_2 = vaddq_s32(output_2, output_5);
    step_3 = vaddq_s32(output_3, output_4);
    step_4 = vsubq_s32(output_3, output_4);
    step_5 = vsubq_s32(output_2, output_5);
    step_6 = vsubq_s32(output_1, output_6);
    step_7 = vsubq_s32(output_0, output_7);
    step_8 = output_8;
    step_9 = output_9;
    step_10 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(output_13, output_10), cospi_16_64), DCT_CONST_BITS);
    step_11 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(output_12, output_11), cospi_16_64), DCT_CONST_BITS);
    step_12 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(output_12, output_11), cospi_16_64), DCT_CONST_BITS);
    step_13 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(output_13, output_10), cospi_16_64), DCT_CONST_BITS);
    step_14 = output_14;
    step_15 = output_15;
    step_16 = vaddq_s32(output_16, output_23);
    step_17 = vaddq_s32(output_17, output_22);
    step_18 = vaddq_s32(output_18, output_21);
    step_19 = vaddq_s32(output_19, output_20);
    step_20 = vsubq_s32(output_19, output_20);
    step_21 = vsubq_s32(output_18, output_21);
    step_22 = vsubq_s32(output_17, output_22);
    step_23 = vsubq_s32(output_16, output_23);
    step_24 = vsubq_s32(output_31, output_24);
    step_25 = vsubq_s32(output_30, output_25);
    step_26 = vsubq_s32(output_29, output_26);
    step_27 = vsubq_s32(output_28, output_27);
    step_28 = vaddq_s32(output_28, output_27);
    step_29 = vaddq_s32(output_29, output_26);
    step_30 = vaddq_s32(output_30, output_25);
    step_31 = vaddq_s32(output_31, output_24);
    output_0 = vaddq_s32(step_0, step_3);
    output_1 = vaddq_s32(step_1, step_2);
    output_2 = vsubq_s32(step_1, step_2);
    output_3 = vsubq_s32(step_0, step_3);
    output_4 = step_4;
    output_5 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vsubq_s32(step_6, step_5), cospi_16_64), DCT_CONST_BITS);
    output_6 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(step_6, step_5), cospi_16_64), DCT_CONST_BITS);
    output_7 = step_7;
    output_8 = vaddq_s32(step_8, step_11);
    output_9 = vaddq_s32(step_9, step_10);
    output_10 = vsubq_s32(step_9, step_10);
    output_11 = vsubq_s32(step_8, step_11);
    output_12 = vsubq_s32(step_15, step_12);
    output_13 = vsubq_s32(step_14, step_13);
    output_14 = vaddq_s32(step_14, step_13);
    output_15 = vaddq_s32(step_15, step_12);
    output_16 = step_16;
    output_17 = step_17;
    output_18 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_18, cospi_8_64), step_29, cospi_24_64), DCT_CONST_BITS);
    output_19 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_19, cospi_8_64), step_28, cospi_24_64), DCT_CONST_BITS);
    output_20 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_20, cospi_24_64), step_27, cospi_8_64), DCT_CONST_BITS);
    output_21 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_21, cospi_24_64), step_26, cospi_8_64), DCT_CONST_BITS);
    output_22 = step_22;
    output_23 = step_23;
    output_24 = step_24;
    output_25 = step_25;
    output_26 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_26, cospi_24_64), step_21, cospi_8_64), DCT_CONST_BITS);
    output_27 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_27, cospi_24_64), step_20, cospi_8_64), DCT_CONST_BITS);
    output_28 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_28, cospi_8_64), step_19, cospi_24_64), DCT_CONST_BITS);
    output_29 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_29, cospi_8_64), step_18, cospi_24_64), DCT_CONST_BITS);
    output_30 = step_30;
    output_31 = step_31;
    step_0 = vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), vaddq_s32(output_0, output_1), cospi_16_64), DCT_CONST_BITS);
    step_2 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_2, cospi_24_64), output_3, cospi_8_64), DCT_CONST_BITS);
    step_4 = vaddq_s32(output_4, output_5);
    step_5 = vsubq_s32(output_4, output_5);
    step_6 = vsubq_s32(output_7, output_6);
    step_7 = vaddq_s32(output_7, output_6);
    step_8 = output_8;
    step_9 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_9, cospi_8_64), output_14, cospi_24_64), DCT_CONST_BITS);
    step_10 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_10, cospi_24_64), output_13, cospi_8_64), DCT_CONST_BITS);
    step_11 = output_11;
    step_12 = output_12;
    step_13 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_13, cospi_24_64), output_10, cospi_8_64), DCT_CONST_BITS);
    step_14 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_14, cospi_8_64), output_9, cospi_24_64), DCT_CONST_BITS);
    step_15 = output_15;
    step_16 = vaddq_s32(output_16, output_19);
    step_17 = vaddq_s32(output_17, output_18);
    step_18 = vsubq_s32(output_17, output_18);
    step_19 = vsubq_s32(output_16, output_19);
    step_20 = vsubq_s32(output_23, output_20);
    step_21 = vsubq_s32(output_22, output_21);
    step_22 = vaddq_s32(output_22, output_21);
    step_23 = vaddq_s32(output_23, output_20);
    step_24 = vaddq_s32(output_24, output_27);
    step_25 = vaddq_s32(output_25, output_26);
    step_26 = vsubq_s32(output_25, output_26);
    step_27 = vsubq_s32(output_24, output_27);
    step_28 = vsubq_s32(output_31, output_28);
    step_29 = vsubq_s32(output_30, output_29);
    step_30 = vaddq_s32(output_30, output_29);
    step_31 = vaddq_s32(output_31, output_28);
    output_0 = step_0;
    output_2 = step_2;
    output_4 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_4, cospi_28_64), step_7, cospi_4_64), DCT_CONST_BITS);
    output_6 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_6, cospi_12_64), step_5, cospi_20_64), DCT_CONST_BITS);
    output_8 = vaddq_s32(step_8, step_9);
    output_9 = vsubq_s32(step_8, step_9);
    output_10 = vsubq_s32(step_11, step_10);
    output_11 = vaddq_s32(step_11, step_10);
    output_12 = vaddq_s32(step_12, step_13);
    output_13 = vsubq_s32(step_12, step_13);
    output_14 = vsubq_s32(step_15, step_14);
    output_15 = vaddq_s32(step_15, step_14);
    output_16 = step_16;
    output_17 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_17, cospi_4_64), step_30, cospi_28_64), DCT_CONST_BITS);
    output_18 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_18, cospi_28_64), step_29, cospi_4_64), DCT_CONST_BITS);
    output_19 = step_19;
    output_20 = step_20;
    output_21 = vshrq_n_s32(vmlaq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_21, cospi_20_64), step_26, cospi_12_64), DCT_CONST_BITS);
    output_22 = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_22, cospi_12_64), step_25, cospi_20_64), DCT_CONST_BITS);
    output_23 = step_23;
    output_24 = step_24;
    output_25 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_25, cospi_12_64), step_22, cospi_20_64), DCT_CONST_BITS);
    output_26 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_26, cospi_20_64), step_21, cospi_12_64), DCT_CONST_BITS);
    output_27 = step_27;
    output_28 = step_28;
    output_29 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_29, cospi_28_64), step_18, cospi_4_64), DCT_CONST_BITS);
    output_30 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_30, cospi_4_64), step_17, cospi_28_64), DCT_CONST_BITS);
    output_31 = step_31;
    step_0 = output_0;
    step_2 = output_2;
    step_4 = output_4;
    step_6 = output_6;
    step_8 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_8, cospi_30_64), output_15, cospi_2_64), DCT_CONST_BITS);
    step_10 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_10, cospi_22_64), output_13, cospi_10_64), DCT_CONST_BITS);
    step_12 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_12, cospi_6_64), output_11, cospi_26_64), DCT_CONST_BITS);
    step_14 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), output_14, cospi_14_64), output_9, cospi_18_64), DCT_CONST_BITS);
    step_16 = vaddq_s32(output_16, output_17);
    step_17 = vsubq_s32(output_16, output_17);
    step_18 = vsubq_s32(output_19, output_18);
    step_19 = vaddq_s32(output_19, output_18);
    step_20 = vaddq_s32(output_20, output_21);
    step_21 = vsubq_s32(output_20, output_21);
    step_22 = vsubq_s32(output_23, output_22);
    step_23 = vaddq_s32(output_23, output_22);
    step_24 = vaddq_s32(output_24, output_25);
    step_25 = vsubq_s32(output_24, output_25);
    step_26 = vsubq_s32(output_27, output_26);
    step_27 = vaddq_s32(output_27, output_26);
    step_28 = vaddq_s32(output_28, output_29);
    step_29 = vsubq_s32(output_28, output_29);
    step_30 = vsubq_s32(output_31, output_30);
    step_31 = vaddq_s32(output_31, output_30);
    output_0 = step_0;
    output_8 = step_2;
    output_4 = step_4;
    output_12 = step_6;
    output_2 = step_8;
    output_10 = step_10;
    output_6 = step_12;
    output_14 = step_14;
    output_1 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_16, cospi_31_64), step_31, cospi_1_64), DCT_CONST_BITS);
    output_9 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_18, cospi_23_64), step_29, cospi_9_64), DCT_CONST_BITS);
    output_5 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_20, cospi_27_64), step_27, cospi_5_64), DCT_CONST_BITS);
    output_13 = vshrq_n_s32(vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_22, cospi_19_64), step_25, cospi_13_64), DCT_CONST_BITS);
    output_3 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_24, cospi_3_64), step_23, cospi_29_64), DCT_CONST_BITS);
    output_11 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_26, cospi_11_64), step_21, cospi_21_64), DCT_CONST_BITS);
    output_7 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_28, cospi_7_64), step_19, cospi_25_64), DCT_CONST_BITS);
    output_15 = vshrq_n_s32(vmlsq_n_s32(vmlaq_n_s32(vdupq_n_s32(DCT_CONST_ROUNDING), step_30, cospi_15_64), step_17, cospi_17_64), DCT_CONST_BITS);
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

static void fdct16x16_1d_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t in_high_0, in_high_7, s0, in_high_1, in_high_6, s1, in_high_2, in_high_5, s2, in_high_3, in_high_4,
        s3, s4, s5, s6, s7, x0, x1, x2, x3, t0, t1, t2, t3, out_0, out_4, out_8, out_12, out_2, out_6, out_10,
        out_14, step1_5, step1_2, temp1, step1_4, step1_3, temp2, step2_2, step2_3, step2_4, step2_5, step1_0,
        step3_0, step1_1, step3_1, step3_2, step3_3, step1_7, step3_4, step1_6, step3_5, step3_6, step3_7, step2_1,
        step2_6, out_1, out_9, out_5, out_13, out_3, out_11, out_7, out_15;
    in_high_0 = in[0];
    in_high_1 = in[1];
    in_high_2 = in[2];
    in_high_3 = in[3];
    in_high_4 = in[4];
    in_high_5 = in[5];
    in_high_6 = in[6];
    in_high_7 = in[7];
    step1_0 = in[8];
    step1_1 = in[9];
    step1_2 = in[10];
    step1_3 = in[11];
    step1_4 = in[12];
    step1_5 = in[13];
    step1_6 = in[14];
    step1_7 = in[15];
    s0 = vaddq_s32(in_high_0, in_high_7);
    s1 = vaddq_s32(in_high_1, in_high_6);
    s2 = vaddq_s32(in_high_2, in_high_5);
    s3 = vaddq_s32(in_high_3, in_high_4);
    s4 = vsubq_s32(in_high_3, in_high_4);
    s5 = vsubq_s32(in_high_2, in_high_5);
    s6 = vsubq_s32(in_high_1, in_high_6);
    s7 = vsubq_s32(in_high_0, in_high_7);
    x0 = vaddq_s32(s0, s3);
    x1 = vaddq_s32(s1, s2);
    x2 = vsubq_s32(s1, s2);
    x3 = vsubq_s32(s0, s3);
    t0 = vmulq_n_s32(vaddq_s32(x0, x1), cospi_16_64);
    t1 = vmulq_n_s32(vsubq_s32(x0, x1), cospi_16_64);
    t2 = vmlaq_n_s32(vmulq_n_s32(x3, cospi_8_64), x2, cospi_24_64);
    t3 = vmlsq_n_s32(vmulq_n_s32(x3, cospi_24_64), x2, cospi_8_64);
    out_0 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    out_4 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    out_8 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    out_12 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    t0 = vmulq_n_s32(vsubq_s32(s6, s5), cospi_16_64);
    t1 = vmulq_n_s32(vaddq_s32(s6, s5), cospi_16_64);
    t2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS);
    t3 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS);
    x0 = vaddq_s32(s4, t2);
    x1 = vsubq_s32(s4, t2);
    x2 = vsubq_s32(s7, t3);
    x3 = vaddq_s32(s7, t3);
    t0 = vmlaq_n_s32(vmulq_n_s32(x0, cospi_28_64), x3, cospi_4_64);
    t1 = vmlaq_n_s32(vmulq_n_s32(x1, cospi_12_64), x2, cospi_20_64);
    t2 = vmlsq_n_s32(vmulq_n_s32(x2, cospi_12_64), x1, cospi_20_64);
    t3 = vmlsq_n_s32(vmulq_n_s32(x3, cospi_28_64), x0, cospi_4_64);
    out_2 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    out_6 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    out_10 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    out_14 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    temp1 = vmulq_n_s32(vsubq_s32(step1_5, step1_2), cospi_16_64);
    temp2 = vmulq_n_s32(vsubq_s32(step1_4, step1_3), cospi_16_64);
    step2_2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_3 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    temp1 = vmulq_n_s32(vaddq_s32(step1_4, step1_3), cospi_16_64);
    temp2 = vmulq_n_s32(vaddq_s32(step1_5, step1_2), cospi_16_64);
    step2_4 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_5 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    step3_0 = vaddq_s32(step1_0, step2_3);
    step3_1 = vaddq_s32(step1_1, step2_2);
    step3_2 = vsubq_s32(step1_1, step2_2);
    step3_3 = vsubq_s32(step1_0, step2_3);
    step3_4 = vsubq_s32(step1_7, step2_4);
    step3_5 = vsubq_s32(step1_6, step2_5);
    step3_6 = vaddq_s32(step1_6, step2_5);
    step3_7 = vaddq_s32(step1_7, step2_4);
    temp1 = vmlaq_n_s32(vmulq_n_s32(step3_1, -cospi_8_64), step3_6, cospi_24_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step3_2, cospi_24_64), step3_5, cospi_8_64);
    step2_1 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    temp1 = vmlsq_n_s32(vmulq_n_s32(step3_2, cospi_8_64), step3_5, cospi_24_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step3_1, cospi_24_64), step3_6, cospi_8_64);
    step2_5 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_6 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    step1_0 = vaddq_s32(step3_0, step2_1);
    step1_1 = vsubq_s32(step3_0, step2_1);
    step1_2 = vaddq_s32(step3_3, step2_2);
    step1_3 = vsubq_s32(step3_3, step2_2);
    step1_4 = vsubq_s32(step3_4, step2_5);
    step1_5 = vaddq_s32(step3_4, step2_5);
    step1_6 = vsubq_s32(step3_7, step2_6);
    step1_7 = vaddq_s32(step3_7, step2_6);
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_0, cospi_30_64), step1_7, cospi_2_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_1, cospi_14_64), step1_6, cospi_18_64);
    out_1 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_9 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_2, cospi_22_64), step1_5, cospi_10_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_3, cospi_6_64), step1_4, cospi_26_64);
    out_5 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_13 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_3, -cospi_26_64), step1_4, cospi_6_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_2, -cospi_10_64), step1_5, cospi_22_64);
    out_3 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_11 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_1, -cospi_18_64), step1_6, cospi_14_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_0, -cospi_2_64), step1_7, cospi_30_64);
    out_7 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_15 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    out[0] = out_0;
    out[1] = out_1;
    out[2] = out_2;
    out[3] = out_3;
    out[4] = out_4;
    out[5] = out_5;
    out[6] = out_6;
    out[7] = out_7;
    out[8] = out_8;
    out[9] = out_9;
    out[10] = out_10;
    out[11] = out_11;
    out[12] = out_12;
    out[13] = out_13;
    out[14] = out_14;
    out[15] = out_15;
}

static void fdct16_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t inp_0, inp_15, input_0, inp_1, inp_14, input_1, inp_2, inp_13, input_2, inp_3, inp_12, input_3, inp_4,
        inp_11, input_4, inp_5, inp_10, input_5, inp_6, inp_9, input_6, inp_7, inp_8, input_7, step1_0, step1_1,
        step1_2, step1_3, step1_4, step1_5, step1_6, step1_7, s0, s1, s2, s3, s4, s5, s6, s7, x0, x1, x2, x3, t0,
        t1, t2, t3, out_0, out_4, out_8, out_12, out_2, out_6, out_10, out_14, temp1, temp2, step2_2, step2_3,
        step2_4, step2_5, step3_0, step3_1, step3_2, step3_3, step3_4, step3_5, step3_6, step3_7, step2_1, step2_6,
        out_1, out_9, out_5, out_13, out_3, out_11, out_7, out_15;
    inp_0 = in[0];
    inp_1 = in[1];
    inp_2 = in[2];
    inp_3 = in[3];
    inp_4 = in[4];
    inp_5 = in[5];
    inp_6 = in[6];
    inp_7 = in[7];
    inp_8 = in[8];
    inp_9 = in[9];
    inp_10 = in[10];
    inp_11 = in[11];
    inp_12 = in[12];
    inp_13 = in[13];
    inp_14 = in[14];
    inp_15 = in[15];
    input_0 = vaddq_s32(inp_0, inp_15);
    input_1 = vaddq_s32(inp_1, inp_14);
    input_2 = vaddq_s32(inp_2, inp_13);
    input_3 = vaddq_s32(inp_3, inp_12);
    input_4 = vaddq_s32(inp_4, inp_11);
    input_5 = vaddq_s32(inp_5, inp_10);
    input_6 = vaddq_s32(inp_6, inp_9);
    input_7 = vaddq_s32(inp_7, inp_8);
    step1_0 = vsubq_s32(inp_7, inp_8);
    step1_1 = vsubq_s32(inp_6, inp_9);
    step1_2 = vsubq_s32(inp_5, inp_10);
    step1_3 = vsubq_s32(inp_4, inp_11);
    step1_4 = vsubq_s32(inp_3, inp_12);
    step1_5 = vsubq_s32(inp_2, inp_13);
    step1_6 = vsubq_s32(inp_1, inp_14);
    step1_7 = vsubq_s32(inp_0, inp_15);
    s0 = vaddq_s32(input_0, input_7);
    s1 = vaddq_s32(input_1, input_6);
    s2 = vaddq_s32(input_2, input_5);
    s3 = vaddq_s32(input_3, input_4);
    s4 = vsubq_s32(input_3, input_4);
    s5 = vsubq_s32(input_2, input_5);
    s6 = vsubq_s32(input_1, input_6);
    s7 = vsubq_s32(input_0, input_7);
    x0 = vaddq_s32(s0, s3);
    x1 = vaddq_s32(s1, s2);
    x2 = vsubq_s32(s1, s2);
    x3 = vsubq_s32(s0, s3);
    t0 = vmulq_n_s32(vaddq_s32(x0, x1), cospi_16_64);
    t1 = vmulq_n_s32(vsubq_s32(x0, x1), cospi_16_64);
    t2 = vmlaq_n_s32(vmulq_n_s32(x3, cospi_8_64), x2, cospi_24_64);
    t3 = vmlsq_n_s32(vmulq_n_s32(x3, cospi_24_64), x2, cospi_8_64);
    out_0 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    out_4 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    out_8 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    out_12 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    t0 = vmulq_n_s32(vsubq_s32(s6, s5), cospi_16_64);
    t1 = vmulq_n_s32(vaddq_s32(s6, s5), cospi_16_64);
    t2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS);
    t3 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS);
    x0 = vaddq_s32(s4, t2);
    x1 = vsubq_s32(s4, t2);
    x2 = vsubq_s32(s7, t3);
    x3 = vaddq_s32(s7, t3);
    t0 = vmlaq_n_s32(vmulq_n_s32(x0, cospi_28_64), x3, cospi_4_64);
    t1 = vmlaq_n_s32(vmulq_n_s32(x1, cospi_12_64), x2, cospi_20_64);
    t2 = vmlsq_n_s32(vmulq_n_s32(x2, cospi_12_64), x1, cospi_20_64);
    t3 = vmlsq_n_s32(vmulq_n_s32(x3, cospi_28_64), x0, cospi_4_64);
    out_2 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    out_6 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    out_10 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    out_14 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    temp1 = vmulq_n_s32(vsubq_s32(step1_5, step1_2), cospi_16_64);
    temp2 = vmulq_n_s32(vsubq_s32(step1_4, step1_3), cospi_16_64);
    step2_2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_3 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    temp1 = vmulq_n_s32(vaddq_s32(step1_4, step1_3), cospi_16_64);
    temp2 = vmulq_n_s32(vaddq_s32(step1_5, step1_2), cospi_16_64);
    step2_4 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_5 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    step3_0 = vaddq_s32(step1_0, step2_3);
    step3_1 = vaddq_s32(step1_1, step2_2);
    step3_2 = vsubq_s32(step1_1, step2_2);
    step3_3 = vsubq_s32(step1_0, step2_3);
    step3_4 = vsubq_s32(step1_7, step2_4);
    step3_5 = vsubq_s32(step1_6, step2_5);
    step3_6 = vaddq_s32(step1_6, step2_5);
    step3_7 = vaddq_s32(step1_7, step2_4);
    temp1 = vmlaq_n_s32(vmulq_n_s32(step3_1, -cospi_8_64), step3_6, cospi_24_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step3_2, cospi_24_64), step3_5, cospi_8_64);
    step2_1 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    temp1 = vmlsq_n_s32(vmulq_n_s32(step3_2, cospi_8_64), step3_5, cospi_24_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step3_1, cospi_24_64), step3_6, cospi_8_64);
    step2_5 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS);
    step2_6 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS);
    step1_0 = vaddq_s32(step3_0, step2_1);
    step1_1 = vsubq_s32(step3_0, step2_1);
    step1_2 = vaddq_s32(step3_3, step2_2);
    step1_3 = vsubq_s32(step3_3, step2_2);
    step1_4 = vsubq_s32(step3_4, step2_5);
    step1_5 = vaddq_s32(step3_4, step2_5);
    step1_6 = vsubq_s32(step3_7, step2_6);
    step1_7 = vaddq_s32(step3_7, step2_6);
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_0, cospi_30_64), step1_7, cospi_2_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_1, cospi_14_64), step1_6, cospi_18_64);
    out_1 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_9 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_2, cospi_22_64), step1_5, cospi_10_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_3, cospi_6_64), step1_4, cospi_26_64);
    out_5 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_13 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_3, -cospi_26_64), step1_4, cospi_6_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_2, -cospi_10_64), step1_5, cospi_22_64);
    out_3 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_11 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    temp1 = vmlaq_n_s32(vmulq_n_s32(step1_1, -cospi_18_64), step1_6, cospi_14_64);
    temp2 = vmlaq_n_s32(vmulq_n_s32(step1_0, -cospi_2_64), step1_7, cospi_30_64);
    out_7 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp1), DCT_CONST_BITS));
    out_15 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), temp2), DCT_CONST_BITS));
    out[0] = out_0;
    out[1] = out_1;
    out[2] = out_2;
    out[3] = out_3;
    out[4] = out_4;
    out[5] = out_5;
    out[6] = out_6;
    out[7] = out_7;
    out[8] = out_8;
    out[9] = out_9;
    out[10] = out_10;
    out[11] = out_11;
    out[12] = out_12;
    out[13] = out_13;
    out[14] = out_14;
    out[15] = out_15;
}

static void fadst16_x4(const int32x4_t *in, int32x4_t *out) {
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

static void fdct8x8_1d_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t s0, s3, x0, s1, s2, x1, x2, x3, t0, t1, t2, t3, output_0, output_2, output_4, output_6, s6, s5, s4,
        s7, output_1, output_3, output_5, output_7;
    s0 = in[0];
    s1 = in[1];
    s2 = in[2];
    s3 = in[3];
    s4 = in[4];
    s5 = in[5];
    s6 = in[6];
    s7 = in[7];
    x0 = vaddq_s32(s0, s3);
    x1 = vaddq_s32(s1, s2);
    x2 = vsubq_s32(s1, s2);
    x3 = vsubq_s32(s0, s3);
    t0 = vmulq_n_s32(vaddq_s32(x0, x1), cospi_16_64);
    t1 = vmulq_n_s32(vsubq_s32(x0, x1), cospi_16_64);
    t2 = vmlaq_n_s32(vmulq_n_s32(x2, cospi_24_64), x3, cospi_8_64);
    t3 = vmlaq_n_s32(vmulq_n_s32(x2, -cospi_8_64), x3, cospi_24_64);
    output_0 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    output_2 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    output_4 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    output_6 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    t0 = vmulq_n_s32(vsubq_s32(s6, s5), cospi_16_64);
    t1 = vmulq_n_s32(vaddq_s32(s6, s5), cospi_16_64);
    t2 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS);
    t3 = vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS);
    x0 = vaddq_s32(s4, t2);
    x1 = vsubq_s32(s4, t2);
    x2 = vsubq_s32(s7, t3);
    x3 = vaddq_s32(s7, t3);
    t0 = vmlaq_n_s32(vmulq_n_s32(x0, cospi_28_64), x3, cospi_4_64);
    t1 = vmlaq_n_s32(vmulq_n_s32(x1, cospi_12_64), x2, cospi_20_64);
    t2 = vmlsq_n_s32(vmulq_n_s32(x2, cospi_12_64), x1, cospi_20_64);
    t3 = vmlsq_n_s32(vmulq_n_s32(x3, cospi_28_64), x0, cospi_4_64);
    output_1 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    output_3 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    output_5 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    output_7 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    out[0] = output_0;
    out[1] = output_1;
    out[2] = output_2;
    out[3] = output_3;
    out[4] = output_4;
    out[5] = output_5;
    out[6] = output_6;
    out[7] = output_7;
}

static void fdct8_x4(const int32x4_t *in, int32x4_t *out) {
    int32x4_t input_0, input_7, s0, input_1, input_6, s1, input_2, input_5, s2, input_3, input_4, s3, s4, s5, s6,
        s7, x0, x1, x2, x3, t0, t1, t2, t3, output_0, output_2, output_4, output_6, output_1, output_3, output_5,
        output_7;
    input_0 = in[0];
    input_1 = in[1];
    input_2 = in[2];
    input_3 = in[3];
    input_4 = in[4];
    input_5 = in[5];
    input_6 = in[6];
    input_7 = in[7];
    s0 = vaddq_s32(input_0, input_7);
    s1 = vaddq_s32(input_1, input_6);
    s2 = vaddq_s32(input_2, input_5);
    s3 = vaddq_s32(input_3, input_4);
    s4 = vsubq_s32(input_3, input_4);
    s5 = vsubq_s32(input_2, input_5);
    s6 = vsubq_s32(input_1, input_6);
    s7 = vsubq_s32(input_0, input_7);
    x0 = vaddq_s32(s0, s3);
    x1 = vaddq_s32(s1, s2);
    x2 = vsubq_s32(s1, s2);
    x3 = vsubq_s32(s0, s3);
    t0 = vmulq_n_s32(vaddq_s32(x0, x1), cospi_16_64);
    t1 = vmulq_n_s32(vsubq_s32(x0, x1), cospi_16_64);
    t2 = vmlaq_n_s32(vmulq_n_s32(x2, cospi_24_64), x3, cospi_8_64);
    t3 = vmlaq_n_s32(vmulq_n_s32(x2, -cospi_8_64), x3, cospi_24_64);
    output_0 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    output_2 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    output_4 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    output_6 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    t0 = vmulq_n_s32(vsubq_s32(s6, s5), cospi_16_64);
    t1 = vmulq_n_s32(vaddq_s32(s6, s5), cospi_16_64);
    t2 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    t3 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    x0 = vaddq_s32(s4, t2);
    x1 = vsubq_s32(s4, t2);
    x2 = vsubq_s32(s7, t3);
    x3 = vaddq_s32(s7, t3);
    t0 = vmlaq_n_s32(vmulq_n_s32(x0, cospi_28_64), x3, cospi_4_64);
    t1 = vmlaq_n_s32(vmulq_n_s32(x1, cospi_12_64), x2, cospi_20_64);
    t2 = vmlsq_n_s32(vmulq_n_s32(x2, cospi_12_64), x1, cospi_20_64);
    t3 = vmlsq_n_s32(vmulq_n_s32(x3, cospi_28_64), x0, cospi_4_64);
    output_1 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t0), DCT_CONST_BITS));
    output_3 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t2), DCT_CONST_BITS));
    output_5 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t1), DCT_CONST_BITS));
    output_7 = wrap16_s32(vshrq_n_s32(vaddq_s32(vdupq_n_s32(DCT_CONST_ROUNDING), t3), DCT_CONST_BITS));
    out[0] = output_0;
    out[1] = output_1;
    out[2] = output_2;
    out[3] = output_3;
    out[4] = output_4;
    out[5] = output_5;
    out[6] = output_6;
    out[7] = output_7;
}

static void fadst8_x4(const int32x4_t *in, int32x4_t *out) {
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


/*******************************************
 * 32x32
 *******************************************/
void eb_vp9_fdct32x32_neon(const int16_t *input, tran_low_t *out, int stride) {
    DECLARE_ALIGNED(16, int32_t, inter[32 * 32]); // inter[c * 32 + k]: output k of the transform of column c
    int32x4_t in[32], o[32];
    // Columns, 4 at a time
    for (int c = 0; c < 32; c += 4) {
        for (int j = 0; j < 32; j++) in[j] = vshlq_n_s32(vmovl_s16(vld1_s16(input + j * stride + c)), 2);
        fdct32_x4(in, o);
        for (int j = 0; j < 32; j++) o[j] = half_round_shift_pos_s32(o[j]);
        store_transposed_s32(inter + c * 32, 32, o, 32);
    }
    // Rows, 4 at a time
    for (int r = 0; r < 32; r += 4) {
        for (int j = 0; j < 32; j++) in[j] = vld1q_s32(inter + j * 32 + r);
        fdct32_x4(in, o);
        for (int j = 0; j < 32; j++) o[j] = half_round_shift_s32(o[j]);
        store_transposed_s16(out + r * 32, 32, o, 32);
    }
}

// Only the top-left 16x16 outputs; the other outputs are not written (as in C)
void eb_vpx_partial_fdct32x32_neon(const int16_t *input, tran_low_t *out, int stride) {
    DECLARE_ALIGNED(16, int32_t, inter[32 * 16]); // inter[c * 16 + k], k < 16
    int32x4_t in[32], o[16];
    for (int c = 0; c < 32; c += 4) {
        for (int j = 0; j < 32; j++) in[j] = vshlq_n_s32(vmovl_s16(vld1_s16(input + j * stride + c)), 2);
        partial_fdct32_x4(in, o);
        for (int j = 0; j < 16; j++) o[j] = half_round_shift_pos_s32(o[j]);
        store_transposed_s32(inter + c * 16, 16, o, 16);
    }
    for (int r = 0; r < 16; r += 4) {
        for (int j = 0; j < 32; j++) in[j] = vld1q_s32(inter + j * 16 + r);
        partial_fdct32_x4(in, o);
        for (int j = 0; j < 16; j++) o[j] = half_round_shift_s32(o[j]);
        store_transposed_s16(out + r * 32, 32, o, 16);
    }
}

/*******************************************
 * 16x16
 *******************************************/
ALWAYS_INLINE int32x4_t load_s16x4_s32(const int16_t *p) { return vmovl_s16(vld1_s16(p)); }

void eb_vp9_fdct16x16_neon(const int16_t *input, tran_low_t *output, int stride) {
    DECLARE_ALIGNED(16, tran_low_t, inter[16 * 16]); // inter[c * 16 + k]: output k of column c
    int32x4_t in[16], o[16], v[16];
    // Columns
    for (int c = 0; c < 16; c += 4) {
        for (int k = 0; k < 16; k++) v[k] = load_s16x4_s32(input + k * stride + c);
        for (int k = 0; k < 8; k++) {
            in[k]     = vshlq_n_s32(vaddq_s32(v[k], v[15 - k]), 2);
            in[8 + k] = vshlq_n_s32(vsubq_s32(v[7 - k], v[8 + k]), 2);
        }
        fdct16x16_1d_x4(in, o);
        store_transposed_s16(inter + c * 16, 16, o, 16);
    }
    // Rows
    for (int r = 0; r < 16; r += 4) {
        for (int k = 0; k < 16; k++) v[k] = vshrq_n_s32(vaddq_s32(load_s16x4_s32(inter + k * 16 + r), vdupq_n_s32(1)), 2);
        for (int k = 0; k < 8; k++) {
            in[k]     = vaddq_s32(v[k], v[15 - k]);
            in[8 + k] = vsubq_s32(v[7 - k], v[8 + k]);
        }
        fdct16x16_1d_x4(in, o);
        store_transposed_s16(output + r * 16, 16, o, 16);
    }
}

typedef void (*Txfm1dX4)(const int32x4_t *in, int32x4_t *out);

void eb_vp9_fht16x16_neon(const int16_t *input, tran_low_t *output, int stride, int tx_type) {
    static const Txfm1dX4 cols[4] = {fdct16_x4, fadst16_x4, fdct16_x4, fadst16_x4};
    static const Txfm1dX4 rows[4] = {fdct16_x4, fdct16_x4, fadst16_x4, fadst16_x4};
    if (tx_type == 0) { // DCT_DCT
        eb_vp9_fdct16x16_neon(input, output, stride);
        return;
    }
    DECLARE_ALIGNED(16, tran_low_t, inter[16 * 16]);
    int32x4_t in[16], o[16];
    for (int c = 0; c < 16; c += 4) {
        for (int j = 0; j < 16; j++) in[j] = vmovl_s16(vshl_n_s16(vld1_s16(input + j * stride + c), 2));
        cols[tx_type](in, o);
        for (int j = 0; j < 16; j++) o[j] = half_round_shift_s32(o[j]);
        store_transposed_s16(inter + c * 16, 16, o, 16);
    }
    for (int r = 0; r < 16; r += 4) {
        for (int j = 0; j < 16; j++) in[j] = load_s16x4_s32(inter + j * 16 + r);
        rows[tx_type](in, o);
        store_transposed_s16(output + r * 16, 16, o, 16);
    }
}

/*******************************************
 * 8x8
 *******************************************/
// x / 2 rounded toward zero
ALWAYS_INLINE int32x4_t half_s32(const int32x4_t x) {
    return vshrq_n_s32(vsubq_s32(x, vreinterpretq_s32_u32(vcltzq_s32(x))), 1);
}

void eb_vp9_fdct8x8_neon(const int16_t *input, tran_low_t *output, int stride) {
    DECLARE_ALIGNED(16, tran_low_t, inter[8 * 8]);
    int32x4_t in[8], o[8], v[8];
    for (int c = 0; c < 8; c += 4) {
        for (int k = 0; k < 8; k++) v[k] = load_s16x4_s32(input + k * stride + c);
        for (int k = 0; k < 4; k++) {
            in[k]     = vshlq_n_s32(vaddq_s32(v[k], v[7 - k]), 2);
            in[4 + k] = vshlq_n_s32(vsubq_s32(v[3 - k], v[4 + k]), 2);
        }
        fdct8x8_1d_x4(in, o);
        store_transposed_s16(inter + c * 8, 8, o, 8);
    }
    for (int r = 0; r < 8; r += 4) {
        for (int k = 0; k < 8; k++) v[k] = load_s16x4_s32(inter + k * 8 + r);
        for (int k = 0; k < 4; k++) {
            in[k]     = vaddq_s32(v[k], v[7 - k]);
            in[4 + k] = vsubq_s32(v[3 - k], v[4 + k]);
        }
        fdct8x8_1d_x4(in, o);
        for (int k = 0; k < 8; k++) o[k] = half_s32(o[k]);
        store_transposed_s16(output + r * 8, 8, o, 8);
    }
}

void eb_vp9_fht8x8_neon(const int16_t *input, tran_low_t *output, int stride, int tx_type) {
    static const Txfm1dX4 cols[4] = {fdct8_x4, fadst8_x4, fdct8_x4, fadst8_x4};
    static const Txfm1dX4 rows[4] = {fdct8_x4, fdct8_x4, fadst8_x4, fadst8_x4};
    if (tx_type == 0) { // DCT_DCT
        eb_vp9_fdct8x8_neon(input, output, stride);
        return;
    }
    DECLARE_ALIGNED(16, tran_low_t, inter[8 * 8]);
    int32x4_t in[8], o[8];
    for (int c = 0; c < 8; c += 4) {
        for (int j = 0; j < 8; j++) in[j] = vmovl_s16(vshl_n_s16(vld1_s16(input + j * stride + c), 2));
        cols[tx_type](in, o);
        store_transposed_s16(inter + c * 8, 8, o, 8);
    }
    for (int r = 0; r < 8; r += 4) {
        for (int j = 0; j < 8; j++) in[j] = load_s16x4_s32(inter + j * 8 + r);
        rows[tx_type](in, o);
        for (int j = 0; j < 8; j++) o[j] = half_s32(o[j]);
        store_transposed_s16(output + r * 8, 8, o, 8);
    }
}
