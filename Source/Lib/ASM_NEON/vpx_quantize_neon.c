/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON quantization, bit-exact with quantize.c.
// The C code walks the coefficients in scan order and skips the ones with |coeff| < zbin (they quantize to 0).
// Here all coefficients are processed in raster order with the same per-coefficient arithmetic, and the end of
// block is max(iscan[rc] + 1) over the non-zero results, which equals the C scan-order eob.
// - (tmp * quant) >> 16 == vqdmulh(tmp, quant) >> 1 for tmp >= 0 (no saturation);
// - the following "+ tmp" and "* quant_shift" steps can exceed 16 bits and are done in 32 bits as in C.

#include "EbNeonCommon.h"
#include "vpx_dsp_rtcd.h"
#include "mem.h"

ALWAYS_INLINE void quantize_8(const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                              const int16_t *iscan, const int16x8_t zbin, const int16x8_t round, const int16x8_t quant,
                              const int16x8_t quant_shift, const int16x8_t dequant, int16x8_t *eob_max, int is32) {
    const int16x8_t coeff = vld1q_s16(coeff_ptr);
    const int16x8_t sign  = vshrq_n_s16(coeff, 15);
    // |-32768| saturates to 32767: same result, as C clamps abs + round to 32767 and zbin <= 32767
    const int16x8_t abs_coeff = vqabsq_s16(coeff);
    const uint16x8_t in_zbin  = vcgeq_s16(abs_coeff, zbin);
    const int16x8_t tmp       = vqaddq_s16(abs_coeff, round); // clamp(abs + round, INT16_MIN, INT16_MAX)
    const int16x8_t t1        = vshrq_n_s16(vqdmulhq_s16(tmp, quant), 1);
    const int32x4_t s_lo      = vaddl_s16(vget_low_s16(t1), vget_low_s16(tmp));
    const int32x4_t s_hi      = vaddl_s16(vget_high_s16(t1), vget_high_s16(tmp));
    int32x4_t       q_lo      = vmulq_s32(s_lo, vmovl_s16(vget_low_s16(quant_shift)));
    int32x4_t       q_hi      = vmulq_s32(s_hi, vmovl_s16(vget_high_s16(quant_shift)));
    if (is32) {
        q_lo = vshrq_n_s32(q_lo, 15);
        q_hi = vshrq_n_s32(q_hi, 15);
    } else {
        q_lo = vshrq_n_s32(q_lo, 16);
        q_hi = vshrq_n_s32(q_hi, 16);
    }
    // tmp != 0 tested on the 32-bit value as in C
    const uint16x8_t nz = vandq_u16(vcombine_u16(vmovn_u32(vtstq_s32(q_lo, q_lo)), vmovn_u32(vtstq_s32(q_hi, q_hi))),
                                    in_zbin);
    const int16x8_t q16 = vcombine_s16(vmovn_s32(q_lo), vmovn_s32(q_hi));
    // (int16_t)((tmp ^ sign) - sign), zero outside the zero bin
    const int16x8_t qcoeff = vandq_s16(vsubq_s16(veorq_s16(q16, sign), sign), vreinterpretq_s16_u16(in_zbin));
    vst1q_s16(qcoeff_ptr, qcoeff);
    if (is32) {
        // qcoeff * dequant / 2 (truncating division of the 32-bit product)
        int32x4_t d_lo = vmull_s16(vget_low_s16(qcoeff), vget_low_s16(dequant));
        int32x4_t d_hi = vmull_s16(vget_high_s16(qcoeff), vget_high_s16(dequant));
        d_lo           = vshrq_n_s32(vsubq_s32(d_lo, vreinterpretq_s32_u32(vcltzq_s32(d_lo))), 1);
        d_hi           = vshrq_n_s32(vsubq_s32(d_hi, vreinterpretq_s32_u32(vcltzq_s32(d_hi))), 1);
        vst1q_s16(dqcoeff_ptr, vcombine_s16(vmovn_s32(d_lo), vmovn_s32(d_hi)));
    } else {
        vst1q_s16(dqcoeff_ptr, vmulq_s16(qcoeff, dequant));
    }
    const int16x8_t pos = vaddq_s16(vld1q_s16(iscan), vdupq_n_s16(1));
    *eob_max            = vmaxq_s16(*eob_max, vandq_s16(pos, vreinterpretq_s16_u16(nz)));
}

ALWAYS_INLINE void quantize_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
                                 const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr,
                                 tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
                                 uint16_t *eob_ptr, const int16_t *iscan, int is32) {
    // lane 0 of the first 8 coefficients is DC
    int16_t zb[2] = {zbin_ptr[0], zbin_ptr[1]}, rd[2] = {round_ptr[0], round_ptr[1]};
    if (is32) {
        zb[0] = (int16_t)ROUND_POWER_OF_TWO(zbin_ptr[0], 1);
        zb[1] = (int16_t)ROUND_POWER_OF_TWO(zbin_ptr[1], 1);
        rd[0] = (int16_t)ROUND_POWER_OF_TWO(round_ptr[0], 1);
        rd[1] = (int16_t)ROUND_POWER_OF_TWO(round_ptr[1], 1);
    }
    int16x8_t zbin        = vsetq_lane_s16(zb[0], vdupq_n_s16(zb[1]), 0);
    int16x8_t round       = vsetq_lane_s16(rd[0], vdupq_n_s16(rd[1]), 0);
    int16x8_t quant       = vsetq_lane_s16(quant_ptr[0], vdupq_n_s16(quant_ptr[1]), 0);
    int16x8_t quant_shift = vsetq_lane_s16(quant_shift_ptr[0], vdupq_n_s16(quant_shift_ptr[1]), 0);
    int16x8_t dequant     = vsetq_lane_s16(dequant_ptr[0], vdupq_n_s16(dequant_ptr[1]), 0);
    int16x8_t eob_max     = vdupq_n_s16(0);

    quantize_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan, zbin, round, quant, quant_shift, dequant, &eob_max, is32);
    // AC only from here
    zbin        = vdupq_n_s16(zb[1]);
    round       = vdupq_n_s16(rd[1]);
    quant       = vdupq_n_s16(quant_ptr[1]);
    quant_shift = vdupq_n_s16(quant_shift_ptr[1]);
    dequant     = vdupq_n_s16(dequant_ptr[1]);
    for (intptr_t i = 8; i < n_coeffs; i += 8)
        quantize_8(coeff_ptr + i,
                   qcoeff_ptr + i,
                   dqcoeff_ptr + i,
                   iscan + i,
                   zbin,
                   round,
                   quant,
                   quant_shift,
                   dequant,
                   &eob_max,
                   is32);
    *eob_ptr = (uint16_t)vmaxvq_s16(eob_max);
}

void eb_vp9_quantize_b_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs, int skip_block, const int16_t *zbin_ptr,
                            const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
                            uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan) {
    (void)skip_block;
    (void)scan;
    if (n_coeffs & 7) {
        eb_vp9_quantize_b_c(coeff_ptr, n_coeffs, skip_block, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr,
                            qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);
        return;
    }
    quantize_neon(coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr,
                  dequant_ptr, eob_ptr, iscan, 0);
}

void eb_vp9_quantize_b_32x32_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs, int skip_block,
                                  const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr,
                                  const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                                  const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan,
                                  const int16_t *iscan) {
    (void)skip_block;
    (void)scan;
    if (n_coeffs & 7) {
        eb_vp9_quantize_b_32x32_c(coeff_ptr, n_coeffs, skip_block, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr,
                                  qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);
        return;
    }
    quantize_neon(coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr, quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr,
                  dequant_ptr, eob_ptr, iscan, 1);
}
