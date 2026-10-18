/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON sub-pixel convolution (unscaled). Bit-exact with vpx_convolve.c:
// - The 8-tap sum is computed in int16. Taps 0-2 and 5-7 are accumulated with wrapping MLA: their absolute sum is
//   at most 74 for every VP9 filter, so the partial sum fits in int16. Taps 3 and 4 are non-negative in every VP9
//   filter (<= 128) and are added with saturation: saturation can only happen when the exact sum is >= 32767,
//   where C also clips the result to 255.
// - vqrshrun_n_s16(sum, 7) == clip_pixel(ROUND_POWER_OF_TWO(sum, 7)).
// - The 2D filter rounds the horizontal pass to 8 bits in a 64 x (h + 7) buffer, as the C version does.
// Scaled prediction (step != 16), sub-pixel offsets >= 16 and widths that are not a multiple of 4 use the C code.

#include <string.h>

#include "EbNeonCommon.h"
#include "vpx_dsp_rtcd.h"
#include "vpx_filter.h"
#include "vpx_dsp_common.h"
#include "mem.h"

ALWAYS_INLINE uint8x8_t filter8(const int16x8_t s0, const int16x8_t s1, const int16x8_t s2, const int16x8_t s3,
                                const int16x8_t s4, const int16x8_t s5, const int16x8_t s6, const int16x8_t s7,
                                const int16x8_t f) {
    const int16x4_t lo  = vget_low_s16(f);
    const int16x4_t hi  = vget_high_s16(f);
    int16x8_t       sum = vmulq_lane_s16(s0, lo, 0);
    sum                 = vmlaq_lane_s16(sum, s1, lo, 1);
    sum                 = vmlaq_lane_s16(sum, s2, lo, 2);
    sum                 = vmlaq_lane_s16(sum, s5, hi, 1);
    sum                 = vmlaq_lane_s16(sum, s6, hi, 2);
    sum                 = vmlaq_lane_s16(sum, s7, hi, 3);
    sum                 = vqaddq_s16(sum, vmulq_lane_s16(s3, lo, 3));
    sum                 = vqaddq_s16(sum, vmulq_lane_s16(s4, hi, 0));
    return vqrshrun_n_s16(sum, FILTER_BITS);
}

ALWAYS_INLINE int16x8_t widen(const uint8x8_t v) { return vreinterpretq_s16_u16(vmovl_u8(v)); }

// 8 (or 4) horizontally filtered pixels starting at p. a holds p[-3..4], b holds p[4..11] (p[4..7] for 4 outputs).
ALWAYS_INLINE uint8x8_t horiz8(const uint8x8_t a, const uint8x8_t b, const int16x8_t f) {
    const int16x8_t wa = widen(a);
    const int16x8_t wb = widen(b);
    const int16x8_t hi = vextq_s16(wb, wb, 1); // p[5..11]
    return filter8(wa,
                   vextq_s16(wa, hi, 1),
                   vextq_s16(wa, hi, 2),
                   vextq_s16(wa, hi, 3),
                   vextq_s16(wa, hi, 4),
                   vextq_s16(wa, hi, 5),
                   vextq_s16(wa, hi, 6),
                   vextq_s16(wa, hi, 7),
                   f);
}

#if defined(__ARM_FEATURE_DOTPROD)
// Dot-product horizontal filter: with s = p - 128 (as int8) and taps f (as int8),
// sum f * p = sum f * s + 128 * sum f, computed exactly in int32.
static const uint8_t dot_perm[48] = {0, 1, 2,  3,  1, 2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6,
                                     4, 5, 6,  7,  5, 6,  7,  8,  6,  7,  8,  9,  7,  8,  9,  10,
                                     8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14};

// 8 filtered pixels from q = p[-3..11] (lane 15 unused)
ALWAYS_INLINE uint8x8_t horiz8_dot(const uint8x16_t q, const int8x8_t f, const int32x4_t corr, const uint8x16_t perm0,
                                   const uint8x16_t perm1, const uint8x16_t perm2) {
    const int8x16_t s  = vreinterpretq_s8_u8(veorq_u8(q, vdupq_n_u8(0x80)));
    const int8x16_t p0 = vqtbl1q_s8(s, perm0);
    const int8x16_t p1 = vqtbl1q_s8(s, perm1);
    const int8x16_t p2 = vqtbl1q_s8(s, perm2);
    int32x4_t       lo = vdotq_lane_s32(corr, p0, f, 0);
    int32x4_t       hi = vdotq_lane_s32(corr, p1, f, 0);
    lo                 = vdotq_lane_s32(lo, p1, f, 1);
    hi                 = vdotq_lane_s32(hi, p2, f, 1);
    return vqmovn_u16(vcombine_u16(vqrshrun_n_s32(lo, FILTER_BITS), vqrshrun_n_s32(hi, FILTER_BITS)));
}

static void convolve_horiz_dot(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                               const int16_t *filter, int w, int h, int avg) {
    const int16x8_t  f16   = vld1q_s16(filter);
    const int8x8_t   f     = vmovn_s16(f16);
    const int32x4_t  corr  = vdupq_n_s32(128 * vaddvq_s16(f16));
    const uint8x16_t perm0 = vld1q_u8(dot_perm), perm1 = vld1q_u8(dot_perm + 16), perm2 = vld1q_u8(dot_perm + 32);
    src -= SUBPEL_TAPS / 2 - 1;
    for (int y = 0; y < h; y++) {
        int x = 0;
        // full 16-byte loads while they stay inside the bytes the C code reads (p[-3..w+3] of the row)
        for (; x + 16 <= w; x += 8) {
            uint8x8_t d = horiz8_dot(vld1q_u8(src + x), f, corr, perm0, perm1, perm2);
            if (avg)
                d = vrhadd_u8(d, vld1_u8(dst + x));
            vst1_u8(dst + x, d);
        }
        if (x + 8 <= w) { // last 8: p[-3..11] without touching p[12]
            const uint8x8_t b = vld1_u8(src + x + 7);
            uint8x8_t       d = horiz8_dot(vcombine_u8(vld1_u8(src + x), vext_u8(b, b, 1)), f, corr, perm0, perm1, perm2);
            if (avg)
                d = vrhadd_u8(d, vld1_u8(dst + x));
            vst1_u8(dst + x, d);
            x += 8;
        }
        if (x < w) { // w % 8 == 4: p[-3..7]
            const uint8x8_t b = load_u8_4(src + x + 7);
            uint8x8_t       d = horiz8_dot(vcombine_u8(vld1_u8(src + x), vext_u8(b, b, 1)), f, corr, perm0, perm1, perm2);
            if (avg)
                d = vrhadd_u8(d, load_u8_4(dst + x));
            store_u8_4(dst + x, d);
        }
        src += src_stride;
        dst += dst_stride;
    }
}
#endif

ALWAYS_INLINE void convolve_horiz_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                                       const int16_t *filter, int w, int h, int avg) {
#if defined(__ARM_FEATURE_DOTPROD)
    if (filter[3] != 128 && filter[4] != 128) { // all taps fit in int8 (128 only appears in the full-pel filter)
        convolve_horiz_dot(src, src_stride, dst, dst_stride, filter, w, h, avg);
        return;
    }
#endif
    const int16x8_t f = vld1q_s16(filter);
    src -= SUBPEL_TAPS / 2 - 1;
    for (int y = 0; y < h; y++) {
        int x = 0;
        for (; x + 8 <= w; x += 8) {
            uint8x8_t d = horiz8(vld1_u8(src + x), vld1_u8(src + x + 7), f);
            if (avg)
                d = vrhadd_u8(d, vld1_u8(dst + x));
            vst1_u8(dst + x, d);
        }
        if (x < w) { // w % 8 == 4
            uint8x8_t d = horiz8(vld1_u8(src + x), load_u8_4(src + x + 7), f);
            if (avg)
                d = vrhadd_u8(d, load_u8_4(dst + x));
            store_u8_4(dst + x, d);
        }
        src += src_stride;
        dst += dst_stride;
    }
}

ALWAYS_INLINE uint8x8_t load_row(const uint8_t *p, int w4) { return w4 ? load_u8_4(p) : vld1_u8(p); }

// Vertical filter of an 8 (w4 = 0) or 4 (w4 = 1) pixel wide column
ALWAYS_INLINE void convolve_vert_col(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                                     const int16x8_t f, int h, int avg, int w4) {
    src -= src_stride * (SUBPEL_TAPS / 2 - 1);
    int16x8_t s0 = widen(load_row(src + 0 * src_stride, w4));
    int16x8_t s1 = widen(load_row(src + 1 * src_stride, w4));
    int16x8_t s2 = widen(load_row(src + 2 * src_stride, w4));
    int16x8_t s3 = widen(load_row(src + 3 * src_stride, w4));
    int16x8_t s4 = widen(load_row(src + 4 * src_stride, w4));
    int16x8_t s5 = widen(load_row(src + 5 * src_stride, w4));
    int16x8_t s6 = widen(load_row(src + 6 * src_stride, w4));
    src += 7 * src_stride;
    for (int y = 0; y < h; y++) {
        const int16x8_t s7 = widen(load_row(src, w4));
        uint8x8_t       d  = filter8(s0, s1, s2, s3, s4, s5, s6, s7, f);
        if (avg)
            d = vrhadd_u8(d, load_row(dst, w4));
        if (w4)
            store_u8_4(dst, d);
        else
            vst1_u8(dst, d);
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s5;
        s5 = s6;
        s6 = s7;
        src += src_stride;
        dst += dst_stride;
    }
}

ALWAYS_INLINE void convolve_vert_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                                      const int16_t *filter, int w, int h, int avg) {
    const int16x8_t f = vld1q_s16(filter);
    int             x = 0;
    for (; x + 8 <= w; x += 8) convolve_vert_col(src + x, src_stride, dst + x, dst_stride, f, h, avg, 0);
    if (x < w)
        convolve_vert_col(src + x, src_stride, dst + x, dst_stride, f, h, avg, 1);
}

// The NEON paths cover the unscaled case that the encoder uses
static inline int neon_ok(int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
    return x_step_q4 == 16 && y_step_q4 == 16 && (unsigned)x0_q4 < 16 && (unsigned)y0_q4 < 16 && !(w & 3) && w <= 64 &&
        h <= 64;
}

void eb_vp9_convolve8_horiz_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                                 const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                 int h) {
    if (!neon_ok(x0_q4, x_step_q4, 0, 16, w, h)) {
        eb_vp9_convolve8_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        return;
    }
    convolve_horiz_neon(src, src_stride, dst, dst_stride, filter[x0_q4], w, h, 0);
}

void eb_vp9_convolve8_avg_horiz_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                                     const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                     int w, int h) {
    if (!neon_ok(x0_q4, x_step_q4, 0, 16, w, h)) {
        eb_vp9_convolve8_avg_horiz_c(
            src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        return;
    }
    convolve_horiz_neon(src, src_stride, dst, dst_stride, filter[x0_q4], w, h, 1);
}

void eb_vp9_convolve8_vert_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                int h) {
    if (!neon_ok(0, 16, y0_q4, y_step_q4, w, h)) {
        eb_vp9_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        return;
    }
    convolve_vert_neon(src, src_stride, dst, dst_stride, filter[y0_q4], w, h, 0);
}

void eb_vp9_convolve8_avg_vert_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                                    const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                    int w, int h) {
    if (!neon_ok(0, 16, y0_q4, y_step_q4, w, h)) {
        eb_vp9_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        return;
    }
    convolve_vert_neon(src, src_stride, dst, dst_stride, filter[y0_q4], w, h, 1);
}

void eb_vp9_convolve8_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                           int h) {
    if (!neon_ok(x0_q4, x_step_q4, y0_q4, y_step_q4, w, h)) {
        eb_vp9_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        return;
    }
    DECLARE_ALIGNED(16, uint8_t, temp[64 * (64 + SUBPEL_TAPS - 1)]);
    convolve_horiz_neon(
        src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp, 64, filter[x0_q4], w, h + SUBPEL_TAPS - 1, 0);
    convolve_vert_neon(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride, filter[y0_q4], w, h, 0);
}

void eb_vp9_convolve8_avg_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                               int h) {
    if (!neon_ok(x0_q4, x_step_q4, y0_q4, y_step_q4, w, h)) {
        eb_vp9_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        return;
    }
    DECLARE_ALIGNED(16, uint8_t, temp[64 * (64 + SUBPEL_TAPS - 1)]);
    convolve_horiz_neon(
        src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp, 64, filter[x0_q4], w, h + SUBPEL_TAPS - 1, 0);
    convolve_vert_neon(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride, filter[y0_q4], w, h, 1);
}

ALWAYS_INLINE void avg_rows(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, int w,
                            int h) {
    for (int y = 0; y < h; y++) {
        int x = 0;
        for (; x + 16 <= w; x += 16) vst1q_u8(dst + x, vrhaddq_u8(vld1q_u8(src + x), vld1q_u8(dst + x)));
        if (x + 8 <= w) {
            vst1_u8(dst + x, vrhadd_u8(vld1_u8(src + x), vld1_u8(dst + x)));
            x += 8;
        }
        if (x + 4 <= w) {
            store_u8_4(dst + x, vrhadd_u8(load_u8_4(src + x), load_u8_4(dst + x)));
            x += 4;
        }
        for (; x < w; x++) dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);
        src += src_stride;
        dst += dst_stride;
    }
}

void eb_vp9_convolve_avg_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                              int h) {
    (void)filter;
    (void)x0_q4;
    (void)x_step_q4;
    (void)y0_q4;
    (void)y_step_q4;
    // constant widths for the VP9 block sizes so the row loop is fully unrolled
    switch (w) {
    case 4: avg_rows(src, src_stride, dst, dst_stride, 4, h); break;
    case 8: avg_rows(src, src_stride, dst, dst_stride, 8, h); break;
    case 16: avg_rows(src, src_stride, dst, dst_stride, 16, h); break;
    case 32: avg_rows(src, src_stride, dst, dst_stride, 32, h); break;
    case 64: avg_rows(src, src_stride, dst, dst_stride, 64, h); break;
    default: avg_rows(src, src_stride, dst, dst_stride, w, h); break;
    }
}

void eb_vp9_convolve_copy_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                               int h) {
    (void)filter;
    (void)x0_q4;
    (void)x_step_q4;
    (void)y0_q4;
    (void)y_step_q4;
    // Fixed-size copies for the VP9 block widths (inlined), memcpy otherwise
    switch (w) {
    case 4:
        for (int y = 0; y < h; y++, src += src_stride, dst += dst_stride) memcpy(dst, src, 4);
        break;
    case 8:
        for (int y = 0; y < h; y++, src += src_stride, dst += dst_stride) memcpy(dst, src, 8);
        break;
    case 16:
        for (int y = 0; y < h; y++, src += src_stride, dst += dst_stride) vst1q_u8(dst, vld1q_u8(src));
        break;
    case 32:
        for (int y = 0; y < h; y++, src += src_stride, dst += dst_stride) vst1q_u8_x2(dst, vld1q_u8_x2(src));
        break;
    case 64:
        for (int y = 0; y < h; y++, src += src_stride, dst += dst_stride) vst1q_u8_x4(dst, vld1q_u8_x4(src));
        break;
    default:
        for (int y = 0; y < h; y++, src += src_stride, dst += dst_stride) memcpy(dst, src, w);
        break;
    }
}
