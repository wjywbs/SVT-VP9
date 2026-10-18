/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON AVC-style 4-tap sub-pel interpolation (motion estimation), bit-exact with EbAvcStyleMcp_C.c.
// The taps are {-1, 25, 9, -1}, {-2, 18, 18, -2}, {-1, 9, 25, -1}: the sums lie in [-1020, 8670] and fit in
// 16 bits, and vqrshrun_n_s16(sum, 5) = clip((sum + 16) >> 5) as in C. Only the C read footprint is loaded.
// (The picture average kernel stays in C: clang vectorizes it 64 bytes per iteration.)

#include "EbNeonCommon.h"
#include "EbAvcStyleMcp_NEON.h"

static const uint8_t avc_taps[4][4] = {{0, 0, 0, 0}, {1, 25, 9, 1}, {2, 18, 18, 2}, {1, 9, 25, 1}}; // outer taps < 0

// a * t0' ... with the outer taps negative: t1 * b + t2 * c - t0 * a - t3 * d
ALWAYS_INLINE uint8x16_t filt16(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d, const uint8x8_t *t) {
    uint16x8_t lo = vmull_u8(vget_low_u8(b), t[1]);
    uint16x8_t hi = vmull_high_u8(b, vcombine_u8(t[1], t[1]));
    lo            = vmlal_u8(lo, vget_low_u8(c), t[2]);
    hi            = vmlal_high_u8(hi, c, vcombine_u8(t[2], t[2]));
    lo            = vmlsl_u8(lo, vget_low_u8(a), t[0]);
    hi            = vmlsl_high_u8(hi, a, vcombine_u8(t[0], t[0]));
    lo            = vmlsl_u8(lo, vget_low_u8(d), t[3]);
    hi            = vmlsl_high_u8(hi, d, vcombine_u8(t[3], t[3]));
    // wrapping u16 arithmetic, reinterpreted as the exact signed sum
    return vcombine_u8(vqrshrun_n_s16(vreinterpretq_s16_u16(lo), 5), vqrshrun_n_s16(vreinterpretq_s16_u16(hi), 5));
}

ALWAYS_INLINE uint8x8_t filt8(uint8x8_t a, uint8x8_t b, uint8x8_t c, uint8x8_t d, const uint8x8_t *t) {
    uint16x8_t s = vmull_u8(b, t[1]);
    s            = vmlal_u8(s, c, t[2]);
    s            = vmlsl_u8(s, a, t[0]);
    s            = vmlsl_u8(s, d, t[3]);
    return vqrshrun_n_s16(vreinterpretq_s16_u16(s), 5);
}

ALWAYS_INLINE int filt1(const uint8_t *p, int step, const uint8_t *t) {
    const int v = (t[1] * p[step] + t[2] * p[2 * step] - t[0] * p[0] - t[3] * p[3 * step] + 16) >> 5;
    return v < 0 ? 0 : v > 255 ? 255 : v;
}

// step: distance between taps (1: horizontal, stride: vertical); ref points at the first tap
ALWAYS_INLINE void avc_filter(const uint8_t *ref, int src_stride, int step, uint8_t *dst, int dst_stride, int w,
                              int h, uint32_t frac_pos) {
    const uint8_t *tp   = avc_taps[frac_pos];
    const uint8x8_t t[4] = {vdup_n_u8(tp[0]), vdup_n_u8(tp[1]), vdup_n_u8(tp[2]), vdup_n_u8(tp[3])};
    for (int y = 0; y < h; y++, ref += src_stride, dst += dst_stride) {
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            const uint8_t *r = ref + x;
            vst1q_u8(dst + x, filt16(vld1q_u8(r), vld1q_u8(r + step), vld1q_u8(r + 2 * step), vld1q_u8(r + 3 * step), t));
        }
        if (x + 8 <= w) {
            const uint8_t *r = ref + x;
            vst1_u8(dst + x, filt8(vld1_u8(r), vld1_u8(r + step), vld1_u8(r + 2 * step), vld1_u8(r + 3 * step), t));
            x += 8;
        }
        if (x + 4 <= w) {
            const uint8_t *r = ref + x;
            store_u8_4(dst + x,
                       filt8(load_u8_4(r), load_u8_4(r + step), load_u8_4(r + 2 * step), load_u8_4(r + 3 * step), t));
            x += 4;
        }
        for (; x < w; x++) dst[x] = (uint8_t)filt1(ref + x, step, tp);
    }
}

void eb_vp9_avc_style_luma_interpolation_filter_horizontal_neon(EbByte ref_pic, uint32_t src_stride, EbByte dst,
                                                                uint32_t dst_stride, uint32_t pu_width,
                                                                uint32_t pu_height, EbByte temp_buf,
                                                                uint32_t frac_pos) {
    (void)temp_buf;
    avc_filter(ref_pic - 1, (int)src_stride, 1, dst, (int)dst_stride, (int)pu_width, (int)pu_height, frac_pos);
}

void eb_vp9_avc_style_luma_interpolation_filter_vertical_neon(EbByte ref_pic, uint32_t src_stride, EbByte dst,
                                                              uint32_t dst_stride, uint32_t pu_width,
                                                              uint32_t pu_height, EbByte temp_buf, uint32_t frac_pos) {
    (void)temp_buf;
    avc_filter(ref_pic - src_stride, (int)src_stride, (int)src_stride, dst, (int)dst_stride, (int)pu_width,
               (int)pu_height, frac_pos);
}
