/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON SAD kernels. All results are bit-exact with the C kernels in EbComputeSAD_C.c, including the
// search-order tie-breaking of the "best SAD" searches (first position in raster order wins).

#include <arm_neon.h>

#include "EbComputeSAD_NEON.h"
#include "EbUtility.h"

#define ALWAYS_INLINE static inline __attribute__((always_inline))

/*******************************************
 * Accumulation helpers: |s - r| summed into 32-bit lanes (no overflow for any block size)
 *******************************************/
#if defined(__ARM_FEATURE_DOTPROD)
ALWAYS_INLINE uint32x4_t sad16_acc(uint32x4_t acc, uint8x16_t s, uint8x16_t r) {
    return vdotq_u32(acc, vabdq_u8(s, r), vdupq_n_u8(1));
}
ALWAYS_INLINE uint32x2_t sad8_acc(uint32x2_t acc, uint8x8_t s, uint8x8_t r) {
    return vdot_u32(acc, vabd_u8(s, r), vdup_n_u8(1));
}
#else
ALWAYS_INLINE uint32x4_t sad16_acc(uint32x4_t acc, uint8x16_t s, uint8x16_t r) {
    return vpadalq_u16(acc, vpaddlq_u8(vabdq_u8(s, r)));
}
ALWAYS_INLINE uint32x2_t sad8_acc(uint32x2_t acc, uint8x8_t s, uint8x8_t r) {
    return vpadal_u16(acc, vpaddl_u8(vabd_u8(s, r)));
}
#endif

ALWAYS_INLINE uint8x8_t load_u8_4(const uint8_t *p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return vreinterpret_u8_u32(vset_lane_u32(v, vdup_n_u32(0), 0));
}

// SAD of a w x h block (any w). w is a compile-time constant in the specialized callers.
ALWAYS_INLINE uint32_t sad_wxh(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride,
                               uint32_t w, uint32_t h) {
    uint32x4_t acc  = vdupq_n_u32(0);
    uint32x2_t acc8 = vdup_n_u32(0);
    uint32_t   tail = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 16 <= w; x += 16) acc = sad16_acc(acc, vld1q_u8(src + x), vld1q_u8(ref + x));
        if (x + 8 <= w) {
            acc8 = sad8_acc(acc8, vld1_u8(src + x), vld1_u8(ref + x));
            x += 8;
        }
        if (x + 4 <= w) {
            acc8 = sad8_acc(acc8, load_u8_4(src + x), load_u8_4(ref + x));
            x += 4;
        }
        for (; x < w; x++) tail += EB_ABS_DIFF(src[x], ref[x]);
        src += src_stride;
        ref += ref_stride;
    }
    return vaddvq_u32(acc) + vaddv_u32(acc8) + tail;
}

// SADs of a w x h block at the 4 horizontal reference positions ref + 0..3
ALWAYS_INLINE uint32x4_t sad_wxh_x4(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride,
                                    uint32_t w, uint32_t h) {
    uint32x4_t a0 = vdupq_n_u32(0), a1 = vdupq_n_u32(0), a2 = vdupq_n_u32(0), a3 = vdupq_n_u32(0);
    uint32x2_t b0 = vdup_n_u32(0), b1 = vdup_n_u32(0), b2 = vdup_n_u32(0), b3 = vdup_n_u32(0);
    uint32_t   t[4] = {0, 0, 0, 0};
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 16 <= w; x += 16) {
            const uint8x16_t s = vld1q_u8(src + x);
            a0                 = sad16_acc(a0, s, vld1q_u8(ref + x + 0));
            a1                 = sad16_acc(a1, s, vld1q_u8(ref + x + 1));
            a2                 = sad16_acc(a2, s, vld1q_u8(ref + x + 2));
            a3                 = sad16_acc(a3, s, vld1q_u8(ref + x + 3));
        }
        if (x + 8 <= w) {
            const uint8x8_t s = vld1_u8(src + x);
            b0                = sad8_acc(b0, s, vld1_u8(ref + x + 0));
            b1                = sad8_acc(b1, s, vld1_u8(ref + x + 1));
            b2                = sad8_acc(b2, s, vld1_u8(ref + x + 2));
            b3                = sad8_acc(b3, s, vld1_u8(ref + x + 3));
            x += 8;
        }
        if (x + 4 <= w) {
            const uint8x8_t s = load_u8_4(src + x);
            b0                = sad8_acc(b0, s, load_u8_4(ref + x + 0));
            b1                = sad8_acc(b1, s, load_u8_4(ref + x + 1));
            b2                = sad8_acc(b2, s, load_u8_4(ref + x + 2));
            b3                = sad8_acc(b3, s, load_u8_4(ref + x + 3));
            x += 4;
        }
        for (; x < w; x++)
            for (int p = 0; p < 4; p++) t[p] += EB_ABS_DIFF(src[x], ref[x + p]);
        src += src_stride;
        ref += ref_stride;
    }
    uint32x4_t sum = vpaddq_u32(vpaddq_u32(a0, a1), vpaddq_u32(a2, a3));
    sum            = vaddq_u32(sum, vcombine_u32(vpadd_u32(b0, b1), vpadd_u32(b2, b3)));
    return vaddq_u32(sum, vld1q_u32(t));
}

/*******************************************
 * NxM SAD (MD fast loop / ME refinement)
 *******************************************/
#define NXM_SAD(w)                                                                                              \
    uint32_t eb_vp9_compute##w##x_m_sad_neon(                                                                   \
        uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width) { \
        (void)width;                                                                                            \
        return sad_wxh(src, src_stride, ref, ref_stride, w, height);                                            \
    }
NXM_SAD(4)
NXM_SAD(8)
NXM_SAD(16)
NXM_SAD(24)
NXM_SAD(32)
NXM_SAD(48)
NXM_SAD(64)

uint32_t eb_vp9_fast_loop_nx_m_sad_kernel_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                               uint32_t height, uint32_t width) {
    return sad_wxh(src, src_stride, ref, ref_stride, width, height);
}

/*******************************************
 * SAD against the rounded average of two references
 *******************************************/
uint32_t eb_vp9_combined_averaging_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref1, uint32_t ref1_stride,
                                            uint8_t *ref2, uint32_t ref2_stride, uint32_t height, uint32_t width) {
    uint32x4_t acc  = vdupq_n_u32(0);
    uint32x2_t acc8 = vdup_n_u32(0);
    uint32_t   tail = 0;
    for (uint32_t y = 0; y < height; y++) {
        uint32_t x = 0;
        for (; x + 16 <= width; x += 16)
            acc = sad16_acc(acc, vld1q_u8(src + x), vrhaddq_u8(vld1q_u8(ref1 + x), vld1q_u8(ref2 + x)));
        if (x + 8 <= width) {
            acc8 = sad8_acc(acc8, vld1_u8(src + x), vrhadd_u8(vld1_u8(ref1 + x), vld1_u8(ref2 + x)));
            x += 8;
        }
        if (x + 4 <= width) {
            acc8 = sad8_acc(acc8, load_u8_4(src + x), vrhadd_u8(load_u8_4(ref1 + x), load_u8_4(ref2 + x)));
            x += 4;
        }
        for (; x < width; x++) {
            const uint8_t avg = (uint8_t)((ref1[x] + ref2[x] + 1) >> 1);
            tail += EB_ABS_DIFF(src[x], avg);
        }
        src += src_stride;
        ref1 += ref1_stride;
        ref2 += ref2_stride;
    }
    return vaddvq_u32(acc) + vaddv_u32(acc8) + tail;
}

/*******************************************
 * Full search: best SAD over a search area (HME levels)
 *******************************************/
// Updates the best SAD with the 4 consecutive positions x..x+3 of row y. A position only replaces the best when
// strictly smaller, and among equal SADs the leftmost wins, which is the C kernel's raster-order behaviour.
ALWAYS_INLINE void update_best4(uint32x4_t sad4, uint64_t *best_sad, int16_t *x_center, int16_t *y_center, int x,
                                int y) {
    const uint32_t m = vminvq_u32(sad4);
    if (m < *best_sad) {
        uint32_t s[4];
        vst1q_u32(s, sad4);
        int i = 0;
        while (s[i] != m) i++;
        *best_sad = m;
        *x_center = (int16_t)(x + i);
        *y_center = (int16_t)y;
    }
}

ALWAYS_INLINE void sad_loop(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t h,
                            uint32_t w, uint64_t *best_sad, int16_t *x_center, int16_t *y_center,
                            uint32_t src_stride_raw, int16_t search_area_width, int16_t search_area_height) {
    *best_sad = 0xffffff;
    for (int y = 0; y < search_area_height; y++) {
        int x = 0;
        for (; x + 4 <= search_area_width; x += 4)
            update_best4(sad_wxh_x4(src, src_stride, ref + x, ref_stride, w, h), best_sad, x_center, y_center, x, y);
        for (; x < search_area_width; x++) {
            const uint32_t sad = sad_wxh(src, src_stride, ref + x, ref_stride, w, h);
            if (sad < *best_sad) {
                *best_sad = sad;
                *x_center = (int16_t)x;
                *y_center = (int16_t)y;
            }
        }
        ref += src_stride_raw;
    }
}

void eb_vp9_sad_loop_kernel_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t height,
                                 uint32_t width, uint64_t *best_sad, int16_t *x_search_center,
                                 int16_t *y_search_center, uint32_t src_stride_raw, int16_t search_area_width,
                                 int16_t search_area_height) {
    // Specialize the common HME block widths so the row loops are fully unrolled
    switch (width) {
    case 8:
        sad_loop(src, src_stride, ref, ref_stride, height, 8, best_sad, x_search_center, y_search_center,
                 src_stride_raw, search_area_width, search_area_height);
        break;
    case 16:
        sad_loop(src, src_stride, ref, ref_stride, height, 16, best_sad, x_search_center, y_search_center,
                 src_stride_raw, search_area_width, search_area_height);
        break;
    case 32:
        sad_loop(src, src_stride, ref, ref_stride, height, 32, best_sad, x_search_center, y_search_center,
                 src_stride_raw, search_area_width, search_area_height);
        break;
    case 64:
        sad_loop(src, src_stride, ref, ref_stride, height, 64, best_sad, x_search_center, y_search_center,
                 src_stride_raw, search_area_width, search_area_height);
        break;
    default:
        sad_loop(src, src_stride, ref, ref_stride, height, width, best_sad, x_search_center, y_search_center,
                 src_stride_raw, search_area_width, search_area_height);
        break;
    }
}

/*******************************************
 * 8 horizontal search points for the 4 8x8 blocks and the 16x16 block of a 16x16 PU
 * (8x8 SADs use every other row, as Subsad8x8 in C)
 *******************************************/

// SADs of two side-by-side 8x8 blocks (rows 0, 2, 4, 6 of src) at the 8 positions ref + 0..7.
// out_l[k] / out_r[k]: SAD of the left / right 8x8 block at position k.
ALWAYS_INLINE void sad_2x8x8_sub_x8(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride,
                                    uint16x8_t *out_l, uint16x8_t *out_r) {
    uint16x8_t acc[8];
    for (int k = 0; k < 8; k++) acc[k] = vdupq_n_u16(0);
    for (int y = 0; y < 4; y++) {
        const uint8x16_t s  = vld1q_u8(src + 2 * y * src_stride);
        const uint8x16_t r0 = vld1q_u8(ref + 2 * y * ref_stride); // bytes 0..15
        // bytes 16..22 (the last needed byte is ref[22]: never read past it)
        const uint8x16_t r1 = vextq_u8(vld1q_u8(ref + 2 * y * ref_stride + 7), vdupq_n_u8(0), 9);
        acc[0]              = vpadalq_u8(acc[0], vabdq_u8(s, r0));
        acc[1]              = vpadalq_u8(acc[1], vabdq_u8(s, vextq_u8(r0, r1, 1)));
        acc[2]              = vpadalq_u8(acc[2], vabdq_u8(s, vextq_u8(r0, r1, 2)));
        acc[3]              = vpadalq_u8(acc[3], vabdq_u8(s, vextq_u8(r0, r1, 3)));
        acc[4]              = vpadalq_u8(acc[4], vabdq_u8(s, vextq_u8(r0, r1, 4)));
        acc[5]              = vpadalq_u8(acc[5], vabdq_u8(s, vextq_u8(r0, r1, 5)));
        acc[6]              = vpadalq_u8(acc[6], vabdq_u8(s, vextq_u8(r0, r1, 6)));
        acc[7]              = vpadalq_u8(acc[7], vabdq_u8(s, vextq_u8(r0, r1, 7)));
    }
    // acc[k] lanes 0..3: left block partial sums, lanes 4..7: right block
    const uint16x8_t p01   = vpaddq_u16(acc[0], acc[1]);
    const uint16x8_t p23   = vpaddq_u16(acc[2], acc[3]);
    const uint16x8_t p45   = vpaddq_u16(acc[4], acc[5]);
    const uint16x8_t p67   = vpaddq_u16(acc[6], acc[7]);
    const uint16x8_t q0123 = vpaddq_u16(p01, p23); // L0 R0 L1 R1 L2 R2 L3 R3
    const uint16x8_t q4567 = vpaddq_u16(p45, p67); // L4 R4 ... L7 R7
    *out_l                 = vuzp1q_u16(q0123, q4567);
    *out_r                 = vuzp2q_u16(q0123, q4567);
}

ALWAYS_INLINE void update_best8(uint16x8_t sad8, uint32_t *p_best_sad, uint32_t *p_best_mv, uint32_t mv) {
    const uint32_t m = vminvq_u16(sad8);
    if (2 * m < *p_best_sad) {
        uint16_t s[8];
        vst1q_u16(s, sad8);
        int k = 0;
        while (s[k] != m) k++;
        const int16_t x_mv = (int16_t)(_MVXT(mv) + (int16_t)k * 4);
        const int16_t y_mv = _MVYT(mv);
        *p_best_sad        = 2 * m;
        *p_best_mv         = ((uint32_t)(uint16_t)y_mv << 16) | ((uint16_t)x_mv);
    }
}

void eb_vp9_get_eight_horizontal_search_point_results_8x8_16x16_pu_neon(uint8_t *src, uint32_t src_stride,
                                                                        uint8_t *ref, uint32_t ref_stride,
                                                                        uint32_t *p_best_sad8x8, uint32_t *p_best_mv8x8,
                                                                        uint32_t *p_best_sad16x16,
                                                                        uint32_t *p_best_mv16x16, uint32_t mv,
                                                                        uint16_t *p_sad16x16) {
    uint16x8_t s0, s1, s2, s3;
    sad_2x8x8_sub_x8(src, src_stride, ref, ref_stride, &s0, &s1);
    sad_2x8x8_sub_x8(src + 8 * src_stride, src_stride, ref + 8 * ref_stride, ref_stride, &s2, &s3);
    const uint16x8_t s16 = vaddq_u16(vaddq_u16(s0, s1), vaddq_u16(s2, s3));
    vst1q_u16(p_sad16x16, s16);

    // Each block's sequential "if (2 * sad < best)" update over the 8 positions keeps the first minimum
    update_best8(s0, &p_best_sad8x8[0], &p_best_mv8x8[0], mv);
    update_best8(s1, &p_best_sad8x8[1], &p_best_mv8x8[1], mv);
    update_best8(s2, &p_best_sad8x8[2], &p_best_mv8x8[2], mv);
    update_best8(s3, &p_best_sad8x8[3], &p_best_mv8x8[3], mv);
    update_best8(s16, &p_best_sad16x16[0], &p_best_mv16x16[0], mv);
}
