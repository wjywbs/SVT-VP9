/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON directional and TM intra predictors (8x8, 16x16, 32x32), bit-exact with intrapred.c. (V, H and DC are
// memset / memcpy loops in C and are not faster in NEON.)
// The directional predictors are written as one border array (computed with AVG2 / AVG3 on vectors) from which
// every row is a shifted copy. The edge pixels are first copied to local padded buffers, reading exactly the
// pixels the C code reads.
// AVG2(a, b) = vrhadd(a, b); AVG3(a, b, c) = (a + 2b + c + 2) >> 2 = vrhadd(vhadd(a, c), b) (exact).

#include <string.h>

#include "EbNeonCommon.h"
#include "vpx_dsp_rtcd.h"
#include "mem.h"

ALWAYS_INLINE uint8x16_t avg3(const uint8x16_t a, const uint8x16_t b, const uint8x16_t c) {
    return vrhaddq_u8(vhaddq_u8(a, c), b);
}

// dst[i] = AVG3(x[i], x[i + 1], x[i + 2]) for i < n (n rounded up to 16; x padded)
ALWAYS_INLINE void avg3_buf(uint8_t *dst, const uint8_t *x, int n) {
    for (int i = 0; i < n; i += 16) vst1q_u8(dst + i, avg3(vld1q_u8(x + i), vld1q_u8(x + i + 1), vld1q_u8(x + i + 2)));
}

// dst[i] = AVG2(x[i], x[i + 1])
ALWAYS_INLINE void avg2_buf(uint8_t *dst, const uint8_t *x, int n) {
    for (int i = 0; i < n; i += 16) vst1q_u8(dst + i, vrhaddq_u8(vld1q_u8(x + i), vld1q_u8(x + i + 1)));
}

ALWAYS_INLINE void fill_buf(uint8_t *dst, uint8_t v, int n) {
    const uint8x16_t d = vdupq_n_u8(v);
    for (int i = 0; i < n; i += 16) vst1q_u8(dst + i, d);
}

// dst[i] = src[n - 1 - i], n multiple of 8
ALWAYS_INLINE void reverse_buf(uint8_t *dst, const uint8_t *src, int n) {
    for (int i = 0; i < n; i += 8) vst1_u8(dst + i, vrev64_u8(vld1_u8(src + n - 8 - i)));
}

ALWAYS_INLINE void copy_row(uint8_t *dst, const uint8_t *src, int bs) {
    if (bs == 8)
        vst1_u8(dst, vld1_u8(src));
    else if (bs == 16)
        vst1q_u8(dst, vld1q_u8(src));
    else
        vst1q_u8_x2(dst, vld1q_u8_x2(src));
}

/*******************************************
 * Directional
 *******************************************/
ALWAYS_INLINE void d45_neon(uint8_t *dst, ptrdiff_t stride, int bs, const uint8_t *above) {
    DECLARE_ALIGNED(16, uint8_t, a[64 + 16]);
    DECLARE_ALIGNED(16, uint8_t, b[96]);
    memcpy(a, above, bs + 1); // above[0..bs]
    avg3_buf(b, a, bs); // b[k] = AVG3(above[k], above[k + 1], above[k + 2]), k < bs - 1
    fill_buf(b + bs - 1, above[bs - 1], bs + 16); // above_right from k = bs - 1
    for (int r = 0; r < bs; r++) copy_row(dst + r * stride, b + r, bs);
}

ALWAYS_INLINE void d63_neon(uint8_t *dst, ptrdiff_t stride, int bs, const uint8_t *above) {
    DECLARE_ALIGNED(16, uint8_t, a[64 + 16]);
    DECLARE_ALIGNED(16, uint8_t, b0[80]);
    DECLARE_ALIGNED(16, uint8_t, b1[80]);
    memcpy(a, above, bs + 2); // above[0..bs + 1]
    avg2_buf(b0, a, bs);
    avg3_buf(b1, a, bs);
    copy_row(dst, b0, bs);
    copy_row(dst + stride, b1, bs);
    // rows 2j / 2j + 1 (j >= 1): b0 / b1 from j, padded with above[bs - 1] from index bs - 1
    fill_buf(b0 + bs - 1, above[bs - 1], bs);
    fill_buf(b1 + bs - 1, above[bs - 1], bs);
    for (int j = 1; j < bs / 2; j++) {
        copy_row(dst + (2 * j) * stride, b0 + j, bs);
        copy_row(dst + (2 * j + 1) * stride, b1 + j, bs);
    }
}

ALWAYS_INLINE void d207_neon(uint8_t *dst, ptrdiff_t stride, int bs, const uint8_t *left) {
    DECLARE_ALIGNED(16, uint8_t, l[64 + 16]);
    DECLARE_ALIGNED(16, uint8_t, c0[48]);
    DECLARE_ALIGNED(16, uint8_t, c1[48]);
    DECLARE_ALIGNED(16, uint8_t, e[128]);
    memcpy(l, left, bs);
    fill_buf(l + bs, left[bs - 1], 16); // AVG2 / AVG3 past the end use left[bs - 1]
    avg2_buf(c0, l, bs); // first column
    avg3_buf(c1, l, bs); // second column
    // e[2k] = c0[k], e[2k + 1] = c1[k]; row r = e[2r ..]
    for (int k = 0; k < bs; k += 16) {
        const uint8x16x2_t z = vzipq_u8(vld1q_u8(c0 + k), vld1q_u8(c1 + k));
        vst1q_u8(e + 2 * k, z.val[0]);
        vst1q_u8(e + 2 * k + 16, z.val[1]);
    }
    fill_buf(e + 2 * bs, left[bs - 1], bs);
    for (int r = 0; r < bs; r++) copy_row(dst + r * stride, e + 2 * r, bs);
}

ALWAYS_INLINE void d153_neon(uint8_t *dst, ptrdiff_t stride, int bs, const uint8_t *above, const uint8_t *left) {
    DECLARE_ALIGNED(16, uint8_t, x[64 + 16]);
    DECLARE_ALIGNED(16, uint8_t, a[64 + 16]);
    DECLARE_ALIGNED(16, uint8_t, c0[48]);
    DECLARE_ALIGNED(16, uint8_t, c1[48]);
    DECLARE_ALIGNED(16, uint8_t, r0[48]);
    DECLARE_ALIGNED(16, uint8_t, r1[48]);
    DECLARE_ALIGNED(16, uint8_t, t[128]);
    // x = {above[0], above[-1], left[0..bs-1]}: c1[r] = AVG3(x[r], x[r + 1], x[r + 2]), c0[r] = AVG2(x[r + 1], x[r + 2])
    x[0] = above[0];
    x[1] = above[-1];
    memcpy(x + 2, left, bs);
    avg2_buf(c0, x + 1, bs);
    avg3_buf(c1, x, bs);
    // top row from column 2: AVG3(above[c - 1], above[c], above[c + 1]), c < bs - 2
    memcpy(a, above - 1, bs); // above[-1..bs-2]
    // t[2bs - 2 - 2k] = c0[k], t[2bs - 1 - 2k] = c1[k]; t[2bs + c] = top[c]; row r = t[2bs - 2 - 2r ..]
    reverse_buf(r0, c0, bs);
    reverse_buf(r1, c1, bs);
    for (int k = 0; k < bs; k += 16) {
        const uint8x16x2_t z = vzipq_u8(vld1q_u8(r0 + k), vld1q_u8(r1 + k));
        vst1q_u8(t + 2 * k, z.val[0]);
        vst1q_u8(t + 2 * k + 16, z.val[1]);
    }
    avg3_buf(t + 2 * bs, a, bs - 2);
    for (int r = 0; r < bs; r++) copy_row(dst + r * stride, t + 2 * bs - 2 - 2 * r, bs);
}

ALWAYS_INLINE void d117_neon(uint8_t *dst, ptrdiff_t stride, int bs, const uint8_t *above, const uint8_t *left) {
    DECLARE_ALIGNED(16, uint8_t, y[64 + 16]);
    DECLARE_ALIGNED(16, uint8_t, z[64 + 16]);
    DECLARE_ALIGNED(16, uint8_t, col[48]);
    DECLARE_ALIGNED(16, uint8_t, b0[96]);
    DECLARE_ALIGNED(16, uint8_t, b1[96]);
    const int off = bs / 2;
    // y = {left[0], above[-1], above[0..bs-1]}: row1[c] = AVG3(y[c], y[c + 1], y[c + 2]), row0[c] = AVG2(y[c + 1], y[c + 2])
    y[0] = left[0];
    memcpy(y + 1, above - 1, bs + 1);
    avg2_buf(b0 + off, y + 1, bs);
    avg3_buf(b1 + off, y, bs);
    // z = {above[-1], left[0..bs-2]}: first column of row r >= 2 = AVG3(z[r - 2], z[r - 1], z[r])
    z[0] = above[-1];
    memcpy(z + 1, left, bs - 1);
    avg3_buf(col, z, bs - 2); // col[r - 2]
    // row 2m + p, column c: c >= m -> row_p[c - m]; c < m -> first column of row 2(m - c) + p
    for (int k = 1; k < off; k++) {
        b0[off - k] = col[2 * k - 2];
        b1[off - k] = col[2 * k - 1];
    }
    for (int m = 0; m < off; m++) {
        copy_row(dst + (2 * m) * stride, b0 + off - m, bs);
        copy_row(dst + (2 * m + 1) * stride, b1 + off - m, bs);
    }
}

ALWAYS_INLINE void d135_neon(uint8_t *dst, ptrdiff_t stride, int bs, const uint8_t *above, const uint8_t *left) {
    DECLARE_ALIGNED(16, uint8_t, w[128]);
    DECLARE_ALIGNED(16, uint8_t, border[96]);
    // w = {left[bs-1..0], above[-1..bs-1]}; border[k] = AVG3(w[k], w[k + 1], w[k + 2]); row i = border[bs - 1 - i ..]
    reverse_buf(w, left, bs);
    memcpy(w + bs, above - 1, bs + 1);
    avg3_buf(border, w, 2 * bs - 1);
    for (int i = 0; i < bs; i++) copy_row(dst + i * stride, border + bs - 1 - i, bs);
}

/*******************************************
 * TM
 *******************************************/
ALWAYS_INLINE void tm_neon(uint8_t *dst, ptrdiff_t stride, int bs, const uint8_t *above, const uint8_t *left) {
    const uint8x8_t tl = vdup_n_u8(above[-1]);
    int16x8_t       d[4]; // above[c] - above[-1]
    for (int c = 0; c < bs; c += 8) d[c / 8] = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(above + c), tl));
    for (int r = 0; r < bs; r++) {
        const int16x8_t l = vdupq_n_s16(left[r]);
        for (int c = 0; c < bs; c += 8) vst1_u8(dst + c, vqmovun_s16(vaddq_s16(d[c / 8], l)));
        dst += stride;
    }
}

/*******************************************
 * Entry points
 *******************************************/
#define INTRA_PRED_NEON(bs)                                                                                         \
    void eb_vp9_d45_predictor_##bs##x##bs##_neon(                                                                   \
        uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left) {                                \
        (void)left;                                                                                                 \
        d45_neon(dst, stride, bs, above);                                                                           \
    }                                                                                                               \
    void eb_vp9_d63_predictor_##bs##x##bs##_neon(                                                                   \
        uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left) {                                \
        (void)left;                                                                                                 \
        d63_neon(dst, stride, bs, above);                                                                           \
    }                                                                                                               \
    void eb_vp9_d207_predictor_##bs##x##bs##_neon(                                                                  \
        uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left) {                                \
        (void)above;                                                                                                \
        d207_neon(dst, stride, bs, left);                                                                           \
    }                                                                                                               \
    void eb_vp9_d153_predictor_##bs##x##bs##_neon(                                                                  \
        uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left) {                                \
        d153_neon(dst, stride, bs, above, left);                                                                    \
    }                                                                                                               \
    void eb_vp9_d117_predictor_##bs##x##bs##_neon(                                                                  \
        uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left) {                                \
        d117_neon(dst, stride, bs, above, left);                                                                    \
    }                                                                                                               \
    void eb_vp9_d135_predictor_##bs##x##bs##_neon(                                                                  \
        uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left) {                                \
        d135_neon(dst, stride, bs, above, left);                                                                    \
    }                                                                                                               \
    void eb_vp9_tm_predictor_##bs##x##bs##_neon(                                                                    \
        uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left) {                                \
        tm_neon(dst, stride, bs, above, left);                                                                      \
    }

INTRA_PRED_NEON(8)
INTRA_PRED_NEON(16)
INTRA_PRED_NEON(32)
