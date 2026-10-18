/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON loop filters, bit-exact with loopfilter.c.
// All pixels of an edge segment are filtered in parallel (16 lanes; the 8-pixel versions duplicate them), and the
// per-pixel C branches (filter4 / filter8 / filter16) become per-lane selects.
// - abs(p0 - q0) * 2 + abs(p1 - q1) / 2 > blimit can exceed 8 bits: it is evaluated in 16 bits;
// - filter + 3 * (qs0 - ps0) is evaluated in 16 bits and clamped once, as in C;
// - the other signed_char_clamp() steps are 8-bit saturating adds / subtracts;
// - the 7- and 15-tap sums are 16-bit running sums, rounded with vrshrn.
// Vertical edges are transposed in registers; only bytes inside the C read footprint are rewritten (unchanged
// bytes are written back as they were).

#include "EbNeonCommon.h"
#include "vpx_dsp_rtcd.h"

// v[0..15] = p7 .. p0, q0 .. q7
#define P(k) v[7 - (k)]
#define Q(k) v[8 + (k)]

ALWAYS_INLINE uint8x16_t max3(uint8x16_t a, uint8x16_t b, uint8x16_t c) { return vmaxq_u8(vmaxq_u8(a, b), c); }

// 16-bit sum helpers on 16 lanes (lo / hi halves)
typedef struct {
    uint16x8_t lo, hi;
} U16x16;

ALWAYS_INLINE U16x16 widen(uint8x16_t a) { return (U16x16){vmovl_u8(vget_low_u8(a)), vmovl_high_u8(a)}; }
ALWAYS_INLINE U16x16 add_u8(U16x16 s, uint8x16_t a) {
    return (U16x16){vaddw_u8(s.lo, vget_low_u8(a)), vaddw_high_u8(s.hi, a)};
}
ALWAYS_INLINE U16x16 sub_u8(U16x16 s, uint8x16_t a) {
    return (U16x16){vsubw_u8(s.lo, vget_low_u8(a)), vsubw_high_u8(s.hi, a)};
}
// s - a - b + c + d
ALWAYS_INLINE U16x16 slide(U16x16 s, uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d) {
    return add_u8(add_u8(sub_u8(sub_u8(s, a), b), c), d);
}
ALWAYS_INLINE uint8x16_t round3(U16x16 s) { return vcombine_u8(vrshrn_n_u16(s.lo, 3), vrshrn_n_u16(s.hi, 3)); }
ALWAYS_INLINE uint8x16_t round4(U16x16 s) { return vcombine_u8(vrshrn_n_u16(s.lo, 4), vrshrn_n_u16(s.hi, 4)); }

// mode 4: filter4, 8: filter8 (else filter4), 16: filter16 (else filter8 / filter4)
ALWAYS_INLINE void lpf_core(uint8x16_t *v, uint8x16_t blimit, uint8x16_t limit, uint8x16_t thresh, int mode) {
    const uint8x16_t p3 = P(3), p2 = P(2), p1 = P(1), p0 = P(0), q0 = Q(0), q1 = Q(1), q2 = Q(2), q3 = Q(3);
    const uint8x16_t ad_p1p0 = vabdq_u8(p1, p0), ad_q1q0 = vabdq_u8(q1, q0);

    // filter_mask
    const uint8x16_t m   = vmaxq_u8(max3(vabdq_u8(p3, p2), vabdq_u8(p2, p1), ad_p1p0),
                                  max3(ad_q1q0, vabdq_u8(q2, q1), vabdq_u8(q3, q2)));
    const uint8x16_t a   = vabdq_u8(p0, q0), b = vshrq_n_u8(vabdq_u8(p1, q1), 1);
    const uint16x8_t blo = vaddw_u8(vshll_n_u8(vget_low_u8(a), 1), vget_low_u8(b));
    const uint16x8_t bhi = vaddw_high_u8(vshll_high_n_u8(a, 1), b);
    const uint8x16_t bok = vcombine_u8(vmovn_u16(vcleq_u16(blo, vmovl_u8(vget_low_u8(blimit)))),
                                       vmovn_u16(vcleq_u16(bhi, vmovl_high_u8(blimit))));
    const uint8x16_t mask = vandq_u8(vcleq_u8(m, limit), bok);

    // filter4
    const uint8x16_t hev = vorrq_u8(vcgtq_u8(ad_p1p0, thresh), vcgtq_u8(ad_q1q0, thresh));
    const int8x16_t  sgn = vdupq_n_s8((int8_t)0x80);
    const int8x16_t  ps1 = veorq_s8(vreinterpretq_s8_u8(p1), sgn), ps0 = veorq_s8(vreinterpretq_s8_u8(p0), sgn);
    const int8x16_t  qs0 = veorq_s8(vreinterpretq_s8_u8(q0), sgn), qs1 = veorq_s8(vreinterpretq_s8_u8(q1), sgn);
    int8x16_t        filter = vandq_s8(vqsubq_s8(ps1, qs1), vreinterpretq_s8_u8(hev));
    int16x8_t        flo    = vmlaq_n_s16(vmovl_s8(vget_low_s8(filter)), vsubl_s8(vget_low_s8(qs0), vget_low_s8(ps0)), 3);
    int16x8_t        fhi    = vmlaq_n_s16(vmovl_high_s8(filter), vsubl_high_s8(qs0, ps0), 3);
    filter                  = vandq_s8(vcombine_s8(vqmovn_s16(flo), vqmovn_s16(fhi)), vreinterpretq_s8_u8(mask));
    const int8x16_t filter1 = vshrq_n_s8(vqaddq_s8(filter, vdupq_n_s8(4)), 3);
    const int8x16_t filter2 = vshrq_n_s8(vqaddq_s8(filter, vdupq_n_s8(3)), 3);
    const int8x16_t f       = vbicq_s8(vrshrq_n_s8(filter1, 1), vreinterpretq_s8_u8(hev));
    const uint8x16_t f4_q0  = vreinterpretq_u8_s8(veorq_s8(vqsubq_s8(qs0, filter1), sgn));
    const uint8x16_t f4_p0  = vreinterpretq_u8_s8(veorq_s8(vqaddq_s8(ps0, filter2), sgn));
    const uint8x16_t f4_q1  = vreinterpretq_u8_s8(veorq_s8(vqsubq_s8(qs1, f), sgn));
    const uint8x16_t f4_p1  = vreinterpretq_u8_s8(veorq_s8(vqaddq_s8(ps1, f), sgn));
    if (mode == 4) {
        P(1) = f4_p1;
        P(0) = f4_p0;
        Q(0) = f4_q0;
        Q(1) = f4_q1;
        return;
    }

    // filter8 where flat && mask
    const uint8x16_t one  = vdupq_n_u8(1);
    const uint8x16_t flat = vandq_u8(
        vcleq_u8(vmaxq_u8(max3(ad_p1p0, ad_q1q0, vabdq_u8(p2, p0)), max3(vabdq_u8(q2, q0), vabdq_u8(p3, p0), vabdq_u8(q3, q0))),
                 one),
        mask);
    uint8x16_t o[6] = {p2, f4_p1, f4_p0, f4_q0, f4_q1, q2}; // p2 .. q2
    if (vmaxvq_u8(flat)) {
        U16x16 s = add_u8(add_u8(add_u8(add_u8(add_u8(add_u8(widen(p3), p3), p3), p2), p2), p1), p0);
        s        = add_u8(s, q0);
        uint8x16_t f8[6];
        f8[0] = round3(s);
        s     = slide(s, p3, p2, p1, q1);
        f8[1] = round3(s);
        s     = slide(s, p3, p1, p0, q2);
        f8[2] = round3(s);
        s     = slide(s, p3, p0, q0, q3);
        f8[3] = round3(s);
        s     = slide(s, p2, q0, q1, q3);
        f8[4] = round3(s);
        s     = slide(s, p1, q1, q2, q3);
        f8[5] = round3(s);
        for (int k = 0; k < 6; k++) o[k] = vbslq_u8(flat, f8[k], o[k]);
    }
    if (mode == 16) {
        // filter16 where flat2 && flat && mask
        const uint8x16_t flat2 = vandq_u8(vcleq_u8(vmaxq_u8(max3(vabdq_u8(P(4), p0), vabdq_u8(P(5), p0), vabdq_u8(P(6), p0)),
                                                            vmaxq_u8(max3(vabdq_u8(P(7), p0), vabdq_u8(Q(4), q0), vabdq_u8(Q(5), q0)),
                                                                     vmaxq_u8(vabdq_u8(Q(6), q0), vabdq_u8(Q(7), q0)))),
                                                   one),
                                          flat);
        if (vmaxvq_u8(flat2)) {
            const uint8x16_t p7 = P(7);
            const uint8x16_t q7 = Q(7);
            // op6 = 7 * p7 + 2 * p6 + p5 + ... + q0
            U16x16 s0 = {vmull_u8(vget_low_u8(p7), vdup_n_u8(7)), vmull_high_u8(p7, vdupq_n_u8(7))};
            s0 = add_u8(add_u8(s0, P(6)), P(6));
            for (int k = 5; k >= 0; k--) s0 = add_u8(s0, P(k));
            s0 = add_u8(s0, q0);
            uint8x16_t f16[14]; // op6 .. op0, oq0 .. oq6
            f16[0] = round4(s0);
            for (int k = 1; k < 7; k++) { // op5 .. op0: - p7 - p(7-k) + p(6-k) + q(k)
                s0     = slide(s0, p7, P(7 - k), P(6 - k), Q(k));
                f16[k] = round4(s0);
            }
            s0     = slide(s0, p7, p0, q0, q7); // oq0
            f16[7] = round4(s0);
            for (int k = 1; k < 7; k++) { // oq1 .. oq6: - p(7-k) - q(k-1) + q(k) + q7
                s0         = slide(s0, P(7 - k), Q(k - 1), Q(k), q7);
                f16[7 + k] = round4(s0);
            }
            for (int k = 0; k < 4; k++) {
                P(6 - k) = vbslq_u8(flat2, f16[k], P(6 - k));
                Q(3 + k) = vbslq_u8(flat2, f16[10 + k], Q(3 + k));
            }
            for (int k = 0; k < 6; k++) o[k] = vbslq_u8(flat2, f16[4 + k], o[k]);
        }
    }
    P(2) = o[0];
    P(1) = o[1];
    P(0) = o[2];
    Q(0) = o[3];
    Q(1) = o[4];
    Q(2) = o[5];
}

ALWAYS_INLINE uint8x16_t dup2(const uint8_t *a, const uint8_t *b) { return vcombine_u8(vdup_n_u8(*a), vdup_n_u8(*b)); }

/*******************************************
 * Horizontal edges (filter across rows)
 *******************************************/
// n = 8 or 16 pixels, taps = 4 (rows p3..q3) or 8 (rows p7..q7)
ALWAYS_INLINE void lpf_horizontal(uint8_t *s, int pitch, uint8x16_t blimit, uint8x16_t limit, uint8x16_t thresh,
                                  int mode, int n) {
    uint8x16_t v[16];
    const int  taps = mode == 16 ? 8 : 4;
    for (int k = 8 - taps; k < 8 + taps; k++) {
        const uint8_t *row = s + (k - 8) * pitch;
        // 8 pixels: duplicated into the high half so that the flat / flat2 early-outs see only real pixels
        v[k] = n == 16 ? vld1q_u8(row) : vcombine_u8(vld1_u8(row), vld1_u8(row));
    }
    lpf_core(v, blimit, limit, thresh, mode);
    const int w = mode == 16 ? 7 : mode == 8 ? 3 : 2; // rows written: p(w-1) .. q(w-1)
    for (int k = 8 - w; k < 8 + w; k++) {
        uint8_t *row = s + (k - 8) * pitch;
        if (n == 16)
            vst1q_u8(row, v[k]);
        else
            vst1_u8(row, vget_low_u8(v[k]));
    }
}

void eb_vp9_lpf_horizontal_4_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                  const uint8_t *thresh) {
    lpf_horizontal(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 4, 8);
}
void eb_vp9_lpf_horizontal_4_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
                                       const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
                                       const uint8_t *thresh1) {
    lpf_horizontal(s, pitch, dup2(blimit0, blimit1), dup2(limit0, limit1), dup2(thresh0, thresh1), 4, 16);
}
void eb_vp9_lpf_horizontal_8_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                  const uint8_t *thresh) {
    lpf_horizontal(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 8, 8);
}
void eb_vp9_lpf_horizontal_8_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
                                       const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
                                       const uint8_t *thresh1) {
    lpf_horizontal(s, pitch, dup2(blimit0, blimit1), dup2(limit0, limit1), dup2(thresh0, thresh1), 8, 16);
}
void eb_vp9_lpf_horizontal_16_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                   const uint8_t *thresh) {
    lpf_horizontal(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 16, 8);
}
void eb_vp9_lpf_horizontal_16_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                        const uint8_t *thresh) {
    lpf_horizontal(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 16, 16);
}

/*******************************************
 * Vertical edges (filter across columns)
 *******************************************/
// Transposes 8 vectors as two independent 8x8 byte blocks (low and high halves)
ALWAYS_INLINE void transpose8x8_x2(uint8x16_t *r) {
    const uint8x16_t a0 = vtrn1q_u8(r[0], r[1]), a1 = vtrn2q_u8(r[0], r[1]);
    const uint8x16_t a2 = vtrn1q_u8(r[2], r[3]), a3 = vtrn2q_u8(r[2], r[3]);
    const uint8x16_t a4 = vtrn1q_u8(r[4], r[5]), a5 = vtrn2q_u8(r[4], r[5]);
    const uint8x16_t a6 = vtrn1q_u8(r[6], r[7]), a7 = vtrn2q_u8(r[6], r[7]);
#define U16(x) vreinterpretq_u16_u8(x)
#define U32(x) vreinterpretq_u32_u16(x)
    const uint16x8_t b0 = vtrn1q_u16(U16(a0), U16(a2)), b2 = vtrn2q_u16(U16(a0), U16(a2));
    const uint16x8_t b1 = vtrn1q_u16(U16(a1), U16(a3)), b3 = vtrn2q_u16(U16(a1), U16(a3));
    const uint16x8_t b4 = vtrn1q_u16(U16(a4), U16(a6)), b6 = vtrn2q_u16(U16(a4), U16(a6));
    const uint16x8_t b5 = vtrn1q_u16(U16(a5), U16(a7)), b7 = vtrn2q_u16(U16(a5), U16(a7));
    r[0] = vreinterpretq_u8_u32(vtrn1q_u32(U32(b0), U32(b4)));
    r[4] = vreinterpretq_u8_u32(vtrn2q_u32(U32(b0), U32(b4)));
    r[1] = vreinterpretq_u8_u32(vtrn1q_u32(U32(b1), U32(b5)));
    r[5] = vreinterpretq_u8_u32(vtrn2q_u32(U32(b1), U32(b5)));
    r[2] = vreinterpretq_u8_u32(vtrn1q_u32(U32(b2), U32(b6)));
    r[6] = vreinterpretq_u8_u32(vtrn2q_u32(U32(b2), U32(b6)));
    r[3] = vreinterpretq_u8_u32(vtrn1q_u32(U32(b3), U32(b7)));
    r[7] = vreinterpretq_u8_u32(vtrn2q_u32(U32(b3), U32(b7)));
#undef U16
#undef U32
}

// rows = 8 or 16; mode 4 / 8 use columns s - 4 .. s + 3, mode 16 uses s - 8 .. s + 7
ALWAYS_INLINE void lpf_vertical(uint8_t *s, int pitch, uint8x16_t blimit, uint8x16_t limit, uint8x16_t thresh,
                                int mode, int rows) {
    uint8x16_t v[16];
    // column block b (8 columns starting at s + c0) -> v[8 * (b) + ...]; lanes = rows (rows 8..15 in the high half)
    const int nblk = mode == 16 ? 2 : 1;
    for (int b = 0; b < nblk; b++) {
        const int   c0 = mode == 16 ? (b ? 0 : -8) : -4;
        uint8x16_t *r  = v + (mode == 16 ? 8 * b : 4);
        for (int k = 0; k < 8; k++) {
            const uint8x8_t lo = vld1_u8(s + k * pitch + c0);
            const uint8x8_t hi = rows == 16 ? vld1_u8(s + (k + 8) * pitch + c0) : lo;
            r[k]               = vcombine_u8(lo, hi);
        }
        transpose8x8_x2(r);
    }
    lpf_core(v, blimit, limit, thresh, mode);
    for (int b = 0; b < nblk; b++) {
        const int   c0 = mode == 16 ? (b ? 0 : -8) : -4;
        uint8x16_t *r  = v + (mode == 16 ? 8 * b : 4);
        transpose8x8_x2(r);
        for (int k = 0; k < 8; k++) {
            vst1_u8(s + k * pitch + c0, vget_low_u8(r[k]));
            if (rows == 16)
                vst1_u8(s + (k + 8) * pitch + c0, vget_high_u8(r[k]));
        }
    }
}

void eb_vp9_lpf_vertical_4_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                const uint8_t *thresh) {
    lpf_vertical(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 4, 8);
}
void eb_vp9_lpf_vertical_4_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
                                     const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
                                     const uint8_t *thresh1) {
    lpf_vertical(s, pitch, dup2(blimit0, blimit1), dup2(limit0, limit1), dup2(thresh0, thresh1), 4, 16);
}
void eb_vp9_lpf_vertical_8_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                const uint8_t *thresh) {
    lpf_vertical(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 8, 8);
}
void eb_vp9_lpf_vertical_8_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
                                     const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
                                     const uint8_t *thresh1) {
    lpf_vertical(s, pitch, dup2(blimit0, blimit1), dup2(limit0, limit1), dup2(thresh0, thresh1), 8, 16);
}
void eb_vp9_lpf_vertical_16_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                 const uint8_t *thresh) {
    lpf_vertical(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 16, 8);
}
void eb_vp9_lpf_vertical_16_dual_neon(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                      const uint8_t *thresh) {
    lpf_vertical(s, pitch, vdupq_n_u8(*blimit), vdupq_n_u8(*limit), vdupq_n_u8(*thresh), 16, 16);
}
