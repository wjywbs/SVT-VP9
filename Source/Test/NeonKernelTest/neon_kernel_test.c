/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

// NEON kernel test: every NEON kernel against its C version.
//   - correctness: random and extreme inputs (all 0 / all 255 / 0-255 checkerboard / flat images for ties),
//     all block sizes, outputs must be bit-identical. Input buffers end at a guard page, so reading past the
//     last byte the C kernel reads crashes the test.
//   - speed: ns per call of C and NEON on typical sizes.
// Usage: SvtVp9NeonKernelTest [filter]   (runs the tests whose name contains filter)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "EbDefinitions.h"
#include "EbComputeSAD_C.h"
#include "EbComputeSAD_NEON.h"
#include "EbPictureOperators_C.h"
#include "EbPictureOperators_NEON.h"
#include "vpx_dsp_rtcd.h"
#include "vp9_filter.h"
#include "vp9_rtcd.h"
#include "vp9_scan.h"

/*******************************************
 * Utilities
 *******************************************/
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}
static int rnd_range(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

// Buffer of 'size' bytes that ends exactly at a PROT_NONE guard page
static void *alloc_guarded(size_t size) {
    const size_t page  = (size_t)getpagesize();
    const size_t total = (size + page - 1) / page * page + page;
    uint8_t     *base  = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (base == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    mprotect(base + total - page, page, PROT_NONE);
    return base + total - page - size;
}

enum { PAT_RANDOM, PAT_ZERO, PAT_MAX, PAT_CHECKER, PAT_FLAT_NOISE, PAT_COUNT };
static void fill_u8(uint8_t *p, size_t n, int pattern) {
    const uint8_t base = (uint8_t)rnd();
    for (size_t i = 0; i < n; i++) {
        switch (pattern) {
        case PAT_RANDOM: p[i] = (uint8_t)rnd(); break;
        case PAT_ZERO: p[i] = 0; break;
        case PAT_MAX: p[i] = 255; break;
        case PAT_CHECKER: p[i] = (i ^ (i >> 7)) & 1 ? 255 : 0; break;
        default: p[i] = (uint8_t)(base + (rnd() % 3)); break; // near flat: many equal SADs
        }
    }
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int g_fail;
#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            if (g_fail < 20) {                   \
                printf("  MISMATCH: " __VA_ARGS__); \
                printf("\n");                    \
            }                                    \
            g_fail++;                            \
        }                                        \
    } while (0)

// Times fn(arg) for both versions; prints ns/call and speedup
#define BENCH(label, iters, call_c, call_neon)                                                            \
    do {                                                                                                  \
        double t0 = now_ns();                                                                             \
        for (int it_ = 0; it_ < (iters); it_++) { call_c; }                                               \
        double t1 = now_ns();                                                                             \
        for (int it_ = 0; it_ < (iters); it_++) { call_neon; }                                            \
        double t2 = now_ns();                                                                             \
        printf("  %-34s C %9.1f ns  NEON %8.1f ns  x%.1f\n", label, (t1 - t0) / (iters), (t2 - t1) / (iters), \
               (t1 - t0) / (t2 - t1));                                                                    \
    } while (0)

static volatile uint64_t sink;

/*******************************************
 * SAD
 *******************************************/
static void test_nxm_sad(void) {
    typedef uint32_t (*Fn)(uint8_t *, uint32_t, uint8_t *, uint32_t, uint32_t, uint32_t);
    static const struct {
        int w;
        Fn  f;
    } k[] = {{4, eb_vp9_compute4x_m_sad_neon},
             {8, eb_vp9_compute8x_m_sad_neon},
             {16, eb_vp9_compute16x_m_sad_neon},
             {24, eb_vp9_compute24x_m_sad_neon},
             {32, eb_vp9_compute32x_m_sad_neon},
             {48, eb_vp9_compute48x_m_sad_neon},
             {64, eb_vp9_compute64x_m_sad_neon},
             {40, eb_vp9_fast_loop_nx_m_sad_kernel_neon},
             {56, eb_vp9_fast_loop_nx_m_sad_kernel_neon},
             {12, eb_vp9_fast_loop_nx_m_sad_kernel_neon},
             {7, eb_vp9_fast_loop_nx_m_sad_kernel_neon}};
    const int stride = 160;
    uint8_t  *src    = alloc_guarded(stride * 64);
    uint8_t  *ref    = alloc_guarded(stride * 64);
    for (size_t i = 0; i < sizeof(k) / sizeof(k[0]); i++) {
        for (int pat = 0; pat < PAT_COUNT; pat++) {
            for (int h = 1; h <= 64; h++) {
                fill_u8(src, stride * 64, pat);
                fill_u8(ref, stride * 64, pat == PAT_CHECKER ? PAT_RANDOM : pat);
                if (pat == PAT_CHECKER)
                    for (int j = 0; j < stride * 64; j++) ref[j] = (uint8_t)~src[j];
                // block placed so its last byte is the last byte of the buffer
                uint8_t *s = src + stride * (64 - h) + stride - k[i].w;
                uint8_t *r = ref + stride * (64 - h) + stride - k[i].w;
                uint32_t a = eb_vp9_fast_loop_nx_m_sad_kernel(s, stride, r, stride, h, k[i].w);
                uint32_t b = k[i].f(s, stride, r, stride, h, k[i].w);
                CHECK(a == b, "nxm_sad w%d h%d pat%d: C %u NEON %u", k[i].w, h, pat, a, b);
            }
        }
    }
    fill_u8(src, stride * 64, PAT_RANDOM);
    fill_u8(ref, stride * 64, PAT_RANDOM);
    BENCH("nxm_sad 8x8", 2000000, sink += eb_vp9_fast_loop_nx_m_sad_kernel(src, stride, ref, stride, 8, 8),
          sink += eb_vp9_compute8x_m_sad_neon(src, stride, ref, stride, 8, 8));
    BENCH("nxm_sad 16x16", 1000000, sink += eb_vp9_fast_loop_nx_m_sad_kernel(src, stride, ref, stride, 16, 16),
          sink += eb_vp9_compute16x_m_sad_neon(src, stride, ref, stride, 16, 16));
    BENCH("nxm_sad 32x32", 300000, sink += eb_vp9_fast_loop_nx_m_sad_kernel(src, stride, ref, stride, 32, 32),
          sink += eb_vp9_compute32x_m_sad_neon(src, stride, ref, stride, 32, 32));
    BENCH("nxm_sad 64x64", 100000, sink += eb_vp9_fast_loop_nx_m_sad_kernel(src, stride, ref, stride, 64, 64),
          sink += eb_vp9_compute64x_m_sad_neon(src, stride, ref, stride, 64, 64));
}

static void test_avg_sad(void) {
    const int stride = 96;
    uint8_t  *src    = alloc_guarded(stride * 64);
    uint8_t  *r1     = alloc_guarded(stride * 64);
    uint8_t  *r2     = alloc_guarded(stride * 64);
    static const int widths[] = {4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 5};
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        const int w = widths[i];
        for (int pat = 0; pat < PAT_COUNT; pat++) {
            for (int h = 1; h <= 64; h += (h < 8 ? 1 : 7)) {
                fill_u8(src, stride * 64, pat);
                fill_u8(r1, stride * 64, pat);
                fill_u8(r2, stride * 64, pat == PAT_CHECKER ? PAT_RANDOM : pat);
                uint8_t *s = src + stride * (64 - h) + stride - w;
                uint8_t *a = r1 + stride * (64 - h) + stride - w;
                uint8_t *b = r2 + stride * (64 - h) + stride - w;
                uint32_t c = eb_vp9_combined_averaging_sad(s, stride, a, stride, b, stride, h, w);
                uint32_t n = eb_vp9_combined_averaging_sad_neon(s, stride, a, stride, b, stride, h, w);
                CHECK(c == n, "avg_sad w%d h%d pat%d: C %u NEON %u", w, h, pat, c, n);
            }
        }
    }
    BENCH("combined_averaging_sad 16x16", 1000000,
          sink += eb_vp9_combined_averaging_sad(src, stride, r1, stride, r2, stride, 16, 16),
          sink += eb_vp9_combined_averaging_sad_neon(src, stride, r1, stride, r2, stride, 16, 16));
}

static void test_sad_loop(void) {
    const int src_stride = 64 * 2, ref_raw = 256;
    const int ref_size   = ref_raw * 2 * 40 + ref_raw * 80;
    uint8_t  *src        = alloc_guarded(src_stride * 34);
    uint8_t  *ref        = alloc_guarded(ref_size);
    static const int widths[] = {4, 8, 12, 16, 20, 24, 32, 48, 64, 6};
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        const int w = widths[i];
        for (int pat = 0; pat < PAT_COUNT; pat++) {
            for (int t = 0; t < 30; t++) {
                const int h   = rnd_range(1, 32);
                const int saw = rnd_range(1, 96), sah = rnd_range(1, 20);
                fill_u8(src, src_stride * 34, pat);
                fill_u8(ref, ref_size, pat);
                // the kernel reads up to ref + (sah - 1) * ref_raw + (h - 1) * 2 * ref_raw + saw - 1 + w - 1
                const int need = (sah - 1) * ref_raw + (h - 1) * 2 * ref_raw + saw - 1 + w;
                uint8_t  *r    = ref + ref_size - need;
                uint64_t  bc, bn;
                int16_t   xc = -1, yc = -1, xn = -1, yn = -1;
                eb_vp9_sad_loop_kernel(src, src_stride, r, ref_raw * 2, h, w, &bc, &xc, &yc, ref_raw, saw, sah);
                eb_vp9_sad_loop_kernel_neon(src, src_stride, r, ref_raw * 2, h, w, &bn, &xn, &yn, ref_raw, saw, sah);
                CHECK(bc == bn && xc == xn && yc == yn,
                      "sad_loop w%d h%d sa %dx%d pat%d: C %llu (%d,%d) NEON %llu (%d,%d)", w, h, saw, sah, pat,
                      (unsigned long long)bc, xc, yc, (unsigned long long)bn, xn, yn);
            }
        }
    }
    fill_u8(src, src_stride * 34, PAT_RANDOM);
    fill_u8(ref, ref_size, PAT_RANDOM);
    uint64_t b;
    int16_t  x, y;
    BENCH("sad_loop 16x8, area 64x32 (HME L0)", 2000,
          eb_vp9_sad_loop_kernel(src, src_stride, ref, ref_raw * 2, 8, 16, &b, &x, &y, ref_raw, 64, 32),
          eb_vp9_sad_loop_kernel_neon(src, src_stride, ref, ref_raw * 2, 8, 16, &b, &x, &y, ref_raw, 64, 32));
    BENCH("sad_loop 64x32, area 16x16 (HME L2)", 2000,
          eb_vp9_sad_loop_kernel(src, src_stride, ref, ref_raw * 2, 32, 64, &b, &x, &y, ref_raw, 16, 16),
          eb_vp9_sad_loop_kernel_neon(src, src_stride, ref, ref_raw * 2, 32, 64, &b, &x, &y, ref_raw, 16, 16));
}

static void test_eight_point(void) {
    const int stride = 64;
    uint8_t  *src    = alloc_guarded(stride * 16);
    uint8_t  *ref    = alloc_guarded(stride * 16);
    for (int pat = 0; pat < PAT_COUNT; pat++) {
        for (int t = 0; t < 2000; t++) {
            fill_u8(src, stride * 16, pat);
            fill_u8(ref, stride * 16, pat);
            // the C kernel reads ref columns 0..22 of rows 0..14
            uint8_t *s = src + stride * 16 - (15 * stride + 16);
            uint8_t *r = ref + stride * 16 - (14 * stride + 23);
            uint32_t bs8_c[4], bm8_c[4], bs16_c, bm16_c, bs8_n[4], bm8_n[4], bs16_n, bm16_n;
            uint16_t s16_c[8], s16_n[8];
            for (int i = 0; i < 4; i++) {
                bs8_c[i] = bs8_n[i] = (t & 1) ? 0xffffffff : rnd() % 4000;
                bm8_c[i] = bm8_n[i] = rnd();
            }
            bs16_c = bs16_n = (t & 2) ? 0xffffffff : rnd() % 16000;
            bm16_c = bm16_n = rnd();
            const uint32_t mv = rnd();
            eb_vp9_get_eight_horizontal_search_point_results_8x8_16x16_pu(
                s, stride, r, stride, bs8_c, bm8_c, &bs16_c, &bm16_c, mv, s16_c);
            eb_vp9_get_eight_horizontal_search_point_results_8x8_16x16_pu_neon(
                s, stride, r, stride, bs8_n, bm8_n, &bs16_n, &bm16_n, mv, s16_n);
            CHECK(!memcmp(bs8_c, bs8_n, sizeof(bs8_c)) && !memcmp(bm8_c, bm8_n, sizeof(bm8_c)) &&
                      bs16_c == bs16_n && bm16_c == bm16_n && !memcmp(s16_c, s16_n, sizeof(s16_c)),
                  "eight_point pat%d t%d", pat, t);
        }
    }
    uint32_t bs8[4] = {~0u, ~0u, ~0u, ~0u}, bm8[4], bs16 = ~0u, bm16;
    uint16_t s16[8];
    BENCH("eight_point_8x8_16x16", 1000000,
          eb_vp9_get_eight_horizontal_search_point_results_8x8_16x16_pu(src, stride, ref, stride, bs8, bm8, &bs16,
                                                                        &bm16, 0, s16),
          eb_vp9_get_eight_horizontal_search_point_results_8x8_16x16_pu_neon(src, stride, ref, stride, bs8, bm8,
                                                                             &bs16, &bm16, 0, s16));
}

/*******************************************
 * Convolution
 *******************************************/
typedef void (*ConvFn)(const uint8_t *, ptrdiff_t, uint8_t *, ptrdiff_t, const InterpKernel *, int, int, int, int,
                       int, int);
enum { DIR_NONE, DIR_H, DIR_V, DIR_2D };
static const struct {
    const char *name;
    ConvFn      c, neon;
    int         dir;
} conv_fns[] = {
    {"convolve8_horiz", eb_vp9_convolve8_horiz_c, eb_vp9_convolve8_horiz_neon, DIR_H},
    {"convolve8_avg_horiz", eb_vp9_convolve8_avg_horiz_c, eb_vp9_convolve8_avg_horiz_neon, DIR_H},
    {"convolve8_vert", eb_vp9_convolve8_vert_c, eb_vp9_convolve8_vert_neon, DIR_V},
    {"convolve8_avg_vert", eb_vp9_convolve8_avg_vert_c, eb_vp9_convolve8_avg_vert_neon, DIR_V},
    {"convolve8", eb_vp9_convolve8_c, eb_vp9_convolve8_neon, DIR_2D},
    {"convolve8_avg", eb_vp9_convolve8_avg_c, eb_vp9_convolve8_avg_neon, DIR_2D},
    {"convolve_avg", eb_vp9_convolve_avg_c, eb_vp9_convolve_avg_neon, DIR_NONE},
    {"convolve_copy", eb_vp9_convolve_copy_c, eb_vp9_convolve_copy_neon, DIR_NONE},
};

static void test_convolve(void) {
    const int stride = 96, size = stride * 80;
    uint8_t  *src    = alloc_guarded(size);
    uint8_t  *dst_c  = alloc_guarded(size);
    uint8_t  *dst_n  = alloc_guarded(size);
    static const int widths[]  = {4, 8, 16, 32, 64, 12, 24, 48};
    static const int heights[] = {1, 2, 3, 4, 7, 8, 16, 32, 64};
    for (size_t f = 0; f < sizeof(conv_fns) / sizeof(conv_fns[0]); f++) {
        const int dir = conv_fns[f].dir;
        for (int kernel = 0; kernel < 4; kernel++) {
            for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
                for (size_t hi = 0; hi < sizeof(heights) / sizeof(heights[0]); hi++) {
                    const int w = widths[wi], h = heights[hi];
                    for (int t = 0; t < 8; t++) {
                        const int pat = t < PAT_COUNT ? t : PAT_RANDOM;
                        const int x0  = dir == DIR_V || dir == DIR_NONE ? 0 : rnd_range(1, 15);
                        const int y0  = dir == DIR_H || dir == DIR_NONE ? 0 : rnd_range(1, 15);
                        fill_u8(src, size, pat);
                        fill_u8(dst_c, size, PAT_RANDOM);
                        memcpy(dst_n, dst_c, size);
                        // the source block is placed so the last byte the C code reads is the last buffer byte
                        const int right = dir == DIR_H || dir == DIR_2D ? 4 : 0;
                        const int below = dir == DIR_V || dir == DIR_2D ? 4 : 0;
                        const uint8_t *s = src + size - ((h - 1 + below) * stride + w + right);
                        const int      d_off = size - ((h - 1) * stride + w);
                        conv_fns[f].c(s, stride, dst_c + d_off, stride, eb_vp9_filter_kernels[kernel], x0, 16, y0, 16, w, h);
                        conv_fns[f].neon(s, stride, dst_n + d_off, stride, eb_vp9_filter_kernels[kernel], x0, 16, y0, 16, w, h);
                        CHECK(!memcmp(dst_c, dst_n, size), "%s kernel%d %dx%d x0 %d y0 %d pat%d", conv_fns[f].name,
                              kernel, w, h, x0, y0, pat);
                    }
                }
            }
        }
    }
    fill_u8(src, size, PAT_RANDOM);
    const uint8_t *s = src + 4 * stride + 8;
    for (size_t f = 0; f < sizeof(conv_fns) / sizeof(conv_fns[0]); f++) {
        char label[64];
        snprintf(label, sizeof(label), "%s 16x16", conv_fns[f].name);
        BENCH(label, 300000, conv_fns[f].c(s, stride, dst_c, stride, eb_vp9_filter_kernels[0], 5, 16, 7, 16, 16, 16),
              conv_fns[f].neon(s, stride, dst_n, stride, eb_vp9_filter_kernels[0], 5, 16, 7, 16, 16, 16));
    }
    BENCH("convolve8 64x64", 30000,
          eb_vp9_convolve8_c(s, stride, dst_c, stride, eb_vp9_filter_kernels[0], 5, 16, 7, 16, 64, 64),
          eb_vp9_convolve8_neon(s, stride, dst_n, stride, eb_vp9_filter_kernels[0], 5, 16, 7, 16, 64, 64));
}

/*******************************************
 * Forward transforms
 *******************************************/
// Residual patterns: random in [-255, 255], extremes (+-255 constant / checkerboard / random sign), small values
static void fill_residual(int16_t *p, int n, int stride, int pattern) {
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            int16_t v;
            switch (pattern % 7) {
            case 0: v = (int16_t)rnd_range(-255, 255); break;
            case 1: v = 255; break;
            case 2: v = -255; break;
            case 3: v = ((r ^ c) & 1) ? 255 : -255; break;
            case 4: v = (rnd() & 1) ? 255 : -255; break;
            case 5: v = (int16_t)rnd_range(-8, 8); break;
            default: v = (r < n / 2) == (c < n / 2) ? 255 : -255; break;
            }
            p[r * stride + c] = v;
        }
}

typedef void (*FdctFn)(const int16_t *, tran_low_t *, int);
static const struct {
    const char *name;
    FdctFn      c, neon;
    int         n;
} fdct_fns[] = {
    {"fdct32x32", eb_vp9_fdct32x32_c, eb_vp9_fdct32x32_neon, 32},
    {"partial_fdct32x32", eb_vpx_partial_fdct32x32_c, eb_vpx_partial_fdct32x32_neon, 32},
    {"fdct16x16", eb_vp9_fdct16x16_c, eb_vp9_fdct16x16_neon, 16},
    {"fdct8x8", eb_vp9_fdct8x8_c, eb_vp9_fdct8x8_neon, 8},
};

typedef void (*FhtFn)(const int16_t *, tran_low_t *, int, int);
static const struct {
    const char *name;
    FhtFn       c, neon;
    int         n;
} fht_fns[] = {
    {"fht16x16", eb_vp9_fht16x16_c, eb_vp9_fht16x16_neon, 16},
    {"fht8x8", eb_vp9_fht8x8_c, eb_vp9_fht8x8_neon, 8},
};

static void test_fdct(void) {
    const int stride = 40;
    int16_t  *in     = alloc_guarded(sizeof(int16_t) * stride * 32);
    for (size_t f = 0; f < sizeof(fdct_fns) / sizeof(fdct_fns[0]); f++) {
        const int  n = fdct_fns[f].n;
        tran_low_t out_c[32 * 32], out_n[32 * 32];
        for (int t = 0; t < 3000; t++) {
            fill_residual(in, n, stride, t);
            memset(out_c, 0x5a, sizeof(out_c));
            memset(out_n, 0x5a, sizeof(out_n));
            fdct_fns[f].c(in, out_c, stride);
            fdct_fns[f].neon(in, out_n, stride);
            CHECK(!memcmp(out_c, out_n, sizeof(out_c)), "%s pattern %d", fdct_fns[f].name, t % 7);
        }
        fill_residual(in, n, stride, 0);
        char label[64];
        snprintf(label, sizeof(label), "%s", fdct_fns[f].name);
        BENCH(label, 200000 / n, fdct_fns[f].c(in, out_c, stride), fdct_fns[f].neon(in, out_n, stride));
    }
    for (size_t f = 0; f < sizeof(fht_fns) / sizeof(fht_fns[0]); f++) {
        const int  n = fht_fns[f].n;
        tran_low_t out_c[32 * 32], out_n[32 * 32];
        for (int tx_type = 0; tx_type < 4; tx_type++) {
            for (int t = 0; t < 2000; t++) {
                fill_residual(in, n, stride, t);
                memset(out_c, 0x5a, sizeof(out_c));
                memset(out_n, 0x5a, sizeof(out_n));
                fht_fns[f].c(in, out_c, stride, tx_type);
                fht_fns[f].neon(in, out_n, stride, tx_type);
                CHECK(!memcmp(out_c, out_n, sizeof(out_c)), "%s tx_type %d pattern %d", fht_fns[f].name, tx_type, t % 7);
            }
            fill_residual(in, n, stride, 0);
            char label[64];
            snprintf(label, sizeof(label), "%s tx_type %d", fht_fns[f].name, tx_type);
            BENCH(label, 200000 / n, fht_fns[f].c(in, out_c, stride, tx_type),
                  fht_fns[f].neon(in, out_n, stride, tx_type));
        }
    }
}

/*******************************************
 * Quantization
 *******************************************/
typedef void (*QuantFn)(const tran_low_t *, intptr_t, int, const int16_t *, const int16_t *, const int16_t *,
                        const int16_t *, tran_low_t *, tran_low_t *, const int16_t *, uint16_t *, const int16_t *,
                        const int16_t *);

static void quant_params(int d, int16_t *zbin, int16_t *round, int16_t *quant, int16_t *shift, int16_t *dequant) {
    // as vp9_init_quant (zbin factor 84 / round factor 48 for AC, up to 80 / 64 for DC)
    unsigned t = d;
    int      l = 0;
    while (t > 1) {
        t >>= 1;
        l++;
    }
    *quant   = (int16_t)(1 + (1 << (16 + l)) / d - (1 << 16));
    *shift   = (int16_t)(1 << (16 - l));
    *zbin    = (int16_t)ROUND_POWER_OF_TWO(rnd_range(64, 84) * d, 7);
    *round   = (int16_t)((rnd_range(40, 64) * d) >> 7);
    *dequant = (int16_t)d;
}

static void test_quantize(void) {
    static const struct {
        const char *name;
        QuantFn     c, neon;
        int         tx_size;
    } fns[] = {
        {"quantize_b 4x4", eb_vp9_quantize_b_c, eb_vp9_quantize_b_neon, 0},
        {"quantize_b 8x8", eb_vp9_quantize_b_c, eb_vp9_quantize_b_neon, 1},
        {"quantize_b 16x16", eb_vp9_quantize_b_c, eb_vp9_quantize_b_neon, 2},
        {"quantize_b_32x32", eb_vp9_quantize_b_32x32_c, eb_vp9_quantize_b_32x32_neon, 3},
    };
    DECLARE_ALIGNED(16, tran_low_t, coeff[1024]);
    DECLARE_ALIGNED(16, tran_low_t, qc_c[1024]);
    DECLARE_ALIGNED(16, tran_low_t, dq_c[1024]);
    DECLARE_ALIGNED(16, tran_low_t, qc_n[1024]);
    DECLARE_ALIGNED(16, tran_low_t, dq_n[1024]);
    for (size_t f = 0; f < sizeof(fns) / sizeof(fns[0]); f++) {
        const int n = 16 << (2 * fns[f].tx_size);
        for (int t = 0; t < 20000; t++) {
            const scan_order *so = &eb_vp9_scan_orders[fns[f].tx_size][t & 3];
            int16_t           zbin[2], round[2], quant[2], shift[2], dequant[2];
            const int         d = t % 5 == 0 ? rnd_range(4, 16) : rnd_range(4, 1828);
            quant_params(d, &zbin[0], &round[0], &quant[0], &shift[0], &dequant[0]);
            quant_params(t % 3 ? d : rnd_range(4, 1828), &zbin[1], &round[1], &quant[1], &shift[1], &dequant[1]);
            const int kind = t % 4;
            for (int i = 0; i < n; i++) {
                switch (kind) {
                case 0: coeff[i] = (tran_low_t)rnd_range(-3 * d, 3 * d); break; // mostly zeros / small levels
                case 1: coeff[i] = (tran_low_t)rnd(); break; // full int16 range
                case 2: coeff[i] = (rnd() % 8) ? 0 : (tran_low_t)rnd_range(-40 * d, 40 * d); break; // sparse
                default: coeff[i] = (rnd() & 1) ? 32767 : -32768; break;
                }
            }
            uint16_t eob_c = 0xffff, eob_n = 0xfffe;
            fns[f].c(coeff, n, 0, zbin, round, quant, shift, qc_c, dq_c, dequant, &eob_c, so->scan, so->iscan);
            fns[f].neon(coeff, n, 0, zbin, round, quant, shift, qc_n, dq_n, dequant, &eob_n, so->scan, so->iscan);
            CHECK(eob_c == eob_n && !memcmp(qc_c, qc_n, n * sizeof(tran_low_t)) &&
                      !memcmp(dq_c, dq_n, n * sizeof(tran_low_t)),
                  "%s d %d kind %d eob %d / %d", fns[f].name, d, kind, eob_c, eob_n);
        }
        int16_t zbin[2], round[2], quant[2], shift[2], dequant[2];
        quant_params(60, &zbin[0], &round[0], &quant[0], &shift[0], &dequant[0]);
        quant_params(60, &zbin[1], &round[1], &quant[1], &shift[1], &dequant[1]);
        for (int i = 0; i < n; i++) coeff[i] = (rnd() % 4) ? 0 : (tran_low_t)rnd_range(-600, 600);
        const scan_order *so = &eb_vp9_scan_orders[fns[f].tx_size][0];
        uint16_t          eob;
        BENCH(fns[f].name, 4000000 / n,
              fns[f].c(coeff, n, 0, zbin, round, quant, shift, qc_c, dq_c, dequant, &eob, so->scan, so->iscan),
              fns[f].neon(coeff, n, 0, zbin, round, quant, shift, qc_n, dq_n, dequant, &eob, so->scan, so->iscan));
    }
}

/*******************************************
 * Intra prediction
 *******************************************/
typedef void (*IntraFn)(uint8_t *, ptrdiff_t, const uint8_t *, const uint8_t *);
// above_end / left_end: number of above / left pixels the C code reads (as a function of bs), so the buffers can
// end at a guard page; a = above[-1] is read when needed and is always available
#define INTRA(t, n, ae, le) {#t "_" #n "x" #n, eb_vp9_##t##_predictor_##n##x##n##_c, \
                             eb_vp9_##t##_predictor_##n##x##n##_neon, n, ae, le}
#define INTRA_SIZES(t, ae, le) INTRA(t, 8, ae, le), INTRA(t, 16, ae, le), INTRA(t, 32, ae, le)
static const struct {
    const char *name;
    IntraFn     c, neon;
    int         bs, above_end, left_end; // ends: 0 = bs - 1, 1 = bs, 2 = bs + 1, 3 = bs + 2, -1 = not read
} intra_fns[] = {
    INTRA_SIZES(d45, 2, -1),  INTRA_SIZES(d63, 3, -1),     INTRA_SIZES(d207, -1, 1),   INTRA_SIZES(d153, 0, 1),
    INTRA_SIZES(d117, 1, 0),  INTRA_SIZES(d135, 1, 1),     INTRA_SIZES(tm, 1, 1),
};

static void test_intra(void) {
    uint8_t  *above_buf = alloc_guarded(128);
    uint8_t  *left_buf  = alloc_guarded(128);
    const int stride    = 48;
    uint8_t   dst_c[48 * 32], dst_n[48 * 32];
    for (size_t f = 0; f < sizeof(intra_fns) / sizeof(intra_fns[0]); f++) {
        const int bs = intra_fns[f].bs;
        const int ae = intra_fns[f].above_end < 0 ? 0 : bs - 1 + intra_fns[f].above_end;
        const int le = intra_fns[f].left_end < 0 ? 0 : bs - 1 + intra_fns[f].left_end;
        const uint8_t *above = above_buf + 128 - ae;
        const uint8_t *left  = left_buf + 128 - le;
        for (int t = 0; t < 2000; t++) {
            fill_u8(above_buf, 128, t % PAT_COUNT);
            fill_u8(left_buf, 128, (t / PAT_COUNT) % PAT_COUNT);
            fill_u8(dst_c, sizeof(dst_c), PAT_RANDOM);
            memcpy(dst_n, dst_c, sizeof(dst_c));
            intra_fns[f].c(dst_c, stride, above, left);
            intra_fns[f].neon(dst_n, stride, above, left);
            CHECK(!memcmp(dst_c, dst_n, sizeof(dst_c)), "%s t %d", intra_fns[f].name, t);
        }
        if (bs >= 16)
            BENCH(intra_fns[f].name, 4000000 / (bs * bs), intra_fns[f].c(dst_c, stride, above, left),
                  intra_fns[f].neon(dst_n, stride, above, left));
    }
}

/*******************************************
 * Inverse transforms
 *******************************************/
typedef void (*IdctFn)(const tran_low_t *, uint8_t *, int);
#define IDCT(name, n) {#name, n, eb_vp9_##name##_c, eb_vp9_##name##_neon}
static const struct {
    const char *name;
    int         n;
    IdctFn      c, neon;
} idct_fns[] = {
    IDCT(idct8x8_64_add, 8),       IDCT(idct8x8_12_add, 8),     IDCT(idct8x8_1_add, 8),
    IDCT(idct16x16_256_add, 16),   IDCT(idct16x16_38_add, 16),  IDCT(idct16x16_10_add, 16),
    IDCT(idct16x16_1_add, 16),     IDCT(idct32x32_1024_add, 32), IDCT(idct32x32_135_add, 32),
    IDCT(idct32x32_34_add, 32),    IDCT(idct32x32_1_add, 32),
};
typedef void (*IhtFn)(const tran_low_t *, uint8_t *, int, int);
static const struct {
    const char *name;
    int         n;
    IhtFn       c, neon;
} iht_fns[] = {
    {"iht8x8_64_add", 8, eb_vp9_iht8x8_64_add_c, eb_vp9_iht8x8_64_add_neon},
    {"iht16x16_256_add", 16, eb_vp9_iht16x16_256_add_c, eb_vp9_iht16x16_256_add_neon},
};

// Coefficient patterns: dequantized-like ranges, full int16 range (idct only: the C iadst overflows int there),
// sparse (top-left only) and DC only
static void fill_coeffs(tran_low_t *p, int n, int pattern, int full_range) {
    for (int i = 0; i < n * n; i++) {
        const int r = i / n, c = i % n;
        int       v;
        switch (pattern % 6) {
        case 0: v = rnd_range(-4096, 4096); break;
        case 1: v = full_range ? (int16_t)rnd() : rnd_range(-16384, 16384); break;
        case 2: v = (r < 8 && c < 8) ? rnd_range(-2048, 2048) : 0; break;
        case 3: v = (r < 4 && c < 4) ? rnd_range(-600, 600) : 0; break;
        case 4: v = i == 0 ? rnd_range(-8000, 8000) : 0; break;
        default: v = rnd_range(-40, 40); break;
        }
        p[i] = (tran_low_t)v;
    }
}

static void test_idct(void) {
    enum { STRIDE = 48 };
    tran_low_t *coef = alloc_guarded(32 * 32 * sizeof(tran_low_t));
    uint8_t    *d_c = alloc_guarded(32 * STRIDE), *d_n = alloc_guarded(32 * STRIDE);
    for (size_t f = 0; f < sizeof(idct_fns) / sizeof(idct_fns[0]); f++) {
        const int n = idct_fns[f].n;
        for (int iter = 0; iter < 600; iter++) {
            fill_coeffs(coef, n, iter, 1);
            fill_u8(d_c, 32 * STRIDE, iter % PAT_COUNT);
            memcpy(d_n, d_c, 32 * STRIDE);
            idct_fns[f].c(coef, d_c, STRIDE);
            idct_fns[f].neon(coef, d_n, STRIDE);
            CHECK(!memcmp(d_c, d_n, 32 * STRIDE), "%s iter %d", idct_fns[f].name, iter);
        }
        fill_coeffs(coef, n, 0, 1);
        BENCH(idct_fns[f].name, 2000000 / (n * n), idct_fns[f].c(coef, d_c, STRIDE), idct_fns[f].neon(coef, d_n, STRIDE));
    }
    for (size_t f = 0; f < sizeof(iht_fns) / sizeof(iht_fns[0]); f++) {
        const int n = iht_fns[f].n;
        for (int tx = 0; tx < 4; tx++) {
            for (int iter = 0; iter < 600; iter++) {
                fill_coeffs(coef, n, iter, 0);
                fill_u8(d_c, 32 * STRIDE, iter % PAT_COUNT);
                memcpy(d_n, d_c, 32 * STRIDE);
                iht_fns[f].c(coef, d_c, STRIDE, tx);
                iht_fns[f].neon(coef, d_n, STRIDE, tx);
                CHECK(!memcmp(d_c, d_n, 32 * STRIDE), "%s tx %d iter %d", iht_fns[f].name, tx, iter);
            }
            char name[64];
            snprintf(name, sizeof(name), "%s tx %d", iht_fns[f].name, tx);
            fill_coeffs(coef, n, 0, 0);
            BENCH(name, 2000000 / (n * n), iht_fns[f].c(coef, d_c, STRIDE, tx), iht_fns[f].neon(coef, d_n, STRIDE, tx));
        }
    }
}

/*******************************************
 * Picture operators (addition, copy, distortion)
 *******************************************/
static void fill_s16(int16_t *p, int n, int pattern) {
    for (int i = 0; i < n; i++) {
        switch (pattern % 5) {
        case 0: p[i] = (int16_t)rnd_range(-255, 255); break;
        case 1: p[i] = (int16_t)rnd(); break; // full range: wrapping sums / squares
        case 2: p[i] = (i & 1) ? 32767 : -32768; break;
        case 3: p[i] = (int16_t)rnd_range(-4000, 4000); break;
        default: p[i] = (int16_t)rnd_range(-3, 3); break;
        }
    }
}

static void test_picops(void) {
    enum { S = 80, H = 64, N = S * H };
    static const uint32_t widths[] = {4, 8, 16, 32, 64, 12, 24, 48, 6, 1, 20};
    uint8_t *a = alloc_guarded(N), *b = alloc_guarded(N), *d_c = alloc_guarded(N), *d_n = alloc_guarded(N);
    int16_t *c = alloc_guarded(N * 2), *r = alloc_guarded(N * 2);
    for (int iter = 0; iter < 400; iter++) {
        const uint32_t w = widths[iter % (sizeof(widths) / sizeof(widths[0]))];
        const uint32_t h = (iter & 3) ? w : (uint32_t)rnd_range(1, H);
        // the last row ends at the guard page
        const uint32_t off = N - ((h - 1) * S + w);
        fill_u8(a, N, iter % PAT_COUNT);
        fill_u8(b, N, (iter / 3) % PAT_COUNT);
        fill_s16(c, N, iter);
        fill_s16(r, N, iter / 5);

        memset(d_c, 0x55, N);
        memset(d_n, 0x55, N);
        eb_vp9_picture_addition_kernel(a + off, S, c + off, S, d_c + off, S, w, h);
        eb_vp9_picture_addition_kernel_neon(a + off, S, c + off, S, d_n + off, S, w, h);
        CHECK(!memcmp(d_c, d_n, N), "addition %ux%u", w, h);

        memset(d_c, 0x55, N);
        memset(d_n, 0x55, N);
        copy_kernel8_bit(a + off, S, d_c + off, S, w, h);
        eb_vp9_copy_kernel8_bit_neon(a + off, S, d_n + off, S, w, h);
        CHECK(!memcmp(d_c, d_n, N), "copy %ux%u", w, h);

        uint64_t dc[2], dn[2];
        full_distortion_kernel32bit(c + off, S, r + off, S, dc, w, h);
        eb_vp9_full_distortion_kernel32bit_neon(c + off, S, r + off, S, dn, w, h);
        CHECK(dc[0] == dn[0] && dc[1] == dn[1], "full_distortion %ux%u", w, h);
        full_distortion_kernel_eob_zero32bit(c + off, S, r + off, S, dc, w, h);
        eb_vp9_full_distortion_kernel_eob_zero32bit_neon(c + off, S, r + off, S, dn, w, h);
        CHECK(dc[0] == dn[0] && dc[1] == dn[1], "full_distortion_eob_zero %ux%u", w, h);
        full_distortion_kernel_intra32bit(c + off, S, r + off, S, dc, w, h);
        eb_vp9_full_distortion_kernel_intra32bit_neon(c + off, S, r + off, S, dn, w, h);
        CHECK(dc[0] == dn[0] && dc[1] == dn[1], "full_distortion_intra %ux%u", w, h);
        CHECK(eb_vp9_spatial_full_distortion_kernel(a + off, S, b + off, S, w, h) ==
                  eb_vp9_spatial_full_distortion_kernel_neon(a + off, S, b + off, S, w, h),
              "spatial_distortion %ux%u", w, h);
    }
    fill_u8(a, N, PAT_RANDOM);
    fill_u8(b, N, PAT_RANDOM);
    fill_s16(c, N, 0);
    fill_s16(r, N, 0);
    for (uint32_t w = 8; w <= 64; w *= 2) {
        char     name[64];
        uint64_t dd[2];
        const int it = 4000000 / (w * w);
        snprintf(name, sizeof(name), "addition %ux%u", w, w);
        BENCH(name, it, eb_vp9_picture_addition_kernel(a, S, c, S, d_c, S, w, w),
              eb_vp9_picture_addition_kernel_neon(a, S, c, S, d_n, S, w, w));
        snprintf(name, sizeof(name), "copy %ux%u", w, w);
        BENCH(name, it, copy_kernel8_bit(a, S, d_c, S, w, w), eb_vp9_copy_kernel8_bit_neon(a, S, d_n, S, w, w));
        snprintf(name, sizeof(name), "full_distortion %ux%u", w, w);
        BENCH(name, it, full_distortion_kernel32bit(c, S, r, S, dd, w, w); sink += dd[0],
              eb_vp9_full_distortion_kernel32bit_neon(c, S, r, S, dd, w, w); sink += dd[0]);
        snprintf(name, sizeof(name), "spatial_distortion %ux%u", w, w);
        BENCH(name, it, sink += eb_vp9_spatial_full_distortion_kernel(a, S, b, S, w, w),
              sink += eb_vp9_spatial_full_distortion_kernel_neon(a, S, b, S, w, w));
    }
}

/*******************************************
 * Main
 *******************************************/
static const struct {
    const char *name;
    void (*fn)(void);
} tests[] = {
    {"nxm_sad", test_nxm_sad},
    {"avg_sad", test_avg_sad},
    {"sad_loop", test_sad_loop},
    {"eight_point", test_eight_point},
    {"convolve", test_convolve},
    {"fdct", test_fdct},
    {"quantize", test_quantize},
    {"intra", test_intra},
    {"picops", test_picops},
    {"idct", test_idct},
};

int main(int argc, char **argv) {
    const char *filter = argc > 1 ? argv[1] : "";
    int         total  = 0;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        if (!strstr(tests[i].name, filter))
            continue;
        g_fail = 0;
        printf("%s\n", tests[i].name);
        tests[i].fn();
        printf("  -> %s (%d mismatches)\n", g_fail ? "FAIL" : "ok", g_fail);
        total += g_fail;
    }
    printf("%s\n", total ? "FAILED" : "ALL BIT-EXACT");
    return total != 0;
}
