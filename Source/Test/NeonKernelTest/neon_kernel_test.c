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
