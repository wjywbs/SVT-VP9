/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbComputeMean_h
#define EbComputeMean_h

#ifdef ARCH_X86_64
#include "EbComputeMean_SSE2.h"
#endif
#include "EbComputeMean_C.h"
#ifdef ARCH_AARCH64
#include "EbComputeMean_NEON.h"
#endif
#ifdef ARCH_X86_64
#include "EbCombinedAveragingSAD_Intrinsic_AVX2.h"
#endif
#include "EbDefinitions.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sub-sampled 8x8 mean kernels, called directly (SSE2 is the x86 baseline).
#ifdef ARCH_X86_64
#define compute_sub_mean8x8 eb_vp9_compute_sub_mean8x8_sse2_intrin
#define compute_subd_mean_of_squared_values8x8 eb_vp9_compute_subd_mean_of_squared_values8x8_sse2_intrin
#elif defined(ARCH_AARCH64)
#define compute_sub_mean8x8 eb_vp9_compute_sub_mean8x8_neon
#define compute_subd_mean_of_squared_values8x8 eb_vp9_compute_subd_mean_of_squared_values8x8_neon
#else
#define compute_sub_mean8x8 eb_vp9_compute_sub_mean8x8_c
#define compute_subd_mean_of_squared_values8x8 eb_vp9_compute_subd_mean_of_squared_values8x8_c
#endif

typedef uint64_t (*EbComputeMeanFunc)(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width,
                                      uint32_t input_area_height);

static const EbComputeMeanFunc compute_mean_func[2][ASM_TYPE_TOTAL] = {
    {// C_DEFAULT
     compute_mean,
#ifdef ARCH_X86_64
     // AVX2
     eb_vp9_compute_mean8x8_avx2_intrin
#elif defined(ARCH_AARCH64)
     // NEON
     eb_vp9_compute_mean_neon
#else
     compute_mean,
#endif
    },
    {// C_DEFAULT
     compute_mean_of_squared_values,
#ifdef ARCH_X86_64
     // AVX2
     eb_vp9_compute_mean_of_squared_values8x8_sse2_intrin
#elif defined(ARCH_AARCH64)
     // NEON
     eb_vp9_compute_mean_of_squared_values_neon
#else
     compute_mean_of_squared_values,
#endif
    }};

#ifdef __cplusplus
}
#endif

#endif
