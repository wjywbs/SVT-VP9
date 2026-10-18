/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbComputeMean_NEON_h
#define EbComputeMean_NEON_h

#include "EbDefinitions.h"
#ifdef __cplusplus
extern "C" {
#endif

// Bit-exact with EbComputeMean_C.c. The full-block versions take any size (widths that are not a multiple of 8
// use the C code).
uint64_t eb_vp9_compute_mean_neon(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width,
                                  uint32_t input_area_height);
uint64_t eb_vp9_compute_mean_of_squared_values_neon(uint8_t *input_samples, uint32_t input_stride,
                                                    uint32_t input_area_width, uint32_t input_area_height);
uint64_t eb_vp9_compute_sub_mean8x8_neon(uint8_t *input_samples, uint16_t input_stride);
uint64_t eb_vp9_compute_subd_mean_of_squared_values8x8_neon(uint8_t *input_samples, uint16_t input_stride);

#ifdef __cplusplus
}
#endif
#endif // EbComputeMean_NEON_h
