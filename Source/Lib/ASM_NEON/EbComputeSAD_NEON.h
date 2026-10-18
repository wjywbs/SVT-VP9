/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbComputeSAD_NEON_h
#define EbComputeSAD_NEON_h

#include "EbDefinitions.h"
#ifdef __cplusplus
extern "C" {
#endif

// All kernels are bit-exact with the C versions in EbComputeSAD_C.c (same SADs, same tie-breaking).

uint32_t eb_vp9_compute4x_m_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                     uint32_t height, uint32_t width);
uint32_t eb_vp9_compute8x_m_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                     uint32_t height, uint32_t width);
uint32_t eb_vp9_compute16x_m_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                      uint32_t height, uint32_t width);
uint32_t eb_vp9_compute24x_m_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                      uint32_t height, uint32_t width);
uint32_t eb_vp9_compute32x_m_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                      uint32_t height, uint32_t width);
uint32_t eb_vp9_compute48x_m_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                      uint32_t height, uint32_t width);
uint32_t eb_vp9_compute64x_m_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                      uint32_t height, uint32_t width);
// Any width (multiple of 4 is fastest)
uint32_t eb_vp9_fast_loop_nx_m_sad_kernel_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                               uint32_t height, uint32_t width);

uint32_t eb_vp9_combined_averaging_sad_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref1, uint32_t ref1_stride,
                                            uint8_t *ref2, uint32_t ref2_stride, uint32_t height, uint32_t width);

void eb_vp9_sad_loop_kernel_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t height,
                                 uint32_t width, uint64_t *best_sad, int16_t *x_search_center,
                                 int16_t *y_search_center, uint32_t src_stride_raw, int16_t search_area_width,
                                 int16_t search_area_height);

void eb_vp9_get_eight_horizontal_search_point_results_8x8_16x16_pu_neon(uint8_t *src, uint32_t src_stride,
                                                                        uint8_t *ref, uint32_t ref_stride,
                                                                        uint32_t *p_best_sad8x8, uint32_t *p_best_mv8x8,
                                                                        uint32_t *p_best_sad16x16,
                                                                        uint32_t *p_best_mv16x16, uint32_t mv,
                                                                        uint16_t *p_sad16x16);

#ifdef __cplusplus
}
#endif
#endif // EbComputeSAD_NEON_h
