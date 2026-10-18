/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbAvcStyleMcp_NEON_h
#define EbAvcStyleMcp_NEON_h

#include "EbDefinitions.h"
#ifdef __cplusplus
extern "C" {
#endif

// Bit-exact with EbAvcStyleMcp_C.c, any width / height
void eb_vp9_avc_style_luma_interpolation_filter_horizontal_neon(EbByte ref_pic, uint32_t src_stride, EbByte dst,
                                                                uint32_t dst_stride, uint32_t pu_width,
                                                                uint32_t pu_height, EbByte temp_buf,
                                                                uint32_t frac_pos);
void eb_vp9_avc_style_luma_interpolation_filter_vertical_neon(EbByte ref_pic, uint32_t src_stride, EbByte dst,
                                                              uint32_t dst_stride, uint32_t pu_width,
                                                              uint32_t pu_height, EbByte temp_buf, uint32_t frac_pos);

#ifdef __cplusplus
}
#endif
#endif // EbAvcStyleMcp_NEON_h
