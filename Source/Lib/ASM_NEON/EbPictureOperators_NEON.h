/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbPictureOperators_NEON_h
#define EbPictureOperators_NEON_h

#include "EbDefinitions.h"
#ifdef __cplusplus
extern "C" {
#endif

// All kernels are bit-exact with the C versions in EbPictureOperators_C.c and accept any width / height.

void eb_vp9_picture_addition_kernel_neon(uint8_t *pred_ptr, uint32_t pred_stride, int16_t *residual_ptr,
                                         uint32_t residual_stride, uint8_t *recon_ptr, uint32_t recon_stride,
                                         uint32_t width, uint32_t height);
void eb_vp9_copy_kernel8_bit_neon(EbByte src, uint32_t src_stride, EbByte dst, uint32_t dst_stride,
                                  uint32_t area_width, uint32_t area_height);
void eb_vp9_full_distortion_kernel32bit_neon(int16_t *coeff, uint32_t coeff_stride, int16_t *recon_coeff,
                                             uint32_t recon_coeff_stride, uint64_t distortion_result[2],
                                             uint32_t area_width, uint32_t area_height);
void eb_vp9_full_distortion_kernel_eob_zero32bit_neon(int16_t *coeff, uint32_t coeff_stride, int16_t *recon_coeff,
                                                      uint32_t recon_coeff_stride, uint64_t distortion_result[2],
                                                      uint32_t area_width, uint32_t area_height);
void eb_vp9_full_distortion_kernel_intra32bit_neon(int16_t *coeff, uint32_t coeff_stride, int16_t *recon_coeff,
                                                   uint32_t recon_coeff_stride, uint64_t distortion_result[2],
                                                   uint32_t area_width, uint32_t area_height);
uint64_t eb_vp9_spatial_full_distortion_kernel_neon(uint8_t *input, uint32_t input_stride, uint8_t *recon,
                                                    uint32_t recon_stride, uint32_t area_width, uint32_t area_height);

#ifdef __cplusplus
}
#endif
#endif // EbPictureOperators_NEON_h
