/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbNeonCommon_h
#define EbNeonCommon_h

#include <arm_neon.h>
#include <stdint.h>

#define ALWAYS_INLINE static inline __attribute__((always_inline))

// 4-byte loads/stores (no alignment requirement, never touch bytes outside [p, p + 4))
ALWAYS_INLINE uint8x8_t load_u8_4(const uint8_t *p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return vreinterpret_u8_u32(vset_lane_u32(v, vdup_n_u32(0), 0));
}

ALWAYS_INLINE void store_u8_4(uint8_t *p, uint8x8_t v) {
    const uint32_t x = vget_lane_u32(vreinterpret_u32_u8(v), 0);
    __builtin_memcpy(p, &x, 4);
}

#endif // EbNeonCommon_h
