# ~~~
# Copyright(c) 2019 Intel Corporation
# SPDX - License - Identifier: BSD - 2 - Clause - Patent
# ~~~

# NEON kernel test: compares every NEON kernel with its C version on random and extreme inputs
# (bit-exact check) and times both. Needs the static library (-DBUILD_SHARED_LIBS=OFF).

include_directories(${PROJECT_SOURCE_DIR}/Source/API
                    ${PROJECT_SOURCE_DIR}/Source/Lib/Codec
                    ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT
                    ${PROJECT_SOURCE_DIR}/Source/Lib/VPX
                    ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_NEON)

add_executable(SvtVp9NeonKernelTest neon_kernel_test.c)
target_link_libraries(SvtVp9NeonKernelTest SvtVp9Enc ${PLATFORM_LIBS} pthread)
