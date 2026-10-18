# ~~~
# Copyright(c) 2019 Intel Corporation
# SPDX - License - Identifier: BSD - 2 - Clause - Patent
# ~~~

# ASM_NEON Directory CMakeLists.txt (AArch64 NEON kernels)

include_directories(${PROJECT_SOURCE_DIR}/Source/API
                    ${PROJECT_SOURCE_DIR}/Source/Lib/Codec
                    ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT
                    ${PROJECT_SOURCE_DIR}/Source/Lib/VPX
                    ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_NEON)

file(GLOB
     all_files
     "*.h"
     "*.c")

add_library(VP9_ASM_NEON OBJECT ${all_files})
