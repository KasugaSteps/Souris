cmake_minimum_required(VERSION 3.20)

project(souris_code LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# GGML owns tensor allocation, kernels, CPU threading and CUDA execution.
add_subdirectory(ggml)
if (GGML_CUDA)
    enable_language(CUDA)
endif ()
add_subdirectory(src)
