# CUDA and C++ Standard Configuration
# Modify this file to change CUDA architecture and corresponding C++ standard

# Set your target CUDA architecture here
# Options: 61 (Pascal/Quadro P600), 75 (Turing/T400), 86 (Ampere/RTX A500)
set(TARGET_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture")

# Architecture to C++ Standard mapping
# Define the mapping between CUDA architectures and C++ standards
if(TARGET_CUDA_ARCH STREQUAL "61")
    set(TARGET_CXX_STANDARD 17)
elseif(TARGET_CUDA_ARCH STREQUAL "75")
    set(TARGET_CXX_STANDARD 17)
elseif(TARGET_CUDA_ARCH STREQUAL "86")
    set(TARGET_CXX_STANDARD 20)
else()
    message(WARNING "Unknown CUDA architecture ${TARGET_CUDA_ARCH}, defaulting to C++17")
    set(TARGET_CXX_STANDARD 17)
endif()

message(STATUS "Configuration loaded: CUDA Architecture = ${TARGET_CUDA_ARCH}, C++ Standard = ${TARGET_CXX_STANDARD}")


# Set CUDA compiler path for Linux - must be set before project()
if(UNIX AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CUDAToolkit_ROOT "/usr/local/cuda-13.1" CACHE PATH "CUDA toolkit root")
    set(CUDNN_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "CUDNN include directory")
    set(CUDNN_LIBRARY "/usr/lib/x86_64-linux-gnu/libcudnn.so" CACHE FILEPATH "CUDNN library")
    # Add vcpkg installed packages to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-linux/share")
endif()