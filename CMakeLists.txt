cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "CudaPoc")

# Set CUDA compiler path for Linux - must be set before project()
if(UNIX AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CUDAToolkit_ROOT "/usr/local/cuda-13.1" CACHE PATH "CUDA toolkit root")
    set(CUDNN_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "CUDNN include directory")
    set(CUDNN_LIBRARY "/usr/lib/x86_64-linux-gnu/libcudnn.so" CACHE FILEPATH "CUDNN library")
    # Add vcpkg installed packages to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-linux/share")
endif()

# Enable common architectures for better compatibility.
# sm_61 is for Pascal (Quadro P600),
# sm_75 is for Turing (T400),
# sm_86 is for Ampere (RTX A500)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(${PROJECT_NAME}
        LANGUAGES CXX CUDA
        VERSION 0.0.1)

add_definitions(-DFMT_HEADER_ONLY=1)
add_definitions(-DENABLE_LOG=1)

set(CMAKE_CXX_STANDARD 20)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")

add_subdirectory(CudaPoc)
add_subdirectory(example)
add_subdirectory(test)