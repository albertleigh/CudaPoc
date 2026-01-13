cmake_minimum_required(VERSION 4.1)

set(PROJECT_NAME "CudaPocTest")

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find fmt
find_package(fmt CONFIG REQUIRED)

# Find gtest
find_package(GTest CONFIG REQUIRED)