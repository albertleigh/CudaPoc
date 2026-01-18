cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "CudaPoc")

project(${PROJECT_NAME}
        LANGUAGES CXX CUDA
        VERSION 0.0.1)

add_definitions(-DFMT_HEADER_ONLY=1)
add_definitions(-DENABLE_LOG=1)

# Set CUDA path and architecture BEFORE project()

# Enable common architectures for better compatibility.
# sm_61 is for Pascal (Quadro P600),
# sm_75 is for Turing (T400),
# sm_86 is for Ampere (RTX A500)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(CudaPoc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")

add_subdirectory(CudaPoc)
add_subdirectory(example)
add_subdirectory(test)