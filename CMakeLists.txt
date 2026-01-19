cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "CudaPoc")

# Load external configuration BEFORE project()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake_config.cmake)

# Set CUDA path and architecture BEFORE project()
set(CMAKE_CUDA_ARCHITECTURES ${TARGET_CUDA_ARCH})
# Set C++ standard based on configuration
set(CMAKE_CXX_STANDARD ${TARGET_CXX_STANDARD})

project(${PROJECT_NAME}
        LANGUAGES CXX CUDA
        VERSION 0.0.1)

add_definitions(-DFMT_HEADER_ONLY=1)
add_definitions(-DENABLE_LOG=1)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")

add_subdirectory(CudaPoc)
add_subdirectory(example)
add_subdirectory(test)