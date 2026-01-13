cmake_minimum_required(VERSION 4.1)

set(PROJECT_NAME "CudaPoc")

project(${PROJECT_NAME} VERSION 0.0.1)

add_definitions(-DFMT_HEADER_ONLY=1)
add_definitions(-DENABLE_LOG=1)

# Set CUDA path and architecture BEFORE project()

# T400 is compute capability 7.5
set(CMAKE_CUDA_ARCHITECTURES 75)

project(CudaPoc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# Find fmt
find_package(fmt CONFIG REQUIRED)

set(LIBS ${LIBS})

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")

add_subdirectory(CudaPoc)
add_subdirectory(example)
add_subdirectory(test)