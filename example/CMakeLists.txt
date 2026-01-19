cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "CudaPocExample")

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find fmt
find_package(fmt CONFIG REQUIRED)

# add_example
add_executable(add_example add_example.cc)
target_link_libraries(add_example cudaPoc fmt::fmt-header-only)
# Enable separable compilation
set_target_properties(add_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# nccl_example (Linux only)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(nccl_example nccl_example.cc)
    target_link_libraries(nccl_example cudaPoc fmt::fmt-header-only)
    # Enable separable compilation
    set_target_properties(nccl_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()