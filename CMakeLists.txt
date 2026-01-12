cmake_minimum_required(VERSION 4.1)

# Set CUDA path and architecture BEFORE project()

# T400 is compute capability 7.5
set(CMAKE_CUDA_ARCHITECTURES 75)

project(CudaPoc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# T400 is Turing architecture (sm_75)

add_executable(cuda_poc src/main.cpp src/add_cuda.cu)

# Link CUDA libraries
target_link_libraries(cuda_poc PRIVATE CUDA::cudart)

# Enable separable compilation
set_target_properties(cuda_poc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
