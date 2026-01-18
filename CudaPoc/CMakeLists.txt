include_guard(GLOBAL)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

set(SRCS
        src/add_cuda.cu
        src/depth_cuda.cu
        src/gemm_cuda.cu
        src/sum_cuda.cu
        src/transpose_cuda.cu
        src/cuda_utils.h
)

add_library(cudaPoc STATIC ${SRCS})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Allow ncu pick up the source code
    target_compile_options(cudaPoc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif ()

# Enable relocatable device code for dynamic parallelism
target_compile_options(cudaPoc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

target_include_directories(cudaPoc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cudaPoc
        PUBLIC
        CUDA::cudart
        CUDA::cublas
)
# Enable separable compilation and relocatable device code for dynamic parallelism
set_target_properties(cudaPoc PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)