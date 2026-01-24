include_guard(GLOBAL)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find CUTLASS
find_package(NvidiaCutlass CONFIG REQUIRED)

# Find NCCL library manually (Linux only)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(NCCL_LIBRARY
            NAMES nccl
            HINTS
            ${CUDAToolkit_LIBRARY_DIR}
            /usr/local/cuda/lib64
            /usr/local/cuda/lib
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
    )

    if (NOT NCCL_LIBRARY)
        message(WARNING "NCCL library not found. NCCL features will be disabled.")
    endif ()
endif ()

set(SRCS
        src/add_cuda.cu
        src/add_one_cuda.cu
        src/depth_cuda.cu
        src/mul_cuda.cu
        src/multiply_by_cuda.cu
        src/gemm_cuda.cu
        src/pipeline_cuda.cu
        src/sin_cuda.cu
        src/sqrt_cuda.cu
        src/subtract_by_cuda.cu
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
        nvidia::cutlass::cutlass
)

# Link NCCL if found
if (NCCL_LIBRARY)
    target_link_libraries(cudaPoc PUBLIC ${NCCL_LIBRARY})
    message(STATUS "NCCL library found: ${NCCL_LIBRARY}")
endif ()

# Enable separable compilation and relocatable device code for dynamic parallelism
set_target_properties(cudaPoc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)