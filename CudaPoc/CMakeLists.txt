include_guard(GLOBAL)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

set(SRCS
        src/add_cuda.cu
        src/cuda_utils.h
        src/sum_cuda.cu
)

message(STATUS "libs ${LIBS} CUDA::cudart")

add_library(cudaPoc STATIC ${SRCS})
target_include_directories(cudaPoc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cudaPoc
        PRIVATE
        ${LIBS}
)
# Enable separable compilation
set_target_properties(cudaPoc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)