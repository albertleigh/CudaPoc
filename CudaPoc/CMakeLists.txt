include_guard(GLOBAL)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find fmt
find_package(fmt CONFIG REQUIRED)

set(PUBLIC_HEADERS
        src/cuda_utils.h
        src/add_cuda.cuh
)

# Create include structure in build directory
set(EXPORT_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${EXPORT_INC_DIR}/cudaPoc")

foreach(HEADER ${PUBLIC_HEADERS})
    get_filename_component(HEADER_NAME ${HEADER} NAME)
    configure_file("${HEADER}" "${EXPORT_INC_DIR}/cudaPoc/${HEADER_NAME}" COPYONLY)
endforeach()

set(SRCS
        src/add_cuda.cu
        ${PUBLIC_HEADERS}
)

message(STATUS "libs ${LIBS} CUDA::cudart")

add_library(cudaPoc STATIC ${SRCS})

target_include_directories(cudaPoc
    PUBLIC
        $<BUILD_INTERFACE:${EXPORT_INC_DIR}>
    PRIVATE
        src
)

target_link_libraries(cudaPoc
        PRIVATE
        ${LIBS}
        fmt::fmt-header-only
)
# Enable separable compilation
set_target_properties(cudaPoc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)