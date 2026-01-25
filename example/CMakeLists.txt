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

    # nccl_mpi_example (requires MPI)
    find_package(MPI)
    if(MPI_FOUND)
        add_executable(nccl_mpi_example nccl_mpi_example.cu)
        target_link_libraries(nccl_mpi_example cudaPoc fmt::fmt-header-only MPI::MPI_CXX)
        target_include_directories(nccl_mpi_example PRIVATE ${MPI_CXX_INCLUDE_DIRS})
        # Enable separable compilation
        set_target_properties(nccl_mpi_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        message(STATUS "MPI found - building nccl_mpi_example")
    else()
        message(STATUS "MPI not found - skipping nccl_mpi_example (install OpenMPI or MPICH to enable)")
    endif()
endif()