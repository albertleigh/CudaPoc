vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO NVIDIA/cutlass
    REF "v${VERSION}"
    SHA512 a08aac281fb3bdea82c0a044dc643c40e4803d02e55bbea450021cb7a5472aed86e79c5df41cd981976af8403f18cc48d8069045c4e68339430d3a3caeb109ac
    HEAD_REF main
)

vcpkg_find_acquire_program(PYTHON3)
get_filename_component(PYTHON_PATH "${PYTHON3}" PATH)
vcpkg_add_to_path(PREPEND "${PYTHON_PATH}")

# Find CUDA manually
find_program(NVCC nvcc 
    PATHS 
        ENV CUDA_PATH 
        /usr/local/cuda/bin
        /usr/local/cuda-12.6/bin
        /usr/local/cuda-12.5/bin
        /usr/local/cuda-12/bin
    PATH_SUFFIXES bin
)
if(NVCC)
    get_filename_component(CUDA_BIN_DIR "${NVCC}" DIRECTORY)
    get_filename_component(cuda_toolkit_root "${CUDA_BIN_DIR}" DIRECTORY)
    message(STATUS "Found CUDA: ${cuda_toolkit_root}")
    list(APPEND FEATURE_OPTIONS
        "-DCMAKE_CUDA_COMPILER=${NVCC}"
        "-DCUDAToolkit_ROOT=${cuda_toolkit_root}"
    )
else()
    message(WARNING "CUDA not found. CUTLASS may not configure properly.")
endif()

# cuDNN is disabled because cuDNN 9.18 uses a versioned directory structure (include/13.1)
# that is incompatible with CUTLASS's find logic which expects include/cudnn.h directly
# If you need cuDNN support, you'll need to patch CUTLASS's cuDNN.cmake file

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DCMAKE_SUPPRESS_REGENERATION=ON # for some reason it keeps regenerating in Windows
        "-DCUTLASS_REVISION:STRING=v${VERSION}"
        -DCUTLASS_NATIVE_CUDA=OFF
        -DCUTLASS_ENABLE_HEADERS_ONLY=ON
        -DCUTLASS_ENABLE_TOOLS=ON
        -DCUTLASS_ENABLE_LIBRARY=OFF
        -DCUTLASS_ENABLE_PROFILER=OFF
        -DCUTLASS_ENABLE_PERFORMANCE=OFF
        -DCUTLASS_ENABLE_TESTS=OFF
        -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF
        -DCUTLASS_ENABLE_CUBLAS=OFF
        -DCUTLASS_ENABLE_CUDNN=OFF
        "-DPython3_EXECUTABLE:FILEPATH=${PYTHON3}"
        ${FEATURE_OPTIONS}
    MAYBE_UNUSED_VARIABLES
        CUTLASS_NATIVE_CUDA
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/NvidiaCutlass" PACKAGE_NAME "NvidiaCutlass")



file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug"
    "${CURRENT_PACKAGES_DIR}/test"
    "${CURRENT_PACKAGES_DIR}/lib"
)
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
