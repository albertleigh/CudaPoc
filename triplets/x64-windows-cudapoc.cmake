# Custom triplet for CudaPoc project
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

# Disable cuDNN for nvidia-cutlass to avoid dependency issues
if(PORT MATCHES "nvidia-cutlass")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS 
        "-DCUTLASS_ENABLE_CUDNN=OFF"
    )
endif()
