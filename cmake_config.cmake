# CUDA and C++ Standard Configuration
# Modify this file to change CUDA architecture and corresponding C++ standard

# Set your target CUDA architecture here
# Options: 61 (Pascal/Quadro P600), 75 (Turing/T400), 86 (Ampere/RTX A500)
set(TARGET_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture")

# Architecture to C++ Standard mapping
# Define the mapping between CUDA architectures and C++ standards
if(TARGET_CUDA_ARCH STREQUAL "61")
    set(TARGET_CXX_STANDARD 17)
elseif(TARGET_CUDA_ARCH STREQUAL "75")
    set(TARGET_CXX_STANDARD 17)
elseif(TARGET_CUDA_ARCH STREQUAL "86")
    set(TARGET_CXX_STANDARD 20)
else()
    message(WARNING "Unknown CUDA architecture ${TARGET_CUDA_ARCH}, defaulting to C++17")
    set(TARGET_CXX_STANDARD 17)
endif()

message(STATUS "Configuration loaded: CUDA Architecture = ${TARGET_CUDA_ARCH}, C++ Standard = ${TARGET_CXX_STANDARD}")
