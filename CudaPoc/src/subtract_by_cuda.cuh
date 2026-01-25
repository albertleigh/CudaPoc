//
// Created by ali on 1/25/26.
//

#pragma once

#include <cuda_runtime.h>

namespace cuda_poc::inplace {
// CUDA kernel (callable from .cu files only)
template <typename T>
__global__ void subtract_by_kernel(T* a, T b, size_t n, size_t step);

// C++ callable wrapper function
template <typename T>
void vector_subtract_by(T* a, T b, size_t n, dim3 grid_dim, dim3 block_dim);

}  // namespace cuda_poc::inplace
