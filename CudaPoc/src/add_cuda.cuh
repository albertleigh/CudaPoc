#pragma once

#include <cuda_runtime.h>

// C++ callable wrapper function
template <typename T>
void vector_add(T* c, const T* a, const T* b, size_t n, dim3 grid_dim, dim3 block_dim);
