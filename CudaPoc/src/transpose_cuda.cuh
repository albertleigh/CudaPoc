//
// Created by Albert Li on 1/16/2026.
//

#pragma once

#include <cuda_runtime.h>

namespace cuda_poc {
    constexpr int TRANSPOSE_BLOCK_DIM = 32;

    // C++ callable wrapper function for matrix transpose
    // For square matrices: rows == cols
    // For rectangular matrices: input is rows x cols, output is cols x rows
    template<typename T>
    void transpose(T *output, const T *input, int rows, int cols, dim3 grid_dim, dim3 block_dim);
} // namespace cuda_poc
