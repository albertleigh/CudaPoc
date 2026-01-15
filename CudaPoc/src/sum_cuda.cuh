//
// Created by Albert Li on 1/15/2026.
//

#pragma once

#include <cuda_runtime.h>

namespace cuda_poc {
    template<typename T>
    void vector_sum_v1(T *result, T *input, size_t n, dim3 grid, dim3 block);

    template<typename T>
    void vector_sum_v2(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size);

    template<typename T>
    void vector_sum_v3(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size);

    template<typename T>
    void vector_sum_v4(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size);
} //namespace cuda_poc
