//
// Created by wentaoli on 1/15/2026.
//

#include "sum_cuda.cuh"

namespace cuda_poc {
    template<typename T>
    __global__ void sum_kernel_v1(T *result, const T *input, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride) {
            atomicAdd(result, input[i]);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v1(T *result, T *input, size_t n, dim3 grid, dim3 block) {
        sum_kernel_v1<T><<<grid, block>>>(result, input, n);
    }

    template void vector_sum_v1<float>(float *result, float *input, size_t n, dim3 grid, dim3 block);
} //namespace cuda_poc
