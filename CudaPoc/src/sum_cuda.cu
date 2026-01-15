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


    template<typename T>
    __global__ void sum_kernel_v2(T *result, const T *input, size_t n) {
        const size_t tid = threadIdx.x;
        const size_t step = blockDim.x * gridDim.x;
        // Lane is a thread in a warp of 32 thread.
        unsigned int lane_id = tid % 32;
        size_t idx = blockIdx.x * blockDim.x + tid;

        if (lane_id == 0) {
            T wrap_sum = 0;
            // Loop over the 32 threads in the wrap.
            for (int i=0; i< 32; ++i) {
                for (int j = idx+i; j<n; j+=step) {
                    wrap_sum += input[j];
                }
            }
            // Atomically add this wrap's sum to the final output.
            atomicAdd(result, wrap_sum);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v2(T *result, T *input, size_t n, dim3 grid, dim3 block) {
        sum_kernel_v2<T><<<grid, block>>>(result, input, n);
    }

    template void vector_sum_v2<float>(float *result, float *input, size_t n, dim3 grid, dim3 block);

} //namespace cuda_poc
