//
// Created by Albert Li on 1/15/2026.
//

#include "sum_cuda.cuh"

namespace cuda_poc {
    template<typename T>
    __global__ void sum_kernel_v1_all_atomic(T *result, const T *input, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride) {
            atomicAdd(result, input[i]);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v1(T *result, T *input, size_t n, dim3 grid, dim3 block) {
        sum_kernel_v1_all_atomic<T><<<grid, block>>>(result, input, n);
    }

    template void vector_sum_v1<float>(float *result, float *input, size_t n, dim3 grid, dim3 block);


    template<typename T>
    __global__ void sum_kernel_v2_intra_wrap_dual_loop(T *result, const T *input, size_t n, unsigned int wrap_size) {
        const size_t tid = threadIdx.x;
        const size_t step = blockDim.x * gridDim.x;
        // Lane is a thread in a warp of usually 32 threads.
        unsigned int lane_id = tid % wrap_size;
        size_t idx = blockIdx.x * blockDim.x + tid;

        if (lane_id == 0) {
            T wrap_sum = 0;
            // Loop over the 32 threads in the wrap.
            for (int i = 0; i < wrap_size; ++i) {
                for (int j = idx + i; j < n; j += step) {
                    wrap_sum += input[j];
                }
            }
            // Atomically add this wrap's sum to the final output.
            atomicAdd(result, wrap_sum);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v2(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        sum_kernel_v2_intra_wrap_dual_loop<T><<<grid, block>>>(result, input, n, wrap_size);
    }

    template void vector_sum_v2<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);

    template<typename T>
    __global__ void sum_kernel_v3_intra_wrap_loop(T *result, const T *input, size_t n, unsigned int wrap_size) {
        const size_t tid = threadIdx.x;
        // Lane is a thread in a warp of usually 32 threads.
        unsigned int lane_id = tid % wrap_size;

        if (lane_id == 0) {
            const size_t step = blockDim.x * gridDim.x;
            size_t idx = blockIdx.x * blockDim.x + tid;
            T wrap_sum = 0;
            const size_t total_elements = ((n - idx) + step - 1) / step;

            for (size_t linera_idx = 0; linera_idx < total_elements * 32; linera_idx++) {
                const size_t segment = linera_idx / 32;
                const size_t lane_offset = linera_idx % 32;
                const size_t input_idx = idx + segment * step + lane_offset;
                if (input_idx < n) {
                    wrap_sum += input[input_idx];
                }
            }
            // Atomically add this wrap's sum to the final output.
            atomicAdd(result, wrap_sum);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v3(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        sum_kernel_v3_intra_wrap_loop<T><<<grid, block>>>(result, input, n, wrap_size);
    }

    template void vector_sum_v3<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);

    template<typename T>
    __global__ void sum_kernel_v4_intra_block_loop(T *result, const T *input, size_t n, unsigned int wrap_size) {
        const size_t tid = threadIdx.x;
        size_t idx = blockIdx.x * blockDim.x + tid;

        if (tid == 0) {
            T wrap_sum = 0;
            for (size_t i = 0; i < blockDim.x; ++i) {
                for (size_t j = idx + i; j < n; j += blockDim.x * gridDim.x) {
                    wrap_sum += input[j];
                }
            }
            // Atomically add this wrap's sum to the final output.
            atomicAdd(result, wrap_sum);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v4(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        sum_kernel_v4_intra_block_loop<T><<<grid, block>>>(result, input, n, wrap_size);
    }

    template void vector_sum_v4<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);

    template<typename T>
    __global__ void sum_kernel_v5_smem_tree_reduce(T *result, const T *input, size_t n, unsigned int wrap_size) {
        extern __shared__ T smem[];
        const size_t tid = threadIdx.x;
        size_t idx = blockIdx.x * blockDim.x + tid;

        smem[tid] = (idx < n) ? input[idx] : 0;

        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            // Atomically add this wrap's sum to the final output.
            atomicAdd(result, smem[0]);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v5(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        sum_kernel_v5_smem_tree_reduce<T><<<grid, block>>>(result, input, n, wrap_size);
    }

    template void vector_sum_v5<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);


    template<typename T>
    __global__ void sum_kernel_v6_intra_wrap_dual_hardware_loop(T *result, const T *input, size_t n,
                                                                unsigned int wrap_size) {
        const size_t tid = threadIdx.x;
        size_t idx = blockIdx.x * blockDim.x + tid;

        T wrap_sum = 0;

        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
            wrap_sum += input[i];
        }

        for (int offset = wrap_size / 2; offset > 0; offset >>= 1) {
            wrap_sum += __shfl_down_sync(0xffffffff, wrap_sum, offset);
        }

        if (tid % wrap_size == 0) {
            // Atomically add this wrap's sum to the final output.
            atomicAdd(result, wrap_sum);
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v6(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        sum_kernel_v6_intra_wrap_dual_hardware_loop<T><<<grid, block>>>(result, input, n, wrap_size);
    }

    template void vector_sum_v6<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);
} //namespace cuda_poc
