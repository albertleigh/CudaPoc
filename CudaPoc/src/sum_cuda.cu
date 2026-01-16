//
// Created by Albert Li on 1/15/2026.
//
#include "sum_cuda.cuh"
#include <filesystem>
#include "cuda_utils.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)
#include <cooperative_groups.h>
#endif

namespace cuda_poc {
    namespace cg = cooperative_groups;

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
        size_t smem_size = block.x * sizeof(T);
        sum_kernel_v5_smem_tree_reduce<T><<<grid, block, smem_size>>>(result, input, n, wrap_size);
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


    template<typename T>
    __device__ T warp_reduce(T val, unsigned int wrap_size) {
        // suggest compiler unroll the loop
#pragma unroll
        for (unsigned int offset = wrap_size / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        return val;
    }

    template<typename T>
    __global__ void sum_kernel_v7_reduce_intra_wrap_shuffle(T *result, const T *input, size_t n,
                                                            unsigned int wrap_size) {
        extern __shared__ T smem[];
        const size_t tid = threadIdx.x;
        const size_t idx = blockIdx.x * blockDim.x + tid;

        T sum = 0;
        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
            sum += input[i];
        }

        T warp_sum = warp_reduce(sum, wrap_size);

        if (tid % wrap_size == 0) {
            smem[tid / wrap_size] = warp_sum;
        }

        __syncthreads();

        if (tid < wrap_size) {
            T block_sum = (tid < (blockDim.x + wrap_size - 1) / wrap_size) ? smem[tid] : T(0);
            // The magic relies on a specific property of current GPU architectures: The maximum number of threads in a block is 1024.
            // 1024 / 32 = 32, which means at most 32 warps in a block.
            // therefore, we can directly use warp_reduce here to reduce within a warp.
            block_sum = warp_reduce(block_sum, wrap_size);
            if (tid == 0) {
                atomicAdd(result, block_sum);
            }
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v7(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        sum_kernel_v7_reduce_intra_wrap_shuffle<T><<<grid, block>>>(result, input, n, wrap_size);
    }

    template void vector_sum_v7<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);

    template<typename T>
    __global__ void sum_kernel_v8_shuffle_first_pass(T *intermediate, const T *input, size_t n,
                                                     unsigned int wrap_size) {
        extern __shared__ T smem[];
        size_t tid = threadIdx.x;
        size_t idx = blockIdx.x * blockDim.x + tid;

        T sum = 0;

        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
            sum += input[i];
        }

        T warp_sum = warp_reduce(sum, wrap_size);

        if (tid % wrap_size == 0) {
            smem[tid / wrap_size] = warp_sum;
        }

        __syncthreads();

        if (tid < wrap_size) {
            T block_sum = (tid < (blockDim.x + wrap_size - 1) / wrap_size) ? smem[tid] : T(0);
            // The magic relies on a specific property of current GPU architectures: The maximum number of threads in a block is 1024.
            // 1024 / 32 = 32, which means at most 32 warps in a block.
            // therefore, we can directly use warp_reduce here to reduce within a warp.
            block_sum = warp_reduce(block_sum, wrap_size);
            if (tid == 0) {
                intermediate[blockIdx.x] = block_sum;
            }
        }
    }

    template<typename T>
    __global__ void sum_kernel_v8_shuffle_second_pass(T *result, const T *intermediate, size_t n,
                                                      unsigned int wrap_size) {
        extern __shared__ T smem[];
        size_t tid = threadIdx.x;

        T sum = 0;

        for (size_t i = tid; i < n; i += blockDim.x) {
            sum += intermediate[i];
        }

        T warp_sum = warp_reduce(sum, wrap_size);

        if (tid % wrap_size == 0) {
            smem[tid / wrap_size] = warp_sum;
        }

        __syncthreads();

        if (tid < wrap_size) {
            T block_sum = (tid < (blockDim.x + wrap_size - 1) / wrap_size) ? smem[tid] : T(0);
            // The magic relies on a specific property of current GPU architectures: The maximum number of threads in a block is 1024.
            // 1024 / 32 = 32, which means at most 32 warps in a block.
            // therefore, we can directly use warp_reduce here to reduce within a warp.
            block_sum = warp_reduce(block_sum, wrap_size);
            if (tid == 0) {
                *result += block_sum;
            }
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v8(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        // Allocate intermediate buff for block results
        T *d_intermediate;
        CUDA_CHECK(cudaMalloc(&d_intermediate, grid.x * sizeof(T)));

        // First pass: reduce within blocks
        size_t smem_size = ((block.x + wrap_size - 1) / wrap_size) * sizeof(T);
        sum_kernel_v8_shuffle_first_pass<<<grid, block ,smem_size>>>(d_intermediate, input, n, wrap_size);
        CUDA_CHECK(cudaGetLastError());

        // Second pass: reduce block results
        dim3 grid2(1);
        dim3 block2(min(grid.x, block.x));

        size_t smem_size2 = ((block2.x + wrap_size - 1) / wrap_size) * sizeof(T);

        sum_kernel_v8_shuffle_second_pass<<<grid2, block2 ,smem_size2>>>(result, d_intermediate, grid.x, wrap_size);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_intermediate));
    }

    template void vector_sum_v8<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);


#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)
    template<typename T>
    __global__ void sum_kernel_v9_reduce_cooperative(T *output, const T *input, size_t n, unsigned int wrap_size) {
        // (block_size + wrap_size -1) / wrap_size * sizeof(T);
        extern __shared__ __align__(sizeof(T)) unsigned char shared_mem_raw[];
        T *coop_smem = reinterpret_cast<T *>(shared_mem_raw);

        cg::grid_group grid = cg::this_grid();
        // Replace with __syncthreads
        // cg::thread_block block = cg::this_thread_block();
        size_t tid = threadIdx.x;
        size_t idx = blockIdx.x * blockDim.x + tid;

        // Block-level reduction
        T sum = 0;
        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
            sum += input[i];
        }

        T warp_sum = warp_reduce(sum, wrap_size);

        if (tid % wrap_size == 0) {
            coop_smem[tid / wrap_size] = warp_sum;
        }

        // block.sync();
        __syncthreads(); // equivalent to block.sync()

        if (tid < wrap_size) {
            T block_sum = (tid < (blockDim.x + wrap_size - 1) / wrap_size) ? coop_smem[tid] : T(0);
            block_sum = warp_reduce(block_sum, wrap_size);
            if (tid == 0) {
                // grid_size * sizeof(T)
                output[blockIdx.x] = block_sum;
            }
        }

        // Global synchronization
        grid.sync();

        if (blockIdx.x == 0) {
            T final_sum = 0;
            for (size_t i = tid; i < gridDim.x; i += blockDim.x) {
                final_sum += output[i];
            }
            T wrap_val = warp_reduce(final_sum, wrap_size);

            if (tid % wrap_size == 0) {
                // coop_sem should be larger than wrap_size * sizeof(T)
                coop_smem[tid / wrap_size] = wrap_val;
            }

            // block.sync();
            __syncthreads();

            if (tid < wrap_size) {
                T v = (tid < (blockDim.x + wrap_size - 1) / wrap_size) ? coop_smem[tid] : T(0);
                T total = warp_reduce(v, wrap_size);
                if (tid == 0) {
                    output[0] = total;
                }
            }
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void vector_sum_v9(T *result, T *input, size_t n, dim3 grid, dim3 block, unsigned int wrap_size) {
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

        int grid_size = 0;
        int block_size = block.x;
        size_t smem_size = (block.x + wrap_size - 1) / wrap_size * sizeof(T);

        CUDA_CHECK(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, sum_kernel_v9_reduce_cooperative<T>, block_size,
                smem_size));

        grid_size *= props.multiProcessorCount;
        grid_size = min(grid_size, static_cast<int>((n + block_size - 1) / block_size));
        grid_size = max(grid_size, 1);

        int can_lanch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&can_lanch, cudaDevAttrCooperativeLaunch, 0));

        if (!can_lanch) {
            std::cerr << "CUDA device does not support cooperative launch" << '\n';
            exit(EXIT_FAILURE);
        }

        T *d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(T)));


        void *kernelArgs[] = {&d_partial, &input, &n, &wrap_size};

        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void *) sum_kernel_v9_reduce_cooperative<T>,
            grid_size,
            block,
            kernelArgs,
            smem_size,
            0
        ));

        CUDA_CHECK(cudaMemcpy(result, d_partial, sizeof(T), cudaMemcpyDeviceToHost));

        if (d_partial != nullptr) {
            CUDA_CHECK(cudaFree(d_partial));
        }
    }

    template void vector_sum_v9<float>(float *result, float *input, size_t n, dim3 grid, dim3 block,
                                       unsigned int wrap_size);
#endif
} //namespace cuda_poc
