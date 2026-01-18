//
// Created by Albert Li on 1/17/2026.
//

#include "gemm_cuda.cuh"

namespace cuda_poc::linear {
    // C = alpha * A . B + beta * D

    template<typename T>
    __global__ void gemm_v1_naive(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        size_t row = blockIdx.x * blockDim.x + threadIdx.x;
        size_t col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < M && col < N) {
            T sum = T(0);
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void linear_v1(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // naive support
        dim3 block(32, 32);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        gemm_v1_naive<T><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }

    template void linear_v1<float>(int M, int N, int K,
                                   float alpha, const float *A, const float *B, float beta, float *C);

    template<typename T>
    __global__ void gemm_v2_smem(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        __shared__ T As[BLOCK_DIM][BLOCK_DIM];
        __shared__ T bs[BLOCK_DIM][BLOCK_DIM];

        size_t row = blockIdx.y * BLOCK_DIM + threadIdx.y;
        size_t col = blockIdx.x * BLOCK_DIM + threadIdx.x;

        T tmp = T(0);

        for (int bk = 0; bk < K; bk += BLOCK_DIM) {
            // Cooperative loading of A and B tiles into SMEM
            int a_row = row;
            int a_col = bk + threadIdx.x;
            int b_row = bk + threadIdx.y;
            int b_col = col;
            As[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : T(0);
            bs[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : T(0);
            __syncthreads();

            for (int k = 0; k < BLOCK_DIM; ++k) {
                tmp += As[threadIdx.y][k] * bs[k][threadIdx.x];
            }
            __syncthreads();
        }

        // Write results with alpha/beta scaling
        if (row < M && col < N) {
            C[row * N + col] = alpha * tmp + beta * C[row * N + col];
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void linear_v2(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // naive support
        dim3 block(BLOCK_DIM, BLOCK_DIM);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        gemm_v2_smem<T><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }

    template void linear_v2<float>(int M, int N, int K,
                                   float alpha, const float *A, const float *B, float beta, float *C);

    // 1D Block Tiling: each thread computes THREAD_M elements along M dimension
    // Shared Memory Reuse (Main Benefit)
    //
    // Naive version: Each B tile element from shared memory → used once → 1 FMA operation
    // 1D tiling version: Each B tile element → reused 8 times → 8 FMA operations
    // This reduces shared memory traffic by 8× for the inner computation loop
    // Arithmetic Intensity
    //
    // More compute per memory access
    // Ratio improves from 1:1 to 8:1 (operations:memory_loads)
    // Shifts bottleneck from memory-bound toward compute-bound (desired)
    // Register Blocking
    //
    // accum[LINEAR_THREAD_M] lives in registers (fastest memory)
    // Multiple accumulations happen entirely in registers
    // No repeated shared memory writes/reads for intermediate results
    // Addresses MIO Throttle
    //
    // cd C:\Ws\CudaPoc\CudaPoc\src; $vsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat'; cmd /c "`"$vsPath`" && nvcc --ptxas-options=-v -c gemm_cuda.cu 2>&1"
    // **********************************************************************
    // ** Visual Studio 2022 Developer Command Prompt v17.14.24
    // ** Copyright (c) 2025 Microsoft Corporation
    // **********************************************************************
    // [vcvarsall.bat] Environment initialized for: 'x64'
    // gemm_cuda.cu
    // ptxas info    : 0 bytes gmem
    // ptxas info    : Compiling entry function '_ZN8cuda_poc23gemm_v3_1d_block_tilingIfEEviiiT_PKS1_S3_S1_PS1_' for 'sm_75'
    // ptxas info    : Function properties for _ZN8cuda_poc23gemm_v3_1d_block_tilingIfEEviiiT_PKS1_S3_S1_PS1_
    //    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    // ptxas info    : Used 72 registers, used 1 barriers, 36864 bytes smem, 400 bytes cmem[0]
    // ptxas info    : Compile time = 0.000 ms

    template<typename T>
    __global__ void gemm_v3_1d_block_tiling(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        __shared__ T As[THREAD_M * BLOCK_DIM][BLOCK_DIM];
        __shared__ T Bs[BLOCK_DIM][BLOCK_DIM];

        // Each thread computes THREAD_M output elements
        int thread_row_base = blockIdx.y * (THREAD_M * BLOCK_DIM) + threadIdx.y;
        int thread_col = blockIdx.x * BLOCK_DIM + threadIdx.x;

        // Accumulator array for THREAD_M results
        T accum[THREAD_M];
#pragma unroll
        for (int m = 0; m < THREAD_M; ++m) {
            accum[m] = T(0);
        }

        // Loop over K dimension in tiles
        for (int bk = 0; bk < K; bk += BLOCK_DIM) {
            // Cooperative loading of A tiles (THREAD_M rows per thread)
#pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                int a_row = thread_row_base + m * BLOCK_DIM;
                int a_col = bk + threadIdx.x;
                As[threadIdx.y + m * BLOCK_DIM][threadIdx.x] =
                        (a_row < M && a_col < K) ? A[a_row * K + a_col] : T(0);
            }

            // Cooperative loading of B tile
            int b_row = bk + threadIdx.y;
            int b_col = thread_col;
            Bs[threadIdx.y][threadIdx.x] =
                    (b_row < K && b_col < N) ? B[b_row * N + b_col] : T(0);

            __syncthreads();

            // Compute THREAD_M partial results
#pragma unroll
            for (int k = 0; k < BLOCK_DIM; ++k) {
                T b_val = Bs[k][threadIdx.x];
#pragma unroll
                for (int m = 0; m < THREAD_M; ++m) {
                    accum[m] += As[threadIdx.y + m * BLOCK_DIM][k] * b_val;
                }
            }
            __syncthreads();
        }

        // Write THREAD_M results with alpha/beta scaling
#pragma unroll
        for (int m = 0; m < THREAD_M; ++m) {
            int out_row = thread_row_base + m * BLOCK_DIM;
            if (out_row < M && thread_col < N) {
                C[out_row * N + thread_col] = alpha * accum[m] + beta * C[out_row * N + thread_col];
            }
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void linear_v3(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // 1D block tiling: each thread computes THREAD_M outputs
        dim3 block(BLOCK_DIM, BLOCK_DIM);
        int grid_y = (M + THREAD_M * BLOCK_DIM - 1) / (THREAD_M * BLOCK_DIM);
        int grid_x = (N + BLOCK_DIM - 1) / BLOCK_DIM;
        dim3 grid(grid_x, grid_y);

        gemm_v3_1d_block_tiling<T><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }

    template void linear_v3<float>(int M, int N, int K,
                                   float alpha, const float *A, const float *B, float beta, float *C);

    // 2D Block Tiling: each thread computes LINEAR_THREAD_M × LINEAR_THREAD_N output elements
    // Further improves memory reuse by tiling in both M and N dimensions
    // Each A tile element is reused LINEAR_THREAD_N times
    // Each B tile element is reused LINEAR_THREAD_M times
    template<typename T>
    __global__ void gemm_v4_2d_block_tiling(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        __shared__ T As[THREAD_M * BLOCK_DIM][BLOCK_DIM];
        __shared__ T Bs[BLOCK_DIM][THREAD_N * BLOCK_DIM];

        // Each thread computes LINEAR_THREAD_M × LINEAR_THREAD_N output elements
        int thread_row_base = blockIdx.y * (THREAD_M * BLOCK_DIM) + threadIdx.y;
        int thread_col_base = blockIdx.x * (THREAD_N * BLOCK_DIM) + threadIdx.x;

        // 2D Accumulator array for LINEAR_THREAD_M × LINEAR_THREAD_N results
        T accum[THREAD_M][THREAD_N];
#pragma unroll
        for (int m = 0; m < THREAD_M; ++m) {
#pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                accum[m][n] = T(0);
            }
        }

        // Loop over K dimension in tiles
        for (int bk = 0; bk < K; bk += BLOCK_DIM) {
            // Cooperative loading of A tiles (LINEAR_THREAD_M rows per thread)
#pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                int a_row = thread_row_base + m * BLOCK_DIM;
                int a_col = bk + threadIdx.x;
                As[threadIdx.y + m * BLOCK_DIM][threadIdx.x] =
                        (a_row < M && a_col < K) ? A[a_row * K + a_col] : T(0);
            }

            // Cooperative loading of B tiles (LINEAR_THREAD_N columns per thread)
#pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                int b_row = bk + threadIdx.y;
                int b_col = thread_col_base + n * BLOCK_DIM;
                Bs[threadIdx.y][threadIdx.x + n * BLOCK_DIM] =
                        (b_row < K && b_col < N) ? B[b_row * N + b_col] : T(0);
            }

            __syncthreads();

            // Compute LINEAR_THREAD_M × LINEAR_THREAD_N partial results
#pragma unroll
            for (int k = 0; k < BLOCK_DIM; ++k) {
#pragma unroll
                for (int m = 0; m < THREAD_M; ++m) {
                    T a_val = As[threadIdx.y + m * BLOCK_DIM][k];
#pragma unroll
                    for (int n = 0; n < THREAD_N; ++n) {
                        T b_val = Bs[k][threadIdx.x + n * BLOCK_DIM];
                        accum[m][n] += a_val * b_val;
                    }
                }
            }
            __syncthreads();
        }

        // Write LINEAR_THREAD_M × LINEAR_THREAD_N results with alpha/beta scaling
#pragma unroll
        for (int m = 0; m < THREAD_M; ++m) {
#pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                int out_row = thread_row_base + m * BLOCK_DIM;
                int out_col = thread_col_base + n * BLOCK_DIM;
                if (out_row < M && out_col < N) {
                    C[out_row * N + out_col] = alpha * accum[m][n] + beta * C[out_row * N + out_col];
                }
            }
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void linear_v4(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // 2D block tiling: each thread computes LINEAR_THREAD_M × LINEAR_THREAD_N outputs
        dim3 block(BLOCK_DIM, BLOCK_DIM);
        int grid_y = (M + THREAD_M * BLOCK_DIM - 1) / (THREAD_M * BLOCK_DIM);
        int grid_x = (N + THREAD_N * BLOCK_DIM - 1) / (THREAD_N * BLOCK_DIM);
        dim3 grid(grid_x, grid_y);

        gemm_v4_2d_block_tiling<T><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }

    template void linear_v4<float>(int M, int N, int K,
                                   float alpha, const float *A, const float *B, float beta, float *C);

    // Warp Tiling Implementation:
    // - Block = 128 threads = 4 warps arranged vertically
    // - Each warp processes a 32×64 output tile
    // - Each thread in warp processes 8×8 output elements
    // - Total block coverage: 128×64 (4 warps * 32 rows × 64 cols)
    // Benefits:
    // - Reduced synchronization (warps execute in lockstep)
    // - Better instruction-level parallelism
    // - Improved register reuse within warps
    // - More efficient shared memory access patterns
    template<typename T>
    __global__ void gemm_v5_warptiling(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // Shared memory for block-level tiles
        // Block handles 4 warps stacked vertically = 128×64 output
        constexpr int BLOCK_TILE_M = NUM_WARPS * WARP_TILE_M; // 4 * 32 = 128
        constexpr int BLOCK_TILE_N = WARP_TILE_N; // 64

        __shared__ T As[BLOCK_TILE_M][WARP_TILE_K];
        __shared__ T Bs[WARP_TILE_K][BLOCK_TILE_N];

        // Total threads per block = 4 warps × 32 threads = 128 threads
        constexpr int THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE;

        // Warp and thread identification
        int warp_id = threadIdx.x / WARP_SIZE; // 0-3
        int lane_id = threadIdx.x % WARP_SIZE; // 0-31

        // Within warp: 4×8 thread grid
        int thread_m = lane_id / WARP_THREADS_N; // 0-3
        int thread_n = lane_id % WARP_THREADS_N; // 0-7

        // Global position of this thread's 8×8 tile
        int block_row_base = blockIdx.y * BLOCK_TILE_M;
        int block_col_base = blockIdx.x * BLOCK_TILE_N;

        int thread_row_base = block_row_base + warp_id * WARP_TILE_M + thread_m * WARP_THREAD_TILE_M;
        int thread_col_base = block_col_base + thread_n * WARP_THREAD_TILE_N;

        // Register accumulator for 8×8 tile
        T accum[WARP_THREAD_TILE_M][WARP_THREAD_TILE_N];
#pragma unroll
        for (int m = 0; m < WARP_THREAD_TILE_M; ++m) {
#pragma unroll
            for (int n = 0; n < WARP_THREAD_TILE_N; ++n) {
                accum[m][n] = T(0);
            }
        }

        // Loop over K dimension
        for (int bk = 0; bk < K; bk += WARP_TILE_K) {
            // Cooperative loading of A tile: 128×16 elements
            // 128 threads load (128×16)/128 = 16 elements each
            int elements_per_thread = (BLOCK_TILE_M * WARP_TILE_K) / THREADS_PER_BLOCK;
#pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                int idx = threadIdx.x + i * THREADS_PER_BLOCK;
                int a_row = idx / WARP_TILE_K;
                int a_col = idx % WARP_TILE_K;
                int global_row = block_row_base + a_row;
                int global_col = bk + a_col;
                As[a_row][a_col] = (global_row < M && global_col < K) ? A[global_row * K + global_col] : T(0);
            }

            // Cooperative loading of B tile: 16×128 elements
            elements_per_thread = (WARP_TILE_K * BLOCK_TILE_N) / THREADS_PER_BLOCK;
#pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                int idx = threadIdx.x + i * THREADS_PER_BLOCK;
                int b_row = idx / BLOCK_TILE_N;
                int b_col = idx % BLOCK_TILE_N;
                int global_row = bk + b_row;
                int global_col = block_col_base + b_col;
                Bs[b_row][b_col] = (global_row < K && global_col < N) ? B[global_row * N + global_col] : T(0);
            }

            __syncthreads();

            // Warp-level computation: each thread computes its 8×8 tile
            int smem_row_base = warp_id * WARP_TILE_M + thread_m * WARP_THREAD_TILE_M;
            int smem_col_base = thread_n * WARP_THREAD_TILE_N;

#pragma unroll
            for (int k = 0; k < WARP_TILE_K; ++k) {
#pragma unroll
                for (int m = 0; m < WARP_THREAD_TILE_M; ++m) {
                    T a_val = As[smem_row_base + m][k];
#pragma unroll
                    for (int n = 0; n < WARP_THREAD_TILE_N; ++n) {
                        T b_val = Bs[k][smem_col_base + n];
                        accum[m][n] += a_val * b_val;
                    }
                }
            }
            __syncthreads();
        }

        // Write results: each thread writes its 8×8 tile
#pragma unroll
        for (int m = 0; m < WARP_THREAD_TILE_M; ++m) {
#pragma unroll
            for (int n = 0; n < WARP_THREAD_TILE_N; ++n) {
                int out_row = thread_row_base + m;
                int out_col = thread_col_base + n;
                if (out_row < M && out_col < N) {
                    C[out_row * N + out_col] = alpha * accum[m][n] + beta * C[out_row * N + out_col];
                }
            }
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void linear_v5(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // Warp tiling: 128 threads per block (4 warps), each block processes 128×64 output
        constexpr int BLOCK_TILE_M = NUM_WARPS * WARP_TILE_M; // 4 * 32 = 128
        constexpr int BLOCK_TILE_N = WARP_TILE_N; // 64
        constexpr int THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE; // 128

        dim3 block(THREADS_PER_BLOCK, 1);
        int grid_y = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
        int grid_x = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
        dim3 grid(grid_x, grid_y);

        gemm_v5_warptiling<T><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }

    template void linear_v5<float>(int M, int N, int K,
                                   float alpha, const float *A, const float *B, float beta, float *C);


    // cuBLAS wrapper function
    // Note: cuBLAS uses column-major order, our code uses row-major
    // Row-major: C = alpha * A * B + beta * C (C is M×N, A is M×K, B is K×N)
    // Column-major equivalent: C^T = alpha * B^T * A^T + beta * C^T
    // So we swap A↔B and M↔N when calling cuBLAS
    template<typename T>
    void linear_v6(cublasHandle_t handle, int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // cuBLAS SGEMM: C = alpha * op(A) * op(B) + beta * C
        // For row-major to column-major conversion:
        // Swap A and B, swap M and N
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, // Swapped dimensions
                    &alpha,
                    B, N, // B becomes first operand with leading dimension N
                    A, K, // A becomes second operand with leading dimension K
                    &beta,
                    C, N); // C with leading dimension N
    }

    template void linear_v6<float>(cublasHandle_t handle, int M, int N, int K,
                                   float alpha, const float *A, const float *B, float beta, float *C);
}
