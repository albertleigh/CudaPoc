//
// Created by Albert Li on 1/17/2026.
//

#include "gemm_cuda.cuh"

namespace cuda_poc {
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
        __shared__ T As[LINEAR_BLOCK_DIM][LINEAR_BLOCK_DIM];
        __shared__ T bs[LINEAR_BLOCK_DIM][LINEAR_BLOCK_DIM];

        size_t row = blockIdx.y * LINEAR_BLOCK_DIM + threadIdx.y;
        size_t col = blockIdx.x * LINEAR_BLOCK_DIM + threadIdx.x;

        T tmp = T(0);

        for (int bk = 0; bk < K; bk += LINEAR_BLOCK_DIM) {
            // Cooperative loading of A and B tiles into SMEM
            int a_row = row;
            int a_col = bk + threadIdx.x;
            int b_row = bk + threadIdx.y;
            int b_col = col;
            As[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : T(0);
            bs[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : T(0);
            __syncthreads();

            for (int k = 0; k < LINEAR_BLOCK_DIM; ++k) {
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
        dim3 block(LINEAR_BLOCK_DIM, LINEAR_BLOCK_DIM);
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
        __shared__ T As[LINEAR_THREAD_M * LINEAR_BLOCK_DIM][LINEAR_BLOCK_DIM];
        __shared__ T Bs[LINEAR_BLOCK_DIM][LINEAR_BLOCK_DIM];

        // Each thread computes THREAD_M output elements
        int thread_row_base = blockIdx.y * (LINEAR_THREAD_M * LINEAR_BLOCK_DIM) + threadIdx.y;
        int thread_col = blockIdx.x * LINEAR_BLOCK_DIM + threadIdx.x;

        // Accumulator array for THREAD_M results
        T accum[LINEAR_THREAD_M];
#pragma unroll
        for (int m = 0; m < LINEAR_THREAD_M; ++m) {
            accum[m] = T(0);
        }

        // Loop over K dimension in tiles
        for (int bk = 0; bk < K; bk += LINEAR_BLOCK_DIM) {
            // Cooperative loading of A tiles (THREAD_M rows per thread)
#pragma unroll
            for (int m = 0; m < LINEAR_THREAD_M; ++m) {
                int a_row = thread_row_base + m * LINEAR_BLOCK_DIM;
                int a_col = bk + threadIdx.x;
                As[threadIdx.y + m * LINEAR_BLOCK_DIM][threadIdx.x] =
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
            for (int k = 0; k < LINEAR_BLOCK_DIM; ++k) {
                T b_val = Bs[k][threadIdx.x];
#pragma unroll
                for (int m = 0; m < LINEAR_THREAD_M; ++m) {
                    accum[m] += As[threadIdx.y + m * LINEAR_BLOCK_DIM][k] * b_val;
                }
            }
            __syncthreads();
        }

        // Write THREAD_M results with alpha/beta scaling
#pragma unroll
        for (int m = 0; m < LINEAR_THREAD_M; ++m) {
            int out_row = thread_row_base + m * LINEAR_BLOCK_DIM;
            if (out_row < M && thread_col < N) {
                C[out_row * N + thread_col] = alpha * accum[m] + beta * C[out_row * N + thread_col];
            }
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void linear_v3(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C) {
        // 1D block tiling: each thread computes THREAD_M outputs
        dim3 block(LINEAR_BLOCK_DIM, LINEAR_BLOCK_DIM);
        int grid_y = (M + LINEAR_THREAD_M * LINEAR_BLOCK_DIM - 1) / (LINEAR_THREAD_M * LINEAR_BLOCK_DIM);
        int grid_x = (N + LINEAR_BLOCK_DIM - 1) / LINEAR_BLOCK_DIM;
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
        __shared__ T As[LINEAR_THREAD_M * LINEAR_BLOCK_DIM][LINEAR_BLOCK_DIM];
        __shared__ T Bs[LINEAR_BLOCK_DIM][LINEAR_THREAD_N * LINEAR_BLOCK_DIM];

        // Each thread computes LINEAR_THREAD_M × LINEAR_THREAD_N output elements
        int thread_row_base = blockIdx.y * (LINEAR_THREAD_M * LINEAR_BLOCK_DIM) + threadIdx.y;
        int thread_col_base = blockIdx.x * (LINEAR_THREAD_N * LINEAR_BLOCK_DIM) + threadIdx.x;

        // 2D Accumulator array for LINEAR_THREAD_M × LINEAR_THREAD_N results
        T accum[LINEAR_THREAD_M][LINEAR_THREAD_N];
#pragma unroll
        for (int m = 0; m < LINEAR_THREAD_M; ++m) {
#pragma unroll
            for (int n = 0; n < LINEAR_THREAD_N; ++n) {
                accum[m][n] = T(0);
            }
        }

        // Loop over K dimension in tiles
        for (int bk = 0; bk < K; bk += LINEAR_BLOCK_DIM) {
            // Cooperative loading of A tiles (LINEAR_THREAD_M rows per thread)
#pragma unroll
            for (int m = 0; m < LINEAR_THREAD_M; ++m) {
                int a_row = thread_row_base + m * LINEAR_BLOCK_DIM;
                int a_col = bk + threadIdx.x;
                As[threadIdx.y + m * LINEAR_BLOCK_DIM][threadIdx.x] =
                        (a_row < M && a_col < K) ? A[a_row * K + a_col] : T(0);
            }

            // Cooperative loading of B tiles (LINEAR_THREAD_N columns per thread)
#pragma unroll
            for (int n = 0; n < LINEAR_THREAD_N; ++n) {
                int b_row = bk + threadIdx.y;
                int b_col = thread_col_base + n * LINEAR_BLOCK_DIM;
                Bs[threadIdx.y][threadIdx.x + n * LINEAR_BLOCK_DIM] =
                        (b_row < K && b_col < N) ? B[b_row * N + b_col] : T(0);
            }

            __syncthreads();

            // Compute LINEAR_THREAD_M × LINEAR_THREAD_N partial results
#pragma unroll
            for (int k = 0; k < LINEAR_BLOCK_DIM; ++k) {
#pragma unroll
                for (int m = 0; m < LINEAR_THREAD_M; ++m) {
                    T a_val = As[threadIdx.y + m * LINEAR_BLOCK_DIM][k];
#pragma unroll
                    for (int n = 0; n < LINEAR_THREAD_N; ++n) {
                        T b_val = Bs[k][threadIdx.x + n * LINEAR_BLOCK_DIM];
                        accum[m][n] += a_val * b_val;
                    }
                }
            }
            __syncthreads();
        }

        // Write LINEAR_THREAD_M × LINEAR_THREAD_N results with alpha/beta scaling
#pragma unroll
        for (int m = 0; m < LINEAR_THREAD_M; ++m) {
#pragma unroll
            for (int n = 0; n < LINEAR_THREAD_N; ++n) {
                int out_row = thread_row_base + m * LINEAR_BLOCK_DIM;
                int out_col = thread_col_base + n * LINEAR_BLOCK_DIM;
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
        dim3 block(LINEAR_BLOCK_DIM, LINEAR_BLOCK_DIM);
        int grid_y = (M + LINEAR_THREAD_M * LINEAR_BLOCK_DIM - 1) / (LINEAR_THREAD_M * LINEAR_BLOCK_DIM);
        int grid_x = (N + LINEAR_THREAD_N * LINEAR_BLOCK_DIM - 1) / (LINEAR_THREAD_N * LINEAR_BLOCK_DIM);
        dim3 grid(grid_x, grid_y);

        gemm_v4_2d_block_tiling<T><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }

    template void linear_v4<float>(int M, int N, int K,
                                   float alpha, const float *A, const float *B, float beta, float *C);
}
