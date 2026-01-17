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
}
