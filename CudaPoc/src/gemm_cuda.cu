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
            T sum = 0;
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
                                   float alpha, const float *A, const float *B, float beta,
                                   float *C);
}
