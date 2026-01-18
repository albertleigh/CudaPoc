//
// Created by Albert Li on 1/17/2026.
//

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda_poc::linear {
    constexpr int BLOCK_DIM = 32;
    // Each thread computes 8 output elements along M dimension.  Values of 4, 8, or 16 are typical in optimized
    // GEMM implementations. 8 is a sweet spot for many architectures
    constexpr int THREAD_M = 8;
    // Entry function '_ZN8cuda_poc23gemm_v4_2d_block_tilingIfEEviiiT_PKS1_S3_S1_PS1_' uses too much shared data
    // (0x10000 bytes, 0xc000 max)
    // Thus I cannot set as 8 on my GTX A500
    constexpr int THREAD_N = 4;

    // Warp tiling constants
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_TILE_M = 32; // Each warp computes 32 rows
    constexpr int WARP_TILE_N = 64; // Each warp computes 64 columns
    constexpr int WARP_TILE_K = 16; // K-dimension tile size
    constexpr int NUM_WARPS = 4; // 4 warps per block
    constexpr int WARP_THREADS_M = 4; // 4 threads along M per warp
    constexpr int WARP_THREADS_N = 8; // 8 threads along N per warp
    constexpr int WARP_THREAD_TILE_M = 8; // Each thread computes 8 rows
    constexpr int WARP_THREAD_TILE_N = 8; // Each thread computes 8 cols

    // C = alpha * A . B + beta * D
    template<typename T>
    void linear_v1(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    template<typename T>
    void linear_v2(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    template<typename T>
    void linear_v3(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    template<typename T>
    void linear_v4(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    template<typename T>
    void linear_v5(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    //  cublasHandle_t handle is already a pointer type, so pass it directly (not &handle)
    template<typename T>
    void linear_v6(cublasHandle_t handle, int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);
} // namespace cuda_poc::linear
