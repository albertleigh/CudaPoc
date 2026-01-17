//
// Created by Albert Li on 1/17/2026.
//

#pragma once

#include <cuda_runtime.h>

namespace cuda_poc {
    constexpr int LINEAR_BLOCK_DIM = 32;
    // Each thread computes 8 output elements along M dimension.  Values of 4, 8, or 16 are typical in optimized
    // GEMM implementations. 8 is a sweet spot for many architectures
    constexpr int LINEAR_THREAD_M = 8;
    // Entry function '_ZN8cuda_poc23gemm_v4_2d_block_tilingIfEEviiiT_PKS1_S3_S1_PS1_' uses too much shared data
    // (0x10000 bytes, 0xc000 max)
    // Thus I cannot set as 8 on my GTX A500
    constexpr int LINEAR_THREAD_N = 4;

    // C = alpha * A . B + beta * D
    template<typename T>
    void linear_v1(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    template<typename T>
    void linear_v2(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    template<typename T>
    void linear_v3(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);

    template<typename T>
    void linear_v4(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);
} // namespace cuda_poc
