//
// Created by Albert Li on 1/17/2026.
//

#pragma once

#include <cuda_runtime.h>

namespace cuda_poc {
    // C = alpha * A . B + beta * D
    template<typename T>
    void linear_v1(int M, int N, int K, T alpha, const T *A, const T *B, T beta, T *C);
} // namespace cuda_poc
