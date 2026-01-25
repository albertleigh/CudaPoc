//
// Created by Albert Li on 1/17/2026.
//

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda_poc::linear {
constexpr int BLOCK_DIM = 32;
// Each thread computes THREAD_M output elements along M dimension
// Architecture limits and actual register usage (compiler optimizations vary!):
//   sm_75 (Turing):  65536 regs/block, 48KB smem/block
//   sm_86 (Ampere):  65536 regs/block, 99KB smem/block (164KB configurable)
//   sm_89 (Ada):     65536 regs/block, 99KB smem/block (227KB configurable)
// With 32x32 block (1024 threads):
//   THREAD_M=8, THREAD_N=4: Works on sm_86+, FAILS on sm_75 (exceeds register limit)
//   THREAD_M=4, THREAD_N=4: Works on ALL architectures
// Shared memory: As[THREAD_M*32][32] + Bs[32][THREAD_N*32]
//   THREAD_M=8, THREAD_N=4: 32KB + 16KB = 48KB (at limit on sm_75, plenty on sm_86+)
//   THREAD_M=4, THREAD_N=4: 16KB + 16KB = 32KB (safe on all)
// Using conservative THREAD_M=4 for compatibility with sm_75
#if __CUDA_ARCH__ >= 860
constexpr int THREAD_M = 8;  // Ampere and newer
#else
constexpr int THREAD_M = 4;  // Turing and older
#endif
constexpr int THREAD_N = 4;

// Warp tiling constants
constexpr int WARP_SIZE = 32;
constexpr int WARP_TILE_M = 32;        // Each warp computes 32 rows
constexpr int WARP_TILE_N = 64;        // Each warp computes 64 columns
constexpr int WARP_TILE_K = 16;        // K-dimension tile size
constexpr int NUM_WARPS = 4;           // 4 warps per block
constexpr int WARP_THREADS_M = 4;      // 4 threads along M per warp
constexpr int WARP_THREADS_N = 8;      // 8 threads along N per warp
constexpr int WARP_THREAD_TILE_M = 8;  // Each thread computes 8 rows
constexpr int WARP_THREAD_TILE_N = 8;  // Each thread computes 8 cols

// C = alpha * A . B + beta * D
template <typename T>
void linear_v1(int M, int N, int K, T alpha, const T* A, const T* B, T beta, T* C);

template <typename T>
void linear_v2(int M, int N, int K, T alpha, const T* A, const T* B, T beta, T* C);

template <typename T>
void linear_v3(int M, int N, int K, T alpha, const T* A, const T* B, T beta, T* C);

template <typename T>
void linear_v4(int M, int N, int K, T alpha, const T* A, const T* B, T beta, T* C);

template <typename T>
void linear_v5(int M, int N, int K, T alpha, const T* A, const T* B, T beta, T* C);

//  cublasHandle_t handle is already a pointer type, so pass it directly (not &handle)
template <typename T>
void linear_v6(cublasHandle_t handle, int M, int N, int K, T alpha, const T* A, const T* B, T beta, T* C);

// CuTe GEMM implementation
template <typename T>
void linear_v7(int M, int N, int K, T alpha, const T* A, const T* B, T beta, T* C);
}  // namespace cuda_poc::linear
