//
// Created by Albert Li on 1/14/2026.
//
#include <gtest/gtest.h>
#include "add_cuda.cuh"
#include "cuda_utils.h"

namespace cuda_poc::day01 {
using namespace cuda_poc;

TEST(CudaPoc_Day0101, HelloCuda) {
  const size_t SIZE = 1 << 20;  // 4MB
  size_t size_bytes = SIZE * sizeof(float);

  dim3 block_dim(256);
  dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

  std::vector<float> h_a(SIZE, 1);
  std::vector<float> h_b(SIZE, 2);
  std::vector<float> h_c(SIZE, 0);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, size_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, size_bytes));

  KernelConfig config(grid_dim, block_dim);
  timeKernel(
      "vector_add",
      [&]() {
        // === Kernel: vector_add ===
        // Execution time: 0.328128 ms
        // Grid dimensions: (4096, 1, 1)
        // Block dimensions: (256, 1, 1)
        // Total threads: 1048576
        // GPU memory used: 0 MB
        // GPU memory free: 3303.55 MB / 4095.69 MB
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));
        vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
      },
      &config);

  for (size_t i = 0; i < SIZE; ++i) {
    ASSERT_EQ(h_c[i], 3.0f) << "Verification failed at index " << i;
  }

  if (d_a) {
    CUDA_CHECK(cudaFree(d_a));
  }
  if (d_b) {
    CUDA_CHECK(cudaFree(d_b));
  }
  if (d_c) {
    CUDA_CHECK(cudaFree(d_c));
  }
}
}  // namespace cuda_poc::day01
