//
// Created by Albert Li on 1/14/2026.
//
#include <gtest/gtest.h>
#include "add_cuda.cuh"
#include "cuda_utils.h"

namespace cuda_poc::day01 {
using namespace cuda_poc;

void reset_vector_flat(std::vector<float>& target, const size_t size_bytes) {
  for (size_t i = 0; i < size_bytes; ++i) {
    target[i] = 0;
  }
}

void assert_vector_float_equal(const std::vector<float>& target, const size_t size_bytes, const float expected) {
  for (size_t i = 0; i < size_bytes; ++i) {
    ASSERT_EQ(target[i], expected) << "Verification failed at index " << i;
  }
}

// === Kernel: vector_add ===
// Execution time: 3.322304 ms
// Grid dimensions: (4096, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 1048576
// GPU memory used: 0 MB
// GPU memory free: 3290.25 MB / 4095.5625 MB
//
// === Kernel: vector_one_by_one ===
// Execution time: 96.39229 ms
// Grid dimensions: (1, 1, 1)
// Block dimensions: (1, 1, 1)
// Total threads: 1
// GPU memory used: 0 MB
// GPU memory free: 3290.25 MB / 4095.5625 MB
//
// === Kernel: vector_max ===
// Execution time: 3.25808 ms
// Grid dimensions: (1024, 1, 1)
// Block dimensions: (1024, 1, 1)
// Total threads: 1048576
// GPU memory used: 0 MB
// GPU memory free: 3290.25 MB / 4095.5625 MB
// Process finished with exit code 0
TEST(CudaPoc_Day0102, CompareInDims) {
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
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));
        vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
      },
      &config);

  assert_vector_float_equal(h_c, SIZE, 3.0f);
  reset_vector_flat(h_c, SIZE);

  grid_dim = dim3(1);
  block_dim = dim3(1);
  config = KernelConfig(grid_dim, block_dim);

  timeKernel(
      "vector_one_by_one",
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));
        vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
      },
      &config);

  assert_vector_float_equal(h_c, SIZE, 3.0f);
  reset_vector_flat(h_c, SIZE);

  grid_dim = dim3(1024);
  block_dim = dim3(1024);
  config = KernelConfig(grid_dim, block_dim);
  timeKernel(
      "vector_max",
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));
        vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
      },
      &config);
  assert_vector_float_equal(h_c, SIZE, 3.0f);

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
