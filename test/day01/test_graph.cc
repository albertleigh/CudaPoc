//
// Created by ali on 1/25/26.
//

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <cmath>
#include "cuda_utils.h"
#include "sample_graph_cuda.cuh"

namespace cuda_poc::day01 {
using namespace cuda_poc::sample_graph;

// === Kernel: regular_approach ===
// Execution time: 13.4007 ms
// Grid dimensions: (4096, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 1048576
// GPU memory used: 30 MB
// GPU memory free: 3268.25 MB / 4095.56 MB
//
// === Kernel: graph_approach ===
// Execution time: 2.58957 ms
// Grid dimensions: (4096, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 1048576
// GPU memory used: 0 MB
// GPU memory free: 3268.25 MB / 4095.56 MB
class CudaPoc_Day0503_Graph : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
      FAIL() << fmt::format("Failed to get device count: {}", cudaGetErrorString(error));
    }
  }

  void free_device_memory(float* d_data) {
    if (d_data) {
      CUDA_CHECK(cudaFree(d_data));
    }
  }
};

TEST_F(CudaPoc_Day0503_Graph, RegularApproach) {
  constexpr size_t SIZE = 1 << 20;  // 1M elements
  size_t size_bytes = SIZE * sizeof(float);
  constexpr int block_size = 256;

  // Initialize host data
  std::vector<float> h_data(SIZE, 1.0f);

  // Allocate device memory
  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, size_bytes));

  // Create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  dim3 block_dim(block_size);
  dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);
  KernelConfig config(grid_dim, block_dim);

  timeKernel(
      "regular_approach",
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size_bytes, cudaMemcpyHostToDevice));
        regular_approach(d_data, SIZE, block_size, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, size_bytes, cudaMemcpyDeviceToHost));
      },
      &config);

  // Verify results: sin(1) + 1, then sqrt, then *2, then -1
  // Expected: (sqrt(sin(1) + 1) * 2) - 1 ≈ 1.714
  float expected = (std::sqrt(std::sin(1.0f) + 1.0f) * 2.0f) - 1.0f;
  EXPECT_NEAR(h_data[0], expected, 1e-5);
  EXPECT_NEAR(h_data[SIZE - 1], expected, 1e-5);

  CUDA_CHECK(cudaStreamDestroy(stream));
  free_device_memory(d_data);
}

TEST_F(CudaPoc_Day0503_Graph, GraphApproach) {
  constexpr size_t SIZE = 1 << 20;  // 1M elements
  size_t size_bytes = SIZE * sizeof(float);
  constexpr int block_size = 256;

  // Initialize host data
  std::vector<float> h_data(SIZE, 1.0f);

  // Allocate device memory
  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, size_bytes));

  // Create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Create graph
  cudaGraphExec_t graph_exec;
  create_graph(&graph_exec, d_data, SIZE, block_size);

  dim3 block_dim(block_size);
  dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);
  KernelConfig config(grid_dim, block_dim);

  timeKernel(
      "graph_approach",
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size_bytes, cudaMemcpyHostToDevice));
        execute_graph(graph_exec, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, size_bytes, cudaMemcpyDeviceToHost));
      },
      &config);

  // Verify results: sin(1) + 1, then sqrt, then *2, then -1
  // Expected: (sqrt(sin(1) + 1) * 2) - 1 ≈ 1.714
  float expected = (std::sqrt(std::sin(1.0f) + 1.0f) * 2.0f) - 1.0f;
  EXPECT_NEAR(h_data[0], expected, 1e-5);
  EXPECT_NEAR(h_data[SIZE - 1], expected, 1e-5);

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaStreamDestroy(stream));
  free_device_memory(d_data);
}

}  // namespace cuda_poc::day01
