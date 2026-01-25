//
// Created by Albert Li on 1/18/2026.
//
#include <fmt/format.h>
#include <gtest/gtest.h>
#include "depth_cuda.cuh"

namespace cuda_poc::day01 {
class CudaPoc_Day0402 : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
      FAIL() << fmt::format("Failed to get device count: {}", cudaGetErrorString(error));
    }
    fmt::println("Found {} CUDA device(s)", deviceCount);
    cudaDeviceProp prop{};
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) {
      FAIL() << fmt::format("Failed to get device properties: {}", cudaGetErrorString(error));
    }
    fmt::println("Device: {}", prop.name);

    // Set stack size for dynamic parallelism (default is usually 1KB, increase if needed)
    error = cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    if (error != cudaSuccess) {
      FAIL() << fmt::format("Failed to set cudaLimitStackSize: {}", cudaGetErrorString(error));
    }

    // Set device heap size for dynamic parallelism kernel launches (increase if many nested launches)
    error = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    if (error != cudaSuccess) {
      FAIL() << fmt::format("Failed to set cudaLimitMallocHeapSize: {}", cudaGetErrorString(error));
    }
  }
};

// Depth 0: Block 0 Thread 0
// Depth 1: Block 0 Thread 0
// Depth 1: Block 0 Thread 1
// Depth 1: Block 1 Thread 0
// Depth 1: Block 1 Thread 1
// Depth 2: Block 0 Thread 0
// Depth 2: Block 0 Thread 1
// Depth 2: Block 1 Thread 0
// Depth 2: Block 1 Thread 1
// Depth 2: Block 0 Thread 0
// Depth 2: Block 0 Thread 1
// Depth 2: Block 1 Thread 0
// Depth 2: Block 1 Thread 1
// Depth 2: Block 1 Thread 0
// Depth 2: Block 1 Thread 1
// Depth 2: Block 0 Thread 0
// Depth 2: Block 0 Thread 1
// Depth 2: Block 1 Thread 0
// Depth 2: Block 1 Thread 1
// Depth 2: Block 0 Thread 0
// Depth 2: Block 0 Thread 1
TEST_F(CudaPoc_Day0402, DepthVisit_0_2) {
  cuda_poc::depth::depth_visit(0, 2);
  cudaError_t error = cudaDeviceSynchronize();
  EXPECT_EQ(error, cudaSuccess) << fmt::format("CUDA error: {}", cudaGetErrorString(error));
}
}  // namespace cuda_poc::day01
