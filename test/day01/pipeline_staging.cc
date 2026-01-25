//
// Created by Albert Li on 1/24/2026.
//

#include <fmt/format.h>
#include <gtest/gtest.h>
#include "cuda_utils.h"
#include "pipeline_cuda.cuh"

namespace cuda_poc::day01 {

class CudaPoc_Day0501_Pipeline : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
      FAIL() << fmt::format("Failed to get device count: {}", cudaGetErrorString(error));
    }
  }

  void free_device_memory(int* d_input, int* d_output) {
    if (d_input) {
      CUDA_CHECK(cudaFree(d_input));
    }
    if (d_output) {
      CUDA_CHECK(cudaFree(d_output));
    }
  }
};

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)
using namespace cuda_poc::pipeline;

// === Kernel: compute_with_staging_in_one_block ===
// Execution time: 19.4987 ms
// Grid dimensions: (1, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 256
// GPU memory used: 30 MB
// GPU memory free: 3270.25 MB / 4095.56 MB
//
// === Kernel: compute_with_staging_multi_block ===
// Execution time: 0.124256 ms
// Grid dimensions: (4, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 1024
// GPU memory used: 0 MB
// GPU memory free: 3270.25 MB / 4095.56 MB
//
// === Kernel: sync_compute ===
// Execution time: 0.155936 ms  <- You need heavier computation. Try replacing the trivial +1 with something more
// expensive: Grid dimensions: (4, 1, 1) Block dimensions: (256, 1, 1) Total threads: 1024 GPU memory used: 0 MB GPU
// memory free: 3270.25 MB / 4095.56 MB Clock cycles: 7099 (0.004619 ms)
//
// === Kernel: SyncCopy ===
// Execution time: 49.9168 ms
// Grid dimensions: (16384, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 4194304
// GPU memory used: 0 MB
// GPU memory free: 3014.25 MB / 4095.56 MB
// Clock cycles: 367869712 (239.342688 ms)
//
// === Kernel: AsyncCopy ===
// Execution time: 48.5484 ms
// Grid dimensions: (16384, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 4194304
// GPU memory used: 0 MB
// GPU memory free: 3014.25 MB / 4095.56 MB
// Clock cycles: 289900064 (188.614225 ms)

TEST_F(CudaPoc_Day0501_Pipeline, ComputeWithStaging) {
  constexpr size_t BLOCK_SIZE = 256;
  constexpr size_t BATCH_SIZE = 4;
  constexpr size_t NUM_BLOCKS = 1;
  constexpr size_t SIZE = BLOCK_SIZE * BATCH_SIZE * NUM_BLOCKS;  // Total elements

  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim(NUM_BLOCKS);

  // Prepare host data
  std::vector<int> h_input(SIZE);
  std::vector<int> h_output(SIZE, 0);

  // Initialize input with sequential values
  for (size_t i = 0; i < SIZE; ++i) {
    h_input[i] = static_cast<int>(i);
  }

  // Allocate device memory
  int* d_input = nullptr;
  int* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_output, 0, SIZE * sizeof(int)));

  KernelConfig config(grid_dim, block_dim);
  timeKernel(
      "compute_with_staging_in_one_block",
      [&]() {
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel
        compute_with_staging(d_output, d_input, SIZE, BATCH_SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy output back to host
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, SIZE * sizeof(int), cudaMemcpyDeviceToHost));
      },
      &config);

  // Verify results: output should be input + 1 (due to compute function)
  for (size_t i = 0; i < SIZE; ++i) {
    EXPECT_EQ(h_output[i], h_input[i] + 1)
        << fmt::format("Mismatch at index {}: expected {}, got {}", i, h_input[i] + 1, h_output[i]);
  }

  free_device_memory(d_input, d_output);
}

TEST_F(CudaPoc_Day0501_Pipeline, ComputeWithStagingMultiBlock) {
  constexpr size_t BLOCK_SIZE = 256;
  constexpr size_t BATCH_SIZE = 8;
  constexpr size_t NUM_BLOCKS = 4;
  constexpr size_t SIZE = BLOCK_SIZE * BATCH_SIZE * NUM_BLOCKS;

  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim(NUM_BLOCKS);

  // Prepare host data
  std::vector<int> h_input(SIZE);
  std::vector<int> h_output(SIZE, 0);

  for (size_t i = 0; i < SIZE; ++i) {
    h_input[i] = static_cast<int>(i % 1000);
  }

  // Allocate device memory
  int* d_input = nullptr;
  int* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_output, 0, SIZE * sizeof(int)));

  KernelConfig config(grid_dim, block_dim);
  timeKernel(
      "compute_with_staging_multi_block",
      [&]() {
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice));

        compute_with_staging(d_output, d_input, SIZE, BATCH_SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, SIZE * sizeof(int), cudaMemcpyDeviceToHost));
      },
      &config);

  // Verify results
  for (size_t i = 0; i < SIZE; ++i) {
    EXPECT_EQ(h_output[i], h_input[i] + 1) << fmt::format("Mismatch at index {}", i);
  }

  free_device_memory(d_input, d_output);
}

#endif

TEST_F(CudaPoc_Day0501_Pipeline, SyncCompute) {
  constexpr size_t BLOCK_SIZE = 256;
  constexpr size_t COPY_COUNT = 8;  // Each thread copies 8 elements
  constexpr size_t NUM_BLOCKS = 4;
  constexpr size_t SIZE = BLOCK_SIZE * COPY_COUNT * NUM_BLOCKS;

  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim(NUM_BLOCKS);

  // Prepare host data
  std::vector<int> h_input(SIZE);
  std::vector<int> h_output(SIZE, 0);

  // Initialize input with sequential values
  for (size_t i = 0; i < SIZE; ++i) {
    h_input[i] = static_cast<int>(i);
  }

  // Allocate device memory
  int* d_input = nullptr;
  int* d_output = nullptr;
  uint64_t* d_clock = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_clock, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_output, 0, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_clock, 0, sizeof(uint64_t)));

  KernelConfig config(grid_dim, block_dim);
  timeKernel(
      "sync_compute",
      [&]() {
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice));

        // Launch sync_compute kernel
        sync_compute<int>(d_input, d_output, d_clock, COPY_COUNT, SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy output back to host
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, SIZE * sizeof(int), cudaMemcpyDeviceToHost));
      },
      &config);

  // Verify results: output should match input (sync_compute just copies data)
  for (size_t i = 0; i < SIZE; ++i) {
    EXPECT_EQ(h_output[i], h_input[i] + 1)
        << fmt::format("Mismatch at index {}: expected {}, got {}", i, h_input[i], h_output[i]);
  }

  // Verify clock was updated
  uint64_t h_clock = 0;
  CUDA_CHECK(cudaMemcpy(&h_clock, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Get GPU clock rate to convert cycles to seconds
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
  double seconds = static_cast<double>(h_clock) / (static_cast<double>(clock_rate_khz) * 1000.0);

  fmt::print("Clock cycles: {} ({:.6f} ms)\n", h_clock, seconds * 1000.0);
  EXPECT_GT(h_clock, 0) << "Clock should be non-zero after kernel execution";

  // Free device memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_clock));
}

void test_copy_with_kernel_fun(const char* test_name,
                               std::function<void(int*, int*, uint64_t*, size_t, size_t, dim3, dim3)> copy_kernel_fun) {
  constexpr size_t BLOCK_SIZE = 256;
  constexpr size_t COPY_COUNT = 8;  // Each thread copies 8 elements
  constexpr size_t NUM_BLOCKS = 16384;
  constexpr size_t SIZE = BLOCK_SIZE * COPY_COUNT * NUM_BLOCKS;

  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim(NUM_BLOCKS);

  // Prepare host data
  std::vector<int> h_input(SIZE);
  std::vector<int> h_output(SIZE, 0);

  // Initialize input with sequential values
  for (size_t i = 0; i < SIZE; ++i) {
    h_input[i] = static_cast<int>(i);
  }

  // Allocate device memory
  int* d_input = nullptr;
  int* d_output = nullptr;
  uint64_t* d_clock = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_clock, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_output, 0, SIZE * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_clock, 0, sizeof(uint64_t)));

  KernelConfig config(grid_dim, block_dim);
  timeKernel(
      test_name,
      [&]() {
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel
        copy_kernel_fun(d_input, d_output, d_clock, COPY_COUNT, SIZE, grid_dim, block_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy output back to host
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, SIZE * sizeof(int), cudaMemcpyDeviceToHost));
      },
      &config);

  // Verify results: output should match input (sync_compute just copies data)
  for (size_t i = 0; i < SIZE; ++i) {
    EXPECT_EQ(h_output[i], h_input[i]) << fmt::format("Mismatch at index {}: expected {}, got {}", i, h_input[i],
                                                      h_output[i]);
  }

  // Verify clock was updated
  uint64_t h_clock = 0;
  CUDA_CHECK(cudaMemcpy(&h_clock, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Get GPU clock rate to convert cycles to seconds
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
  double seconds = static_cast<double>(h_clock) / (static_cast<double>(clock_rate_khz) * 1000.0);

  fmt::print("Clock cycles: {} ({:.6f} ms)\n", h_clock, seconds * 1000.0);
  EXPECT_GT(h_clock, 0) << "Clock should be non-zero after kernel execution";

  // Free device memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_clock));
}

TEST_F(CudaPoc_Day0501_Pipeline, SyncCopy) {
  test_copy_with_kernel_fun("SyncCopy", sync_copy<int>);
}

TEST_F(CudaPoc_Day0501_Pipeline, AsyncCopy) {
  test_copy_with_kernel_fun("AsyncCopy", async_copy<int>);
}

}  // namespace cuda_poc::day01
