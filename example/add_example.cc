#include <cuda_runtime.h>
#include <fmt/core.h>
#include <iostream>
#include <vector>

#include "add_cuda.cuh"
#include "cuda_utils.h"

using namespace cuda_poc;

const size_t SIZE = 1 << 20;  // 4MB
size_t size_bytes = SIZE * sizeof(float);

dim3 block_dim(256);
dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

void testCuda01() {
  std::vector<float> h_a(SIZE, 1);
  std::vector<float> h_b(SIZE, 2);
  std::vector<float> h_c(SIZE, 0);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, size_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, size_bytes));

  // Using device: NVIDIA T400 4GB
  // Compute capability: 7.5
  // Total memory: 4095 MB
  //
  // === Thread/Block Limits ===
  // Max threads per block: 1024
  // Max block dimensions: (1024, 1024, 64)
  // Max grid dimensions: (2147483647, 65535, 65535)
  // Warp size: 32
  // Max threads per multiprocessor: 1024
  // Number of multiprocessors: 6
  // Shared memory per block: 48 KB
  // Registers per block: 65536

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
        cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice);
        vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
        cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);
      },
      &config);

  timeKernel(
      "vector_add_one_by_one",
      [&]() {
        dim3 inner_block_dim(1);
        dim3 inner_grid_dim(1);
        // === Kernel: vector_add_one_by_one ===
        // Execution time: 100.568 ms
        // Grid dimensions: (4096, 1, 1)
        // Block dimensions: (256, 1, 1)
        // Total threads: 1048576
        // GPU memory used: 0 MB
        // GPU memory free: 3303.55 MB / 4095.69 MB
        cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice);
        vector_add(d_c, d_a, d_b, SIZE, inner_grid_dim, inner_block_dim);
        cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);
      },
      &config);

  // timeKernel("vector_add_max", [&]() {
  //     dim3 inner_block_dim(4096);
  //     dim3 inner_grid_dim(4096);
  //     // === Kernel: vector_add_max ===
  //     // Execution time: 2.39379 ms
  //     // Grid dimensions: (4096, 1, 1)
  //     // Block dimensions: (256, 1, 1)
  //     // Total threads: 1048576
  //     // GPU memory used: 0 MB
  //     // GPU memory free: 3303.55 MB / 4095.69 MB
  //     cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice);
  //     cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice);
  //     cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice);
  //     vector_add(d_c, d_a, d_b, SIZE, inner_grid_dim, inner_block_dim);
  //     cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);
  // }, &config);

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

int main() {
  // Check if CUDA is available
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    fmt::println(stderr, "CUDA initialization failed!");
    fmt::println(stderr, "Error code: {}", static_cast<int>(error));
    fmt::println(stderr, "Error: {}", cudaGetErrorString(error));

    if (error == cudaErrorInsufficientDriver) {
      fmt::println(stderr, "Driver is too old for CUDA 13.1. Please update NVIDIA drivers.");
    } else if (error == cudaErrorNoDevice) {
      fmt::println(stderr, "No CUDA-capable device found.");
    } else if (error == cudaErrorNotSupported) {
      fmt::println(stderr, "CUDA operation not supported. Possible driver/runtime mismatch.");
    }
    return 1;
  }

  if (deviceCount == 0) {
    fmt::println(stderr, "No CUDA-capable devices found!");
    return 1;
  }

  fmt::println("Found {} CUDA device(s)", deviceCount);

  // Get device properties
  cudaDeviceProp prop{};
  error = cudaGetDeviceProperties(&prop, 0);
  if (error != cudaSuccess) {
    fmt::println(stderr, "Failed to get device properties: {}", cudaGetErrorString(error));
    return 1;
  }

  fmt::println("Using device: {}", prop.name);
  fmt::println("Compute capability: {}.{}", prop.major, prop.minor);
  fmt::println("Total memory: {} MB", prop.totalGlobalMem / (1024 * 1024));
  fmt::println("\n=== Thread/Block Limits ===");
  fmt::println("Max threads per block: {}", prop.maxThreadsPerBlock);
  fmt::println("Max block dimensions: ({}, {}, {})", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
  fmt::println("Max grid dimensions: ({}, {}, {})", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  fmt::println("Warp size: {}", prop.warpSize);
  fmt::println("Max threads per multiprocessor: {}", prop.maxThreadsPerMultiProcessor);
  fmt::println("Number of multiprocessors: {}", prop.multiProcessorCount);
  fmt::println("Shared memory per block: {} KB", prop.sharedMemPerBlock / 1024);
  fmt::println("Registers per block: {}", prop.regsPerBlock);
  fmt::println("");

  testCuda01();

  return 0;
}
