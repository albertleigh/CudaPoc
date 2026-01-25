#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

#ifdef __linux__
#include <nccl.h>
#endif

namespace cuda_poc {
#define CUDA_CHECK(call)                                                                                          \
  do {                                                                                                            \
    cudaError_t error = call;                                                                                     \
    if (error != cudaSuccess) {                                                                                   \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << "\n"; \
      exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                             \
  } while (0)

#ifdef __linux__
#define NCCL_CHECK(call)                                                                                          \
  do {                                                                                                            \
    ncclResult_t error = call;                                                                                    \
    if (error != ncclSuccess) {                                                                                   \
      std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << " - " << ncclGetErrorString(error) << "\n"; \
      exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                             \
  } while (0)
#endif

// Structure to hold kernel launch configuration
struct KernelConfig {
  dim3 gridDim;
  dim3 blockDim;

  KernelConfig(dim3 grid, dim3 block) : gridDim(grid), blockDim(block) {}
};

// Enhanced CUDA kernel timing function with detailed info
inline float timeKernel(const std::string& name,
                        const std::function<void()>& kernel_func,
                        const KernelConfig* config = nullptr) {
  // Get memory info before kernel
  size_t freeBefore, totalBefore;
  CUDA_CHECK(cudaMemGetInfo(&freeBefore, &totalBefore));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  kernel_func();  // Execute the kernel
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Get memory info after kernel
  size_t freeAfter, totalAfter;
  CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalAfter));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  // Print detailed info
  std::cout << "\n=== Kernel: " << name << " ===\n";
  std::cout << "Execution time: " << milliseconds << " ms\n";

  if (config) {
    std::cout << "Grid dimensions: (" << config->gridDim.x << ", " << config->gridDim.y << ", " << config->gridDim.z
              << ")\n";
    std::cout << "Block dimensions: (" << config->blockDim.x << ", " << config->blockDim.y << ", " << config->blockDim.z
              << ")\n";

    size_t totalThreads = config->gridDim.x * config->gridDim.y * config->gridDim.z * config->blockDim.x *
                          config->blockDim.y * config->blockDim.z;
    std::cout << "Total threads: " << totalThreads << "\n";
  }

  std::cout << "GPU memory used: " << (freeBefore - freeAfter) / (1024.0 * 1024.0) << " MB\n";
  std::cout << "GPU memory free: " << freeAfter / (1024.0 * 1024.0) << " MB / " << totalAfter / (1024.0 * 1024.0)
            << " MB\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return milliseconds;
}

// Simple version without config info (backwards compatible)
inline float timeKernel(const std::string& name, std::function<void()> kernel_func) {
  return timeKernel(name, kernel_func, nullptr);
}
}  // namespace cuda_poc
