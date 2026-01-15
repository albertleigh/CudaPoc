#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <functional>
#include <string>
#include <fmt/core.h>

namespace cuda_poc {
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fmt::println(stderr, "CUDA error at {}:{} - {}", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

    // Structure to hold kernel launch configuration
    struct KernelConfig {
        dim3 gridDim;
        dim3 blockDim;

        KernelConfig(dim3 grid, dim3 block)
            : gridDim(grid), blockDim(block) {
        }
    };

    // Enhanced CUDA kernel timing function with detailed info
    inline float timeKernel(const std::string &name, const std::function<void()> &kernel_func,
                            const KernelConfig *config = nullptr) {
        // Get memory info before kernel
        size_t freeBefore, totalBefore;
        CUDA_CHECK(cudaMemGetInfo(&freeBefore, &totalBefore));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        kernel_func(); // Execute the kernel
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Get memory info after kernel
        size_t freeAfter, totalAfter;
        CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalAfter));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Print detailed info
        fmt::println("\n=== Kernel: {} ===", name);
        fmt::println("Execution time: {} ms", milliseconds);

        if (config) {
            fmt::println("Grid dimensions: ({}, {}, {})",
                         config->gridDim.x, config->gridDim.y, config->gridDim.z);
            fmt::println("Block dimensions: ({}, {}, {})",
                         config->blockDim.x, config->blockDim.y, config->blockDim.z);

            size_t totalThreads = config->gridDim.x * config->gridDim.y * config->gridDim.z *
                                  config->blockDim.x * config->blockDim.y * config->blockDim.z;
            fmt::println("Total threads: {}", totalThreads);
        }

        fmt::println("GPU memory used: {} MB", (freeBefore - freeAfter) / (1024.0 * 1024.0));
        fmt::println("GPU memory free: {} MB / {} MB",
                     freeAfter / (1024.0 * 1024.0),
                     totalAfter / (1024.0 * 1024.0));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return milliseconds;
    }

    // Simple version without config info (backwards compatible)
    inline float timeKernel(const std::string &name, std::function<void()> kernel_func) {
        return timeKernel(name, kernel_func, nullptr);
    }
} // namespace cuda_poc
