#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <functional>
#include <string>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Structure to hold kernel launch configuration
struct KernelConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMemBytes = 0;
    std::string kernelName;
    
    KernelConfig(dim3 grid, dim3 block, const std::string& name = "", size_t sharedMem = 0)
        : gridDim(grid), blockDim(block), sharedMemBytes(sharedMem), kernelName(name) {}
};

// Enhanced CUDA kernel timing function with detailed info
inline float timeKernel(const std::string& name, std::function<void()> kernel_func, 
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
    std::cout << "\n=== Kernel: " << name << " ===" << std::endl;
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    
    if (config) {
        std::cout << "Grid dimensions: (" << config->gridDim.x << ", " 
                  << config->gridDim.y << ", " << config->gridDim.z << ")" << std::endl;
        std::cout << "Block dimensions: (" << config->blockDim.x << ", " 
                  << config->blockDim.y << ", " << config->blockDim.z << ")" << std::endl;
        
        size_t totalThreads = config->gridDim.x * config->gridDim.y * config->gridDim.z *
                              config->blockDim.x * config->blockDim.y * config->blockDim.z;
        std::cout << "Total threads: " << totalThreads << std::endl;
        
        if (config->sharedMemBytes > 0) {
            std::cout << "Shared memory: " << config->sharedMemBytes << " bytes" << std::endl;
        }
    }
    
    std::cout << "GPU memory used: " << (freeBefore - freeAfter) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "GPU memory free: " << freeAfter / (1024.0 * 1024.0) << " MB / " 
              << totalAfter / (1024.0 * 1024.0) << " MB" << std::endl;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

// Simple version without config info (backwards compatible)
inline float timeKernel(const std::string& name, std::function<void()> kernel_func) {
    return timeKernel(name, kernel_func, nullptr);
}
