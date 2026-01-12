#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "cuda_utils.h"
#include "add_cuda.h"

const size_t SIZE = 1 << 20; // 4MB
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

    KernelConfig config(grid_dim, block_dim, "vector_add_kernel");
    timeKernel("vector_add", [&]() {
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
    }, &config);

    timeKernel("vector_add_one_by_one", [&]() {
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
    }, &config);

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
        std::cerr << "CUDA initialization failed!" << '\n';
        std::cerr << "Error code: " << error << '\n';
        std::cerr << "Error: " << cudaGetErrorString(error) << '\n';

        if (error == cudaErrorInsufficientDriver) {
            std::cerr << "Driver is too old for CUDA 13.1. Please update NVIDIA drivers." << '\n';
        } else if (error == cudaErrorNoDevice) {
            std::cerr << "No CUDA-capable device found." << '\n';
        } else if (error == cudaErrorNotSupported) {
            std::cerr << "CUDA operation not supported. Possible driver/runtime mismatch." << '\n';
        }
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << '\n';
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << '\n';

    // Get device properties
    cudaDeviceProp prop{};
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << '\n';
        return 1;
    }

    std::cout << "Using device: " << prop.name << '\n';
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << '\n';
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << '\n';
    std::cout << "\n=== Thread/Block Limits ===" << '\n';
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << '\n';
    std::cout << "Max block dimensions: (" << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << '\n';
    std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << '\n';
    std::cout << "Warp size: " << prop.warpSize << '\n';
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << '\n';
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << '\n';
    std::cout << "Registers per block: " << prop.regsPerBlock << '\n';
    std::cout << '\n';

    testCuda01();

    return 0;
}
