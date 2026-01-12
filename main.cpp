#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "cuda_utils.h"
#include "add_cuda.h"

const size_t SIZE = 1 << 20;
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

    cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice);

    launch_add_kernel(d_c, d_a, d_b, SIZE, grid_dim, block_dim);

    cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);

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
        std::cerr << "CUDA initialization failed!" << std::endl;
        std::cerr << "Error code: " << error << std::endl;
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        
        if (error == cudaErrorInsufficientDriver) {
            std::cerr << "Driver is too old for CUDA 13.1. Please update NVIDIA drivers." << std::endl;
        } else if (error == cudaErrorNoDevice) {
            std::cerr << "No CUDA-capable device found." << std::endl;
        } else if (error == cudaErrorNotSupported) {
            std::cerr << "CUDA operation not supported. Possible driver/runtime mismatch." << std::endl;
        }
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get device properties
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    testCuda01();
    
    return 0;
}
