//
// Created by Albert Li on 1/15/2026.
//

#include <gtest/gtest.h>
#include "sum_cuda.cuh"
#include "cuda_utils.h"
#include <fmt/format.h>


namespace cuda_poc::day01 {
    using namespace cuda_poc;

    // === Kernel: sum_reduction_v1 ===
    // Execution time: 3.63133 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 2 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v2 ===
    // Execution time: 1.29712 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v3 ===
    // Execution time: 1.18957 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v4 ===
    // Execution time: 1.91325 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v5 ===
    // Execution time: 1.05277 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v6 ===
    // Execution time: 1.01827 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v7 ===
    // Execution time: 1.03414 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v8 ===
    // Execution time: 1.09549 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB
    //
    // === Kernel: sum_reduction_v9 ===
    // Execution time: 0.994528 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3294.25 MB / 4095.56 MB

    class CudaPoc_Day0301 : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            // Check if CUDA is available
            int deviceCount = 0;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);
            if (error != cudaSuccess) {
                FAIL() << fmt::format("Failed to get device count: {}", cudaGetErrorString(error));
            }

            fmt::println("Found {} CUDA device(s)", deviceCount);

            // Get device properties
            cudaDeviceProp prop{};
            error = cudaGetDeviceProperties(&prop, 0);
            if (error != cudaSuccess) {
                FAIL() << fmt::format("Failed to get device properties: {}", cudaGetErrorString(error));
            }

            CudaPoc_Day0301::wrap_size = prop.warpSize;
            fmt::println("Set static wrap size to {}", wrap_size);
        }

        static unsigned int wrap_size;
    };

    // initialize the static value of CudaPoc_Day0301
    unsigned int CudaPoc_Day0301::wrap_size = 0;

    void free_device_memory(float *d_input, float *d_result) {
        if (d_input) { CUDA_CHECK(cudaFree(d_input)); }
        if (d_result) { CUDA_CHECK(cudaFree(d_result)); }
    }

    TEST_F(CudaPoc_Day0301, SumV1) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v1", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v1(d_result, d_input, SIZE, grid_dim, block_dim);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

    TEST_F(CudaPoc_Day0301, SumV2) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v2", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v2(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

    TEST_F(CudaPoc_Day0301, SumV3) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v3", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v3(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

    TEST_F(CudaPoc_Day0301, SumV4) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v4", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v4(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

    TEST_F(CudaPoc_Day0301, SumV5) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v5", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v5(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

    TEST_F(CudaPoc_Day0301, SumV6) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v6", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v6(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

    TEST_F(CudaPoc_Day0301, SumV7) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v7", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v7(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

    TEST_F(CudaPoc_Day0301, SumV8) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v8", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v8(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)
    TEST_F(CudaPoc_Day0301, SumV9) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        std::vector<float> h_input(SIZE, 1);
        float result = 0.0f;
        float *h_result = &result;

        float *d_input;
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("sum_reduction_v9", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            vector_sum_v9(d_result, d_input, SIZE, grid_dim, block_dim, wrap_size);
            CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_EQ(*h_result, SIZE);
        free_device_memory(d_input, d_result);
    }
#endif
} // namespace cuda_poc::day01
