//
// Created by Albert Li on 1/14/2026.
//
#include <gtest/gtest.h>
#include "add_cuda.cuh"
#include "cuda_utils.h"
#include "cuda_fp16.h"

namespace cuda_poc::day01 {
    using namespace cuda_poc;

    template<typename T>
    void assert_vector_equal(const std::vector<T> &target, const size_t size_bytes, const T expected) {
        if constexpr (std::is_same_v<T, float2>) {
            for (size_t i = 0; i < size_bytes; ++i) {
                const T &ref = target[i];
                ASSERT_EQ(ref.x, expected.x) << "Verification failed at index x" << i;
                ASSERT_EQ(ref.y, expected.y) << "Verification failed at index y" << i;
            }
        } else if constexpr (std::is_same_v<T, float3>) {
            for (size_t i = 0; i < size_bytes; ++i) {
                const T &ref = target[i];
                ASSERT_EQ(ref.x, expected.x) << "Verification failed at index x" << i;
                ASSERT_EQ(ref.y, expected.y) << "Verification failed at index y" << i;
                ASSERT_EQ(ref.z, expected.z) << "Verification failed at index z" << i;
            }
        } else if constexpr (std::is_same_v<T, float4>) {
            for (size_t i = 0; i < size_bytes; ++i) {
                const T &ref = target[i];
                ASSERT_EQ(ref.x, expected.x) << "Verification failed at index x" << i;
                ASSERT_EQ(ref.y, expected.y) << "Verification failed at index y" << i;
                ASSERT_EQ(ref.z, expected.z) << "Verification failed at index z" << i;
                ASSERT_EQ(ref.w, expected.w) << "Verification failed at index w" << i;
            }
        } else {
            for (size_t i = 0; i < size_bytes; ++i) {
                ASSERT_EQ(target[i], expected) << "Verification failed at index " << i;
            }
        }
    }

    template<typename T>
    void free_device_ptr(T *d_a, T *d_b, T *d_c) {
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

    // === Kernel: vector_add ===
    // Execution time: 3.329024 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3290.25 MB / 4095.5625 MB
    // Size used: 1048576
    //
    // === Kernel: vector_add_in_float2 ===
    // Execution time: 3.239808 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3290.25 MB / 4095.5625 MB
    // Size used: 524288
    //
    // === Kernel: vector_add_in_float3 ===
    // Execution time: 3.267872 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3284.25 MB / 4095.5625 MB
    // Size used: 349526
    //
    // === Kernel: vector_add_in_float4 ===
    // Execution time: 3.292256 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3290.25 MB / 4095.5625 MB
    // Size used: 262144
    //
    // === Kernel: vector_add_in_half ===
    // Execution time: 1.767296 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3296.25 MB / 4095.5625 MB
    // Size used: 1048576
    //
    // === Kernel: vector_add_in_half2 ===
    // Execution time: 1.799968 ms
    // Grid dimensions: (4096, 1, 1)
    // Block dimensions: (256, 1, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3296.25 MB / 4095.5625 MB
    // Size used: 524288
    TEST(CudaPoc_Day0103, CompareInTypes) {
        constexpr size_t SIZE = 1 << 20; // 4MB
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(256);
        dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

        // float
        {
            std::vector<float> h_a(SIZE, 1);
            std::vector<float> h_b(SIZE, 2);
            std::vector<float> h_c(SIZE, 0);

            float *d_a, *d_b, *d_c;
            CUDA_CHECK(cudaMalloc(&d_a, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
            CUDA_CHECK(cudaMalloc(&d_c, size_bytes));

            KernelConfig config(grid_dim, block_dim);
            timeKernel("vector_add", [&]() {
                CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));
                vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
            }, &config);
            fmt::println("Size used: {}", SIZE);

            assert_vector_equal(h_c, SIZE, 3.0f);
            free_device_ptr(d_a, d_b, d_c);
        }

        // float2
        {
            constexpr size_t SIZE_USED = SIZE / 2;
            size_t size_bytes_used = SIZE_USED * sizeof(float2);
            std::vector<float2> h_a(SIZE_USED, float2(1, 1));
            std::vector<float2> h_b(SIZE_USED, float2(2, 2));
            std::vector<float2> h_c(SIZE_USED, float2(0, 0));

            float2 *d_a, *d_b, *d_c;
            CUDA_CHECK(cudaMalloc(&d_a, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_b, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_c, size_bytes_used));

            KernelConfig config(grid_dim, block_dim);
            timeKernel("vector_add_in_float2", [&]() {
                CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes_used, cudaMemcpyHostToDevice));
                vector_add(d_c, d_a, d_b, SIZE_USED, grid_dim, block_dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes_used, cudaMemcpyDeviceToHost));
            }, &config);
            fmt::println("Size used: {}", SIZE_USED);

            assert_vector_equal(h_c, SIZE_USED, float2(3, 3));
            free_device_ptr(d_a, d_b, d_c);
        }

        // floa3
        {
            constexpr size_t SIZE_USED = (SIZE + 2) / 3;
            size_t size_bytes_used = SIZE_USED * sizeof(float3);
            std::vector<float3> h_a(SIZE_USED, float3(1, 1, 1));
            std::vector<float3> h_b(SIZE_USED, float3(2, 2, 2));
            std::vector<float3> h_c(SIZE_USED, float3(0, 0, 0));

            float3 *d_a, *d_b, *d_c;
            CUDA_CHECK(cudaMalloc(&d_a, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_b, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_c, size_bytes_used));

            KernelConfig config(grid_dim, block_dim);
            timeKernel("vector_add_in_float3", [&]() {
                CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes_used, cudaMemcpyHostToDevice));
                vector_add(d_c, d_a, d_b, SIZE_USED, grid_dim, block_dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes_used, cudaMemcpyDeviceToHost));
            }, &config);
            fmt::println("Size used: {}", SIZE_USED);

            assert_vector_equal(h_c, SIZE_USED, float3(3, 3, 3));
            free_device_ptr(d_a, d_b, d_c);
        }

        // floa4
        {
            constexpr size_t SIZE_USED = SIZE / 4;
            size_t size_bytes_used = SIZE_USED * sizeof(float4);
            std::vector<float4> h_a(SIZE_USED, float4(1, 1, 1, 1));
            std::vector<float4> h_b(SIZE_USED, float4(2, 2, 2, 2));
            std::vector<float4> h_c(SIZE_USED, float4(0, 0, 0, 0));

            float4 *d_a, *d_b, *d_c;
            CUDA_CHECK(cudaMalloc(&d_a, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_b, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_c, size_bytes_used));

            KernelConfig config(grid_dim, block_dim);
            timeKernel("vector_add_in_float4", [&]() {
                CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes_used, cudaMemcpyHostToDevice));
                vector_add(d_c, d_a, d_b, SIZE_USED, grid_dim, block_dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes_used, cudaMemcpyDeviceToHost));
            }, &config);
            fmt::println("Size used: {}", SIZE_USED);

            assert_vector_equal(h_c, SIZE_USED, float4(3, 3, 3, 3));
            free_device_ptr(d_a, d_b, d_c);
        }

        // half
        {
            constexpr size_t SIZE_USED = SIZE;
            size_t size_bytes_used = SIZE_USED * sizeof(half);

            std::vector<half> h_a(SIZE_USED, 1);
            std::vector<half> h_b(SIZE_USED, 2);
            std::vector<half> h_c(SIZE_USED, 0);

            half *d_a, *d_b, *d_c;
            CUDA_CHECK(cudaMalloc(&d_a, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_b, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_c, size_bytes_used));

            KernelConfig config(grid_dim, block_dim);
            timeKernel("vector_add_in_half", [&]() {
                CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes_used, cudaMemcpyHostToDevice));
                vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes_used, cudaMemcpyDeviceToHost));
            }, &config);
            fmt::println("Size used: {}", SIZE_USED);

            assert_vector_equal(h_c, SIZE_USED, half(3.0f));
            free_device_ptr(d_a, d_b, d_c);
        }

        // half2
        {
            constexpr size_t SIZE_USED = SIZE / 2;
            size_t size_bytes_used = SIZE_USED * sizeof(half2);
            std::vector<half2> h_a(SIZE_USED, half2(1, 1));
            std::vector<half2> h_b(SIZE_USED, half2(2, 2));
            std::vector<half2> h_c(SIZE_USED, half2(0, 0));

            half2 *d_a, *d_b, *d_c;
            CUDA_CHECK(cudaMalloc(&d_a, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_b, size_bytes_used));
            CUDA_CHECK(cudaMalloc(&d_c, size_bytes_used));

            KernelConfig config(grid_dim, block_dim);
            timeKernel("vector_add_in_half2", [&]() {
                CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes_used, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes_used, cudaMemcpyHostToDevice));
                vector_add(d_c, d_a, d_b, SIZE_USED, grid_dim, block_dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes_used, cudaMemcpyDeviceToHost));
            }, &config);
            fmt::println("Size used: {}", SIZE_USED);

            assert_vector_equal(h_c, SIZE_USED, half2(3, 3));
            free_device_ptr(d_a, d_b, d_c);
        }
    }
} //namespace cuda_poc::day01
