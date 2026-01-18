//
// Created by Albert Li on 1/18/2026.
//
// === Kernel: gemm_v1_naive (M=4096, N=2048, K=256) ===
// Execution time: 115.988 ms
// Grid dimensions: (128, 64, 1)
// Block dimensions: (32, 32, 1)
// Total threads: 8388608
// GPU memory used: 30 MB
// GPU memory free: 3234.25 MB / 4095.56 MB
//
// === Kernel: gemm_v2_smem (M=4096, N=2048, K=256) ===
// Execution time: 26.6019 ms
// Grid dimensions: (128, 64, 1)
// Block dimensions: (32, 32, 1)
// Total threads: 8388608
// GPU memory used: 0 MB
// GPU memory free: 3234.25 MB / 4095.56 MB
//
// === Kernel: gemm_v3_1d_block_tiling (M=4096, N=2048, K=256) ===
// Execution time: 21.5184 ms
// Grid dimensions: (128, 64, 1)
// Block dimensions: (32, 32, 1)
// Total threads: 8388608
// GPU memory used: 0 MB
// GPU memory free: 3234.25 MB / 4095.56 MB
//
// === Kernel: gemm_v4_2d_block_tiling (M=4096, N=2048, K=256) ===
// Execution time: 15.8737 ms
// Grid dimensions: (128, 64, 1)
// Block dimensions: (32, 32, 1)
// Total threads: 8388608
// GPU memory used: 0 MB
// GPU memory free: 3234.25 MB / 4095.56 MB
//
// === Kernel: gemm_v5_warptiling (M=4096, N=2048, K=256) ===
// Execution time: 16.5212 ms
// Grid dimensions: (128, 64, 1)
// Block dimensions: (32, 32, 1)
// Total threads: 8388608
// GPU memory used: 0 MB
// GPU memory free: 3234.25 MB / 4095.56 MB
//
// === Kernel: gemm_v6_cublasSgemm (M=4096, N=2048, K=256) ===
// Execution time: 55.4402 ms
// Grid dimensions: (128, 64, 1)
// Block dimensions: (32, 32, 1)
// Total threads: 8388608
// GPU memory used: 0 MB
// GPU memory free: 3224.25 MB / 4095.56 MB

#include <gtest/gtest.h>
#include "gemm_cuda.cuh"
#include "cuda_utils.h"
#include <fmt/format.h>
#include <vector>
#include <functional>
#include <cublas_v2.h>

namespace cuda_poc::day01 {
    using namespace cuda_poc::linear;

    class CudaPoc_Day0401 : public ::testing::Test {
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

            fmt::println("Device: {}", prop.name);
        }
    };

    void free_device_memory(float *d_A, float *d_B, float *d_C) {
        if (d_A) { CUDA_CHECK(cudaFree(d_A)); }
        if (d_B) { CUDA_CHECK(cudaFree(d_B)); }
        if (d_C) { CUDA_CHECK(cudaFree(d_C)); }
    }

    void test_gemm_with_kernel_fun(const char *test_name, int M, int N, int K,
                                   std::function<
                                       void(int, int, int, float, const float *, const float *, float, float *)
                                   > gemm_kernel_func) {
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);

        // Initialize host matrices with small values for verification
        std::vector<float> h_A(M * K, 0.01f);
        std::vector<float> h_B(K * N, 0.01f);
        std::vector<float> h_C(M * N, 0.0f);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));

        dim3 block(32, 32);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        KernelConfig config(grid, block);

        timeKernel(test_name, [&]() {
            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

            gemm_kernel_func(M, N, K, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        }, &config);

        // Verify: each element should be approximately K * 0.01 * 0.01 = K * 0.0001
        float expected = K * 0.01f * 0.01f;
        for (int i = 0; i < std::min(100, M * N); ++i) {
            EXPECT_NEAR(h_C[i], expected, 1e-3) << "Mismatch at index " << i;
        }

        free_device_memory(d_A, d_B, d_C);
    }

    TEST_F(CudaPoc_Day0401, GemmV1_4096x2048x256) {
        test_gemm_with_kernel_fun("gemm_v1_naive (M=4096, N=2048, K=256)", 4096, 2048, 256, linear_v1<float>);
    }

    TEST_F(CudaPoc_Day0401, GemmV2_4096x2048x256) {
        // Stall MIO Throttle: Warp was stalled waiting for MIO (memory input/output).
        // on average, each warp of this workload spends 26.2 cycles being stalled waiting for the MIO
        // (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme
        // utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as
        // shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads
        // can reduce pipeline pressure. This stall type represents about 68.4% of the total average of 38.3 cycles
        // between issuing two instructions.
        test_gemm_with_kernel_fun("gemm_v2_smem (M=4096, N=2048, K=256)", 4096, 2048, 256, linear_v2<float>);
    }

    TEST_F(CudaPoc_Day0401, GemmV3_4096x2048x256) {
        // One or more L2 Slices have a much higher number of active cycles than the average number of active cycles.
        // Maximum instance value is 20.04% above the average, while the minimum instance value is 3.69% below the
        // average.
        test_gemm_with_kernel_fun("gemm_v3_1d_block_tiling (M=4096, N=2048, K=256)", 4096, 2048, 256, linear_v3<float>);
    }

    TEST_F(CudaPoc_Day0401, GemmV4_4096x2048x256) {
        test_gemm_with_kernel_fun("gemm_v4_2d_block_tiling (M=4096, N=2048, K=256)", 4096, 2048, 256, linear_v4<float>);
    }

    TEST_F(CudaPoc_Day0401, GemmV5_4096x2048x256) {
        test_gemm_with_kernel_fun("gemm_v5_warptiling (M=4096, N=2048, K=256)", 4096, 2048, 256, linear_v5<float>);
    }

    TEST_F(CudaPoc_Day0401, GemmV6_4096x2048x256) {
        // Create cuBLAS handle once for this test
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        // Use std::bind to bind the handle as the first parameter
        auto gemm_func = std::bind(linear_v6<float>, handle,
                                   std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                   std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                   std::placeholders::_7, std::placeholders::_8);
        
        test_gemm_with_kernel_fun("gemm_v6_cublasSgemm (M=4096, N=2048, K=256)", 4096, 2048, 256, gemm_func);
        
        cublasDestroy(handle);
    }
} // namespace cuda_poc::day01
