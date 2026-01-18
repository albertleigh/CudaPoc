//
// Created by Albert Li on 1/17/2026.
//

#include <gtest/gtest.h>
#include "transpose_cuda.cuh"
#include "cuda_utils.h"
#include <fmt/format.h>
#include <vector>

namespace cuda_poc::day0302Transpose {
    using namespace cuda_poc;

    // W Bank conflicts
    // === Kernel: transpose_small ===
    // Execution time: 0.84304 ms
    // Grid dimensions: (4, 4, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 16384
    // GPU memory used: 2 MB
    // GPU memory free: 3298.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_non_square_blocks ===
    // Execution time: 1.59734 ms
    // Grid dimensions: (32, 32, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3292.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_double ===
    // Execution time: 1.04598 ms
    // Grid dimensions: (16, 16, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 262144
    // GPU memory used: 0 MB
    // GPU memory free: 3296.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_rectangular ===
    // Execution time: 0.97248 ms
    // Grid dimensions: (32, 16, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 524288
    // GPU memory used: 0 MB
    // GPU memory free: 3296.25 MB / 4095.56 MB

    // W/O Bank conflicts
    // === Kernel: transpose_4096 ===
    // Execution time: 32.7436 ms
    // Grid dimensions: (128, 128, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 16777216
    // GPU memory used: 30 MB
    // GPU memory free: 3144.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_4096_swizzling ===
    // Execution time: 23.6546 ms
    // Grid dimensions: (128, 128, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 16777216
    // GPU memory used: 0 MB
    // GPU memory free: 3144.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_non_square_blocks ===
    // Execution time: 1.6353 ms
    // Grid dimensions: (32, 32, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 1048576
    // GPU memory used: 0 MB
    // GPU memory free: 3264.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_double ===
    // Execution time: 1.01888 ms
    // Grid dimensions: (16, 16, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 262144
    // GPU memory used: 0 MB
    // GPU memory free: 3268.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_rectangular ===
    // Execution time: 0.971584 ms
    // Grid dimensions: (32, 16, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 524288
    // GPU memory used: 0 MB
    // GPU memory free: 3268.25 MB / 4095.56 MB
    //
    // === Kernel: transpose_rectangular_v2_swizzling ===
    // Execution time: 1.00518 ms
    // Grid dimensions: (32, 16, 1)
    // Block dimensions: (32, 32, 1)
    // Total threads: 524288
    // GPU memory used: 0 MB
    // GPU memory free: 3268.25 MB / 4095.56 MB


    class CudaPoc_Day0302 : public ::testing::Test {
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
            fmt::println("Compute capability: {}.{}", prop.major, prop.minor);
        }
    };

    void free_device_memory(float *d_input, float *d_output) {
        if (d_input) { CUDA_CHECK(cudaFree(d_input)); }
        if (d_output) { CUDA_CHECK(cudaFree(d_output)); }
    }

    // Helper function to verify transpose correctness
    bool verify_transpose(const std::vector<float> &input, const std::vector<float> &output, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float expected = input[i * cols + j];
                float actual = output[j * rows + i];
                if (std::abs(expected - actual) > 1e-5f) {
                    fmt::println("Mismatch at ({}, {}): expected {}, got {}", i, j, expected, actual);
                    return false;
                }
            }
        }
        return true;
    }

    TEST_F(CudaPoc_Day0302, Transpose) {
        constexpr int MATRIX_DIM = 4096;
        constexpr size_t SIZE = MATRIX_DIM * MATRIX_DIM;
        size_t size_bytes = SIZE * sizeof(float);

        // Use 32x32 blocks to match TRANSPOSE_BLOCK_DIM
        dim3 block_dim(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_dim((MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM,
                      (MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM);

        // Initialize input matrix with sequential values
        std::vector<float> h_input(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            h_input[i] = static_cast<float>(i);
        }
        std::vector<float> h_output(SIZE);

        float *d_input;
        float *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("transpose_4096", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            transpose<float>(d_output, d_input, MATRIX_DIM, MATRIX_DIM, grid_dim, block_dim);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_TRUE(verify_transpose(h_input, h_output, MATRIX_DIM, MATRIX_DIM));
        free_device_memory(d_input, d_output);
    }

    TEST_F(CudaPoc_Day0302, TransposeSwizzling) {
        constexpr int MATRIX_DIM = 4096;
        constexpr size_t SIZE = MATRIX_DIM * MATRIX_DIM;
        size_t size_bytes = SIZE * sizeof(float);

        // Use 32x32 blocks to match TRANSPOSE_BLOCK_DIM
        dim3 block_dim(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_dim((MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM,
                      (MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM);

        // Initialize input matrix with sequential values
        std::vector<float> h_input(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            h_input[i] = static_cast<float>(i);
        }
        std::vector<float> h_output(SIZE);

        float *d_input;
        float *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("transpose_4096_swizzling", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            transpose_v2<float>(d_output, d_input, MATRIX_DIM, MATRIX_DIM, grid_dim, block_dim);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_TRUE(verify_transpose(h_input, h_output, MATRIX_DIM, MATRIX_DIM));
        free_device_memory(d_input, d_output);
    }

    TEST_F(CudaPoc_Day0302, TransposeNonSquareBlocks) {
        // Test with matrix dimension not perfectly divisible by TRANSPOSE_BLOCK_DIM
        constexpr int MATRIX_DIM = 1000;
        constexpr size_t SIZE = MATRIX_DIM * MATRIX_DIM;
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_dim((MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM,
                      (MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM);

        std::vector<float> h_input(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            h_input[i] = static_cast<float>(i % 5000);
        }
        std::vector<float> h_output(SIZE);

        float *d_input;
        float *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("transpose_non_square_blocks", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            transpose<float>(d_output, d_input, MATRIX_DIM, MATRIX_DIM, grid_dim, block_dim);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_TRUE(verify_transpose(h_input, h_output, MATRIX_DIM, MATRIX_DIM));
        free_device_memory(d_input, d_output);
    }

    TEST_F(CudaPoc_Day0302, TransposeDouble) {
        // Test with double precision
        constexpr int MATRIX_DIM = 512;
        constexpr size_t SIZE = MATRIX_DIM * MATRIX_DIM;
        size_t size_bytes = SIZE * sizeof(double);

        dim3 block_dim(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_dim((MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM,
                      (MATRIX_DIM + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM);

        std::vector<double> h_input(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            h_input[i] = static_cast<double>(i) * 0.5;
        }
        std::vector<double> h_output(SIZE);

        double *d_input;
        double *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("transpose_double", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            transpose<double>(d_output, d_input, MATRIX_DIM, MATRIX_DIM, grid_dim, block_dim);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        }, &config);

        // Verify transpose for double
        bool correct = true;
        for (int i = 0; i < MATRIX_DIM && correct; ++i) {
            for (int j = 0; j < MATRIX_DIM; ++j) {
                double expected = h_input[i * MATRIX_DIM + j];
                double actual = h_output[j * MATRIX_DIM + i];
                if (std::abs(expected - actual) > 1e-10) {
                    correct = false;
                    break;
                }
            }
        }
        EXPECT_TRUE(correct);

        if (d_input) { CUDA_CHECK(cudaFree(d_input)); }
        if (d_output) { CUDA_CHECK(cudaFree(d_output)); }
    }

    TEST_F(CudaPoc_Day0302, TransposeRectangular) {
        // Test with rectangular matrix (non-square)
        constexpr int ROWS = 512;
        constexpr int COLS = 1024;
        constexpr size_t SIZE = ROWS * COLS;
        size_t size_bytes = SIZE * sizeof(float);

        // Grid dimensions need to cover both dimensions
        dim3 block_dim(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_dim((COLS + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM, // x covers columns
                      (ROWS + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM); // y covers rows

        // Initialize input matrix (ROWS x COLS)
        std::vector<float> h_input(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            h_input[i] = static_cast<float>(i);
        }
        std::vector<float> h_output(SIZE); // Output will be COLS x ROWS

        float *d_input;
        float *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("transpose_rectangular", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            transpose<float>(d_output, d_input, ROWS, COLS, grid_dim, block_dim);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_TRUE(verify_transpose(h_input, h_output, ROWS, COLS));
        free_device_memory(d_input, d_output);
    }

    TEST_F(CudaPoc_Day0302, TransposeRectangularV2Swizzling) {
        constexpr int ROWS = 512;
        constexpr int COLS = 1024;
        constexpr size_t SIZE = ROWS * COLS;
        size_t size_bytes = SIZE * sizeof(float);

        dim3 block_dim(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_dim((COLS + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM,
                      (ROWS + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM);

        std::vector<float> h_input(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            h_input[i] = static_cast<float>(i);
        }
        std::vector<float> h_output(SIZE);

        float *d_input;
        float *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));

        KernelConfig config(grid_dim, block_dim);
        timeKernel("transpose_rectangular_v2_swizzling", [&]() {
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
            transpose_v2<float>(d_output, d_input, ROWS, COLS, grid_dim, block_dim);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        }, &config);

        EXPECT_TRUE(verify_transpose(h_input, h_output, ROWS, COLS));
        free_device_memory(d_input, d_output);
    }
} // namespace cuda_poc::day0302Transpose
