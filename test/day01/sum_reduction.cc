//
// Created by Albert Li on 1/15/2026.
//

#include <gtest/gtest.h>
#include "sum_cuda.cuh"
#include "cuda_utils.h"


namespace cuda_poc::day01 {

    using namespace cuda_poc;

    TEST(CudaPoc_Day0103, SumV1) {
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
    }
} // namespace cuda_poc::day01
