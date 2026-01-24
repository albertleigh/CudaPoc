//
// Created by ali on 1/24/26.
//

#include <gtest/gtest.h>
#include "add_cuda.cuh"
#include "cuda_utils.h"
#include "mul_cuda.cuh"

namespace cuda_poc::day05cu {
class CudaPoc_Day0502_Stream : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
      FAIL() << "Failed to get device count: " << cudaGetErrorString(error);
    }
  }

  template <typename T>
  void assert_vector_equal(const std::vector<T>& target, const size_t size_bytes, const T expected) {
    if constexpr (std::is_same_v<T, float2>) {
      for (size_t i = 0; i < size_bytes; ++i) {
        const T& ref = target[i];
        ASSERT_EQ(ref.x, expected.x) << "Verification failed at index x" << i;
        ASSERT_EQ(ref.y, expected.y) << "Verification failed at index y" << i;
      }
    } else if constexpr (std::is_same_v<T, float3>) {
      for (size_t i = 0; i < size_bytes; ++i) {
        const T& ref = target[i];
        ASSERT_EQ(ref.x, expected.x) << "Verification failed at index x" << i;
        ASSERT_EQ(ref.y, expected.y) << "Verification failed at index y" << i;
        ASSERT_EQ(ref.z, expected.z) << "Verification failed at index z" << i;
      }
    } else if constexpr (std::is_same_v<T, float4>) {
      for (size_t i = 0; i < size_bytes; ++i) {
        const T& ref = target[i];
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

  template <typename... Args>
  void free_device_ptr(Args*... ptrs) {
    (([&]() {
       if (ptrs) {
         CUDA_CHECK(cudaFree(ptrs));
       }
     })(),
     ...);
  }

  void test_streams_with_kernel_fun(
      std::function<
          void(dim3& grid, dim3& block, float* d_a, float* d_b, float* d_c, float* d_d, size_t SIZE, size_t step)>
          stream_kernel_func) {
    constexpr size_t SIZE = 1 << 20;  // 4MB
    size_t size_bytes = SIZE * sizeof(float);

    dim3 block_dim(256);
    dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);
    size_t step = block_dim.x * grid_dim.x;

    // float
    {
      std::vector<float> h_a(SIZE, 1);
      std::vector<float> h_b(SIZE, 2);
      std::vector<float> h_c(SIZE, 0);
      std::vector<float> h_d(SIZE, 0);

      float *d_a, *d_b, *d_c, *d_d;
      CUDA_CHECK(cudaMalloc(&d_a, size_bytes));
      CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
      CUDA_CHECK(cudaMalloc(&d_c, size_bytes));
      CUDA_CHECK(cudaMalloc(&d_d, size_bytes));

      cudaStream_t stream1, stream2;
      CUDA_CHECK(cudaStreamCreate(&stream1));
      CUDA_CHECK(cudaStreamCreate(&stream2));

      CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_d, h_d.data(), size_bytes, cudaMemcpyHostToDevice));

      stream_kernel_func(grid_dim, block_dim, d_a, d_b, d_c, d_d, SIZE, step);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_d.data(), d_d, size_bytes, cudaMemcpyDeviceToHost));

      assert_vector_equal(h_c, SIZE, 3.0f);
      assert_vector_equal(h_d, SIZE, 2.0f);
      free_device_ptr(d_a, d_b, d_c, d_d);
    }
  }
};

TEST_F(CudaPoc_Day0502_Stream, Stream01) {
  test_streams_with_kernel_fun(
      [](dim3& grid_dim, dim3& block_dim, float* d_a, float* d_b, float* d_c, float* d_d, size_t SIZE, size_t step) {
        cudaStream_t stream1, stream2;
        CUDA_CHECK(cudaStreamCreate(&stream1));
        CUDA_CHECK(cudaStreamCreate(&stream2));

        vector_add_kernel<<<grid_dim, block_dim, 0, stream1>>>(d_c, d_a, d_b, SIZE, step);
        vector_mul_kernel<<<grid_dim, block_dim, 0, stream2>>>(d_d, d_a, d_b, SIZE, step);
      });
}

TEST_F(CudaPoc_Day0502_Stream, Stream02) {
  test_streams_with_kernel_fun(
      [](dim3& grid_dim, dim3& block_dim, float* d_a, float* d_b, float* d_c, float* d_d, size_t SIZE, size_t step) {
        cudaStream_t stream2;
        // CUDA_CHECK(cudaStreamCreate(&stream1));
        CUDA_CHECK(cudaStreamCreate(&stream2));

        vector_add_kernel<<<grid_dim, block_dim>>>(d_c, d_a, d_b, SIZE, step);
        vector_mul_kernel<<<grid_dim, block_dim, 0, stream2>>>(d_d, d_a, d_b, SIZE, step);
      });
}

TEST_F(CudaPoc_Day0502_Stream, Stream03) {
  test_streams_with_kernel_fun(
      [](dim3& grid_dim, dim3& block_dim, float* d_a, float* d_b, float* d_c, float* d_d, size_t SIZE, size_t step) {
        cudaStream_t stream2;
        // CUDA_CHECK(cudaStreamCreate(&stream1));
        // CUDA_CHECK(cudaStreamCreate(&stream2));

        vector_add_kernel<<<grid_dim, block_dim>>>(d_c, d_a, d_b, SIZE, step);
        vector_mul_kernel<<<grid_dim, block_dim>>>(d_d, d_a, d_b, SIZE, step);
      });
}

TEST_F(CudaPoc_Day0502_Stream, Stream04) {
  test_streams_with_kernel_fun(
      [](dim3& grid_dim, dim3& block_dim, float* d_a, float* d_b, float* d_c, float* d_d, size_t SIZE, size_t step) {
        cudaStream_t stream2;
        // CUDA_CHECK(cudaStreamCreate(&stream1));
        // Operations in this stream will not implicitly synchronize with the null stream
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

        vector_add_kernel<<<grid_dim, block_dim>>>(d_c, d_a, d_b, SIZE, step);
        vector_mul_kernel<<<grid_dim, block_dim, 0, stream2>>>(d_d, d_a, d_b, SIZE, step);
      });
}

}  // namespace cuda_poc::day05cu