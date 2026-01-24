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

  template <typename... Args>
  void free_host_ptr(Args*... ptrs) {
    (([&]() {
       if (ptrs) {
         CUDA_CHECK(cudaFreeHost(ptrs));
       }
     })(),
     ...);
  }

  template <typename... Args>
  void free_async_device_ptr(cudaStream_t stream, Args*... ptrs) {
    (([&]() {
       if (ptrs) {
         CUDA_CHECK(cudaFreeAsync(ptrs, stream));
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
        // Wait for all async operations in stream1 to complete before asserting
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
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
        // Wait for all async operations in stream1 to complete before asserting
        CUDA_CHECK(cudaStreamSynchronize(stream2));
      });
}

TEST_F(CudaPoc_Day0502_Stream, Stream03) {
  test_streams_with_kernel_fun(
      [](dim3& grid_dim, dim3& block_dim, float* d_a, float* d_b, float* d_c, float* d_d, size_t SIZE, size_t step) {
        // cudaStream_t stream2;
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
        // Wait for all async operations in stream1 to complete before asserting
        CUDA_CHECK(cudaStreamSynchronize(stream2));
      });
}

// === Kernel: vector_add_wo_stream ===
// Execution time: 3.48784 ms
// Grid dimensions: (4096, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 1048576
// GPU memory used: 0 MB
// GPU memory free: 3260.25 MB / 4095.56 MB
// Size used: 1048576
//
// === Kernel: vector_add_in_stream ===
// Execution time: 3.14851 ms
// Grid dimensions: (4096, 1, 1)
// Block dimensions: (256, 1, 1)
// Total threads: 1048576
// GPU memory used: 0 MB
// GPU memory free: 3240.25 MB / 4095.56 MB
// Size used: 1048576

TEST_F(CudaPoc_Day0502_Stream, AllocateInStream01) {
  constexpr size_t SIZE = 1 << 20;  // 4MB for float
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
    timeKernel(
        "vector_add_wo_stream",
        [&]() {
          CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));
          vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
        },
        &config);
    std::cout << "Size used: " << SIZE << '\n';

    assert_vector_equal(h_c, SIZE, 3.0f);
    free_device_ptr(d_a, d_b, d_c);
  }
}

TEST_F(CudaPoc_Day0502_Stream, AllocateInStream02) {
  constexpr size_t SIZE = 1 << 20;  // 4MB
  size_t size_bytes = SIZE * sizeof(float);

  dim3 block_dim(256);
  dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

  // float
  {
    std::vector<float> h_a(SIZE, 1);
    std::vector<float> h_b(SIZE, 2);
    std::vector<float> h_c(SIZE, 0);

    cudaStream_t stream1;
    // CUDA_CHECK(cudaStreamCreate(&stream1));
    // Operations in this stream will not implicitly synchronize with the null stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMallocAsync(&d_a, size_bytes, stream1));
    CUDA_CHECK(cudaMallocAsync(&d_b, size_bytes, stream1));
    CUDA_CHECK(cudaMallocAsync(&d_c, size_bytes, stream1));

    KernelConfig config(grid_dim, block_dim);
    timeKernel(
        "vector_add_in_stream",
        [&]() {
          size_t step = block_dim.x * grid_dim.x;
          CUDA_CHECK(cudaMemcpyAsync(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice, stream1));
          CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice, stream1));
          CUDA_CHECK(cudaMemcpyAsync(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice, stream1));
          vector_add_kernel<<<grid_dim, block_dim, 0, stream1>>>(d_c, d_a, d_b, SIZE, step);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaMemcpyAsync(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost, stream1));
          // Wait for all async operations in stream1 to complete before asserting
          CUDA_CHECK(cudaStreamSynchronize(stream1));
        },
        &config);
    std::cout << "Size used: " << SIZE << '\n';

    assert_vector_equal(h_c, SIZE, 3.0f);
    free_async_device_ptr(stream1, d_a, d_b, d_c);
  }
}

TEST_F(CudaPoc_Day0502_Stream, AllocateInPinMem) {
  constexpr size_t SIZE = 1 << 20;  // 4MB for float
  size_t size_bytes = SIZE * sizeof(float);

  dim3 block_dim(256);
  dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

  // float - zero-copy with mapped pinned memory
  {
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaHostAlloc(&h_a, size_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_b, size_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_c, size_bytes, cudaHostAllocMapped));

    // Initialize host memory
    for (size_t i = 0; i < SIZE; ++i) {
      h_a[i] = 1.0f;
      h_b[i] = 2.0f;
      h_c[i] = 0.0f;
    }

    // Get device pointers for zero-copy access
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_a, h_a, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_b, h_b, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_c, h_c, 0));

    KernelConfig config(grid_dim, block_dim);
    timeKernel(
        "vector_add_pinned_zerocopy",
        [&]() {
          // No memcpy needed - kernel directly accesses host memory via PCIe
          size_t step = block_dim.x * grid_dim.x;
          vector_add_kernel<<<grid_dim, block_dim>>>(d_c, d_a, d_b, SIZE, step);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          // Results automatically in h_c
        },
        &config);
    std::cout << "Size used: " << SIZE << '\n';

    // Convert to vector for assertion (or create a pointer-based assert function)
    std::vector<float> h_c_vec(h_c, h_c + SIZE);
    assert_vector_equal(h_c_vec, SIZE, 3.0f);
    
    free_host_ptr(h_a, h_b, h_c);
  }
}

TEST_F(CudaPoc_Day0502_Stream, AllocateInMangedMem) {
  constexpr size_t SIZE = 1 << 20;  // 4MB for float
  size_t size_bytes = SIZE * sizeof(float);

  dim3 block_dim(256);
  dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

  // float - unified memory (single pointer for host and device)
  // page fault might impact perf
  {
    float *data_a, *data_b, *data_c;
    CUDA_CHECK(cudaMallocManaged(&data_a, size_bytes));
    CUDA_CHECK(cudaMallocManaged(&data_b, size_bytes));
    CUDA_CHECK(cudaMallocManaged(&data_c, size_bytes));

    // Initialize memory from host - same pointer works everywhere!
    for (size_t i = 0; i < SIZE; ++i) {
      data_a[i] = 1.0f;
      data_b[i] = 2.0f;
      data_c[i] = 0.0f;
    }

    // No cudaHostGetDevicePointer needed - same pointer works on device!

    KernelConfig config(grid_dim, block_dim);
    timeKernel(
        "vector_add_unified_memoryed",
        [&]() {
          // Use same pointers directly - CUDA runtime handles migration
          size_t step = block_dim.x * grid_dim.x;
          vector_add_kernel<<<grid_dim, block_dim>>>(data_c, data_a, data_b, SIZE, step);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          // Results automatically accessible from host
        },
        &config);
    std::cout << "Size used: " << SIZE << '\n';

    // Direct access from host - no copy needed!
    std::vector<float> result_vec(data_c, data_c + SIZE);
    assert_vector_equal(result_vec, SIZE, 3.0f);

    free_device_ptr(data_a, data_b, data_c);
  }
}

// TEST_F(CudaPoc_Day0502_Stream, AllocateInMangedMemWithAdvise) {
//   constexpr size_t SIZE = 1 << 20;  // 4MB for float
//   size_t size_bytes = SIZE * sizeof(float);
//   int device = -1;
//   cudaGetDevice(&device);
//
//   dim3 block_dim(256);
//   dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);
//
//   // float - unified memory (single pointer for host and device)
//   // page fault might impact perf
//   {
//     float *data_a, *data_b, *data_c;
//     CUDA_CHECK(cudaMallocManaged(&data_a, size_bytes));
//     CUDA_CHECK(cudaMallocManaged(&data_b, size_bytes));
//     CUDA_CHECK(cudaMallocManaged(&data_c, size_bytes));
//
//     // Initialize memory from host - same pointer works everywhere!
//     for (size_t i = 0; i < SIZE; ++i) {
//       data_a[i] = 1.0f;
//       data_b[i] = 2.0f;
//       data_c[i] = 0.0f;
//     }
//
//     // Note: cudaMemAdvise and cudaMemPrefetchAsync API changed in some CUDA versions
//     // Commenting out to avoid type compatibility issues with CUDA 13.1
//     // Unified memory will still work, just with automatic on-demand migration
//     // CUDA_CHECK(cudaMemAdvise(data_a, size_bytes, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
//     // CUDA_CHECK(cudaMemAdvise(data_b, size_bytes, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
//     // CUDA_CHECK(cudaMemAdvise(data_c, size_bytes, cudaMemAdviseSetPreferredLocation, device));
//     // CUDA_CHECK(cudaMemPrefetchAsync(data_a, size_bytes, device, 0));
//     // CUDA_CHECK(cudaMemPrefetchAsync(data_b, size_bytes, device, 0));
//     // CUDA_CHECK(cudaMemPrefetchAsync(data_c, size_bytes, device, 0));
//
//     // No cudaHostGetDevicePointer needed - same pointer works on device!
//
//     KernelConfig config(grid_dim, block_dim);
//     timeKernel(
//         "vector_add_unified_memoryed",
//         [&]() {
//           // Use same pointers directly - CUDA runtime handles migration
//           size_t step = block_dim.x * grid_dim.x;
//           vector_add_kernel<<<grid_dim, block_dim>>>(data_c, data_a, data_b, SIZE, step);
//           CUDA_CHECK(cudaGetLastError());
//           CUDA_CHECK(cudaDeviceSynchronize());
//           // Results automatically accessible from host
//         },
//         &config);
//     std::cout << "Size used: " << SIZE << '\n';
//
//     // Direct access from host - no copy needed!
//     std::vector<float> result_vec(data_c, data_c + SIZE);
//     assert_vector_equal(result_vec, SIZE, 3.0f);
//
//     free_device_ptr(data_a, data_b, data_c);
//   }
// }

}  // namespace cuda_poc::day05cu