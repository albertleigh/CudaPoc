#include "add_cuda.cuh"

#include <type_traits>
#include "cuda_fp16.h"

namespace cuda_poc {
template <typename T>
__device__ T add(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, half>) {
    return __hadd(a, b);
  } else if constexpr (std::is_same_v<T, half2>) {
    return __hadd2(a, b);
  } else if constexpr (std::is_same_v<T, float2>) {
    return make_float2(a.x + b.x, a.y + b.y);
  } else if constexpr (std::is_same_v<T, float3>) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
  } else if constexpr (std::is_same_v<T, float4>) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  } else {
    return a + b;
  }
}

template <typename T>
__global__ void vector_add_kernel(T* c, const T* a, const T* b, size_t n, size_t step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += step) {
    c[i] = add(a[i], b[i]);
  }
}

// C++ callable wrapper function
template <typename T>
void vector_add(T* c, const T* a, const T* b, size_t n, dim3 grid_dim, dim3 block_dim) {
  size_t step = block_dim.x * grid_dim.x;
  vector_add_kernel<T><<<grid_dim, block_dim>>>(c, a, b, n, step);
}

template void vector_add<float>(float* c, const float* a, const float* b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add<float2>(float2* c, const float2* a, const float2* b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add<float3>(float3* c, const float3* a, const float3* b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add<float4>(float4* c, const float4* a, const float4* b, size_t n, dim3 grid_dim, dim3 block_dim);

template void vector_add<half>(half* c, const half* a, const half* b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add<half2>(half2* c, const half2* a, const half2* b, size_t n, dim3 grid_dim, dim3 block_dim);
}  // namespace cuda_poc
