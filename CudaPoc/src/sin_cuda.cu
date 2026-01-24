//
// Created by ali on 1/25/26.
//

#include "sin_cuda.cuh"

#include <type_traits>
#include "cuda_fp16.h"

namespace cuda_poc::inplace {

template <typename T>
__device__ T sin_op(const T& a) {
  if constexpr (std::is_same_v<T, half>) {
    return hsin(a);
  } else if constexpr (std::is_same_v<T, half2>) {
    return h2sin(a);
  } else if constexpr (std::is_same_v<T, float>) {
    return sinf(a);
  } else if constexpr (std::is_same_v<T, float2>) {
    return make_float2(sinf(a.x), sinf(a.y));
  } else if constexpr (std::is_same_v<T, float3>) {
    return make_float3(sinf(a.x), sinf(a.y), sinf(a.z));
  } else if constexpr (std::is_same_v<T, float4>) {
    return make_float4(sinf(a.x), sinf(a.y), sinf(a.z), sinf(a.w));
  } else if constexpr (std::is_same_v<T, double>) {
    return sin(a);
  } else {
    return sinf(a);
  }
}

template <typename T>
__global__ void sin_kernel(T* a, size_t n, size_t step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += step) {
    a[i] = sin_op(a[i]);
  }
}

// C++ callable wrapper function
template <typename T>
void vector_sin(T* a, size_t n, dim3 grid_dim, dim3 block_dim) {
  size_t step = block_dim.x * grid_dim.x;
  sin_kernel<T><<<grid_dim, block_dim>>>(a, n, step);
}

// Explicit template instantiations
template void vector_sin<float>(float* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sin<float2>(float2* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sin<float3>(float3* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sin<float4>(float4* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sin<double>(double* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sin<half>(half* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sin<half2>(half2* a, size_t n, dim3 grid_dim, dim3 block_dim);

}  // namespace cuda_poc
