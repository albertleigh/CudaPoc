//
// Created by ali on 1/25/26.
//

#include "sqrt_cuda.cuh"

#include <type_traits>
#include "cuda_fp16.h"

namespace cuda_poc::inplace {

template <typename T>
__device__ T sqrt_op(const T& a) {
  if constexpr (std::is_same_v<T, half>) {
    return hsqrt(a);
  } else if constexpr (std::is_same_v<T, half2>) {
    return h2sqrt(a);
  } else if constexpr (std::is_same_v<T, float>) {
    return sqrtf(a);
  } else if constexpr (std::is_same_v<T, float2>) {
    return make_float2(sqrtf(a.x), sqrtf(a.y));
  } else if constexpr (std::is_same_v<T, float3>) {
    return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
  } else if constexpr (std::is_same_v<T, float4>) {
    return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
  } else if constexpr (std::is_same_v<T, double>) {
    return sqrt(a);
  } else {
    return sqrtf(a);
  }
}

template <typename T>
__global__ void sqrt_kernel(T* a, size_t n, size_t step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += step) {
    a[i] = sqrt_op(a[i]);
  }
}

// C++ callable wrapper function
template <typename T>
void vector_sqrt(T* a, size_t n, dim3 grid_dim, dim3 block_dim) {
  size_t step = block_dim.x * grid_dim.x;
  sqrt_kernel<T><<<grid_dim, block_dim>>>(a, n, step);
}

// Explicit template instantiations
template void vector_sqrt<float>(float* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sqrt<float2>(float2* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sqrt<float3>(float3* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sqrt<float4>(float4* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sqrt<double>(double* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sqrt<half>(half* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_sqrt<half2>(half2* a, size_t n, dim3 grid_dim, dim3 block_dim);

}  // namespace cuda_poc::inplace
