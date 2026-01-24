//
// Created by ali on 1/25/26.
//

#include "subtract_by_cuda.cuh"

#include <type_traits>
#include "cuda_fp16.h"

namespace cuda_poc {

template <typename T>
__device__ T subtract_by_op(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, half>) {
    return __hsub(a, b);
  } else if constexpr (std::is_same_v<T, half2>) {
    return __hsub2(a, b);
  } else if constexpr (std::is_same_v<T, float2>) {
    return make_float2(a.x - b.x, a.y - b.y);
  } else if constexpr (std::is_same_v<T, float3>) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
  } else if constexpr (std::is_same_v<T, float4>) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
  } else {
    return a - b;
  }
}

template <typename T>
__global__ void subtract_by_kernel(T* c, const T* a, T b, size_t n, size_t step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += step) {
    c[i] = subtract_by_op(a[i], b);
  }
}

// C++ callable wrapper function
template <typename T>
void vector_subtract_by(T* c, const T* a, T b, size_t n, dim3 grid_dim, dim3 block_dim) {
  size_t step = block_dim.x * grid_dim.x;
  subtract_by_kernel<T><<<grid_dim, block_dim>>>(c, a, b, n, step);
}

// Explicit template instantiations
template void vector_subtract_by<float>(float* c, const float* a, float b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_subtract_by<float2>(float2* c, const float2* a, float2 b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_subtract_by<float3>(float3* c, const float3* a, float3 b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_subtract_by<float4>(float4* c, const float4* a, float4 b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_subtract_by<double>(double* c, const double* a, double b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_subtract_by<half>(half* c, const half* a, half b, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_subtract_by<half2>(half2* c, const half2* a, half2 b, size_t n, dim3 grid_dim, dim3 block_dim);

}  // namespace cuda_poc
