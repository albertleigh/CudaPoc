//
// Created by ali on 1/25/26.
//

#include "add_one_cuda.cuh"

#include <type_traits>
#include "cuda_fp16.h"

namespace cuda_poc::inplace {

template <typename T>
__device__ T add_one_op(const T& a) {
  if constexpr (std::is_same_v<T, half>) {
    return __hadd(a, __float2half(1.0f));
  } else if constexpr (std::is_same_v<T, half2>) {
    return __hadd2(a, __float2half2_rn(1.0f));
  } else if constexpr (std::is_same_v<T, float2>) {
    return make_float2(a.x + 1.0f, a.y + 1.0f);
  } else if constexpr (std::is_same_v<T, float3>) {
    return make_float3(a.x + 1.0f, a.y + 1.0f, a.z + 1.0f);
  } else if constexpr (std::is_same_v<T, float4>) {
    return make_float4(a.x + 1.0f, a.y + 1.0f, a.z + 1.0f, a.w + 1.0f);
  } else {
    return a + static_cast<T>(1);
  }
}

template <typename T>
__global__ void add_one_kernel(T* a, size_t n, size_t step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += step) {
    a[i] = add_one_op(a[i]);
  }
}

// C++ callable wrapper function
template <typename T>
void vector_add_one(T* a, size_t n, dim3 grid_dim, dim3 block_dim) {
  size_t step = block_dim.x * grid_dim.x;
  add_one_kernel<T><<<grid_dim, block_dim>>>(a, n, step);
}

// Explicit template instantiations
template void vector_add_one<float>(float* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add_one<float2>(float2* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add_one<float3>(float3* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add_one<float4>(float4* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add_one<double>(double* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add_one<half>(half* a, size_t n, dim3 grid_dim, dim3 block_dim);
template void vector_add_one<half2>(half2* a, size_t n, dim3 grid_dim, dim3 block_dim);

}  // namespace cuda_poc::inplace
