#include "cuda_fp16.h"

template <typename T>
__device__ T add(const T& a, const T& b)
{
    if constexpr (std::is_same_v<T, half>)
    {
        return __hadd(a, b);
    }

    if constexpr (std::is_same_v<T, half2>)
    {
        return __hadd2(a, b);
    }

    if constexpr (std::is_same_v<T, float2>)
    {
        return make_float2(a.x + b.x, a.y + b.y);
    }

    if constexpr (std::is_same_v<T, float3>)
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    if constexpr (std::is_same_v<T, float4>)
    {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }


    return a + b;
}

template <typename T>
__global__ void vector_add_kernel(T* c, const T* a, const T* b, size_t n, size_t step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += step)
    {
        c[i] = add(a[i], b[i]);
    }
}

// C++ callable wrapper function
void vector_add(float* c, const float* a, const float* b, int n, dim3 grid_dim, dim3 block_dim)
{
    size_t step = block_dim.x * grid_dim.x;
    vector_add_kernel<float><<<grid_dim, block_dim>>>(c, a, b, n, step);
}
