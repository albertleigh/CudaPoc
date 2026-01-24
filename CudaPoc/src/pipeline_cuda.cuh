//
// Created by ali on 1/23/26.
//

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda_poc::pipeline {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)

void compute_with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz, dim3 grid, dim3 block);
#endif

template <typename T>
void sync_compute(T* global, T* output, uint64_t* clock, size_t copy_count, size_t total_element, dim3 grid, dim3 block);

}  // namespace cuda_poc::pipeline
