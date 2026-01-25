//
// Created by Albert Li on 1/23/26.
//
#include "pipeline_cuda.cuh"

#include <cooperative_groups.h>
#include <cuda_pipeline.h>
#include <cuda/pipeline>

namespace cuda_poc::pipeline {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 900)

// Dummy compute function that processes data from shared memory to global memory
template <typename T>
__device__ void compute(T* global_out, T const* shared_in) {
  auto block = cooperative_groups::this_thread_block();
  size_t tid = block.thread_rank();
  // Simple computation: copy and add 1 to verify pipeline is working
  global_out[tid] = shared_in[tid] + 1;
}

// Producer: acquire -> (submit) -> commit
// Consumer: wait -> (compute) -> release
template <size_t stages_count = 2>
__global__ void with_staging_kernel(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size());

  // stages_count * block.size() * sizeof(int) bytes
  extern __shared__ int shared[];

  size_t shared_offset[stages_count];

  for (int s = 0; s < stages_count; ++s) {
    shared_offset[s] = block.size() * s;
  }

  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state);
  auto block_batch = [&](size_t batch) -> int { return block.group_index().x * block.size() + grid.size() * batch; };

  // compute_basic: next batch to process
  // fetch_basic: next batch to fetch from global memory
  for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
    // The outer loop iterates over the computation of the batches
    for (; fetch_batch < batch_sz && fetch_batch < compute_batch + stages_count; ++fetch_batch) {
      // This inner loop iterates over the memory transfers, making sure that the pipeline is always full.
      pipeline.producer_acquire();
      size_t shared_idx = fetch_batch % stages_count;
      size_t batch_idx = fetch_batch;
      size_t block_batch_idx = block_batch(batch_idx);
      cuda::memcpy_async(block, shared + shared_offset[shared_idx], global_in + block_batch_idx,
                         sizeof(int) * block.size(), pipeline);
      pipeline.producer_commit();
    }

    pipeline.consumer_wait();
    size_t shared_idx = compute_batch % stages_count;
    size_t batch_idx = compute_batch;
    compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
    pipeline.consumer_release();
  }
}

void compute_with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz, dim3 grid, dim3 block) {
  constexpr size_t stages_count = 2;
  size_t smem_size = stages_count * block.x * sizeof(int);
  with_staging_kernel<stages_count><<<grid, block, smem_size>>>(global_out, global_in, size, batch_sz);
}
#endif

template <typename T>
__global__ void sync_compute_kernel(T* global, T* output, uint64_t* clock, size_t copy_count, size_t total_element) {
  extern __shared__ char s[];
  T* shared = reinterpret_cast<T*>(s);

  const size_t block_offset = blockIdx.x * blockDim.x * copy_count;
  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    const size_t local_idx = i * blockDim.x + threadIdx.x;
    const size_t global_idx = block_offset + local_idx;
    if (global_idx < total_element) {
      shared[local_idx] = global[global_idx];
    }
  }

  __syncthreads();

  uint64_t clock_end = clock64();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(clock), clock_end - clock_start);
  }

  for (size_t i = 0; i < copy_count; ++i) {
    const size_t local_idx = i * blockDim.x + threadIdx.x;
    const size_t global_idx = block_offset + local_idx;
    if (global_idx < total_element) {
      output[global_idx] = shared[local_idx] + 1;
    }
  }
}

template <typename T>
void sync_compute(T* global,
                  T* output,
                  uint64_t* clock,
                  size_t copy_count,
                  size_t total_element,
                  dim3 grid,
                  dim3 block) {
  size_t smeme_size = copy_count * block.x * sizeof(T);
  sync_compute_kernel<T><<<grid, block, smeme_size>>>(global, output, clock, copy_count, total_element);
}

template void sync_compute<int>(int* global,
                                int* output,
                                uint64_t* clock,
                                size_t copy_count,
                                size_t total_element,
                                dim3 grid,
                                dim3 block);

template <typename T>
__global__ void sync_copy_kernel(T* global, T* output, uint64_t* clock, size_t copy_count, size_t total_element) {
  extern __shared__ char s[];
  T* shared = reinterpret_cast<T*>(s);

  const size_t block_offset = blockIdx.x * blockDim.x * copy_count;
  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    const size_t local_idx = i * blockDim.x + threadIdx.x;
    const size_t global_idx = block_offset + local_idx;
    if (global_idx < total_element) {
      shared[local_idx] = global[global_idx];
    }
  }

  __syncthreads();

  uint64_t clock_end = clock64();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(clock), clock_end - clock_start);
  }

  for (size_t i = 0; i < copy_count; ++i) {
    const size_t local_idx = i * blockDim.x + threadIdx.x;
    const size_t global_idx = block_offset + local_idx;
    if (global_idx < total_element) {
      output[global_idx] = shared[local_idx];
    }
  }
}

template <typename T>
void sync_copy(T* global, T* output, uint64_t* clock, size_t copy_count, size_t total_element, dim3 grid, dim3 block) {
  size_t smeme_size = copy_count * block.x * sizeof(T);
  sync_copy_kernel<T><<<grid, block, smeme_size>>>(global, output, clock, copy_count, total_element);
}

template void sync_copy<int>(int* global,
                             int* output,
                             uint64_t* clock,
                             size_t copy_count,
                             size_t total_element,
                             dim3 grid,
                             dim3 block);

// pipeline status: issue -> commit -> wait
template <typename T>
__global__ void async_copy_kernel(T* global, T* output, uint64_t* clock, size_t copy_count, size_t total_element) {
  extern __shared__ char s[];
  T* shared = reinterpret_cast<T*>(s);

  const size_t block_offset = blockIdx.x * blockDim.x * copy_count;
  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    const size_t local_idx = i * blockDim.x + threadIdx.x;
    const size_t global_idx = block_offset + local_idx;
    if (global_idx < total_element) {
      // `__pipeline_memcpy_async` can only copy from global to shared memory, and size_and_align must be of 4, 8, or
      // 16;
      __pipeline_memcpy_async(&shared[local_idx], &global[global_idx], sizeof(T));
    }
  }

  __pipeline_commit();
  // Waits for all but the last N async operations to finish.
  __pipeline_wait_prior(0);

  uint64_t clock_end = clock64();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(clock), clock_end - clock_start);
  }

  for (size_t i = 0; i < copy_count; ++i) {
    const size_t local_idx = i * blockDim.x + threadIdx.x;
    const size_t global_idx = block_offset + local_idx;
    if (global_idx < total_element) {
      output[global_idx] = shared[local_idx];
    }
  }
}

template <typename T>
void async_copy(T* global, T* output, uint64_t* clock, size_t copy_count, size_t total_element, dim3 grid, dim3 block) {
  size_t smeme_size = copy_count * block.x * sizeof(T);
  async_copy_kernel<T><<<grid, block, smeme_size>>>(global, output, clock, copy_count, total_element);
}

template void async_copy<int>(int* global,
                              int* output,
                              uint64_t* clock,
                              size_t copy_count,
                              size_t total_element,
                              dim3 grid,
                              dim3 block);

}  // namespace cuda_poc::pipeline