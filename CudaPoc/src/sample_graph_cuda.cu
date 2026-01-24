//
// Created by ali on 1/25/26.
//

#include "add_one_cuda.cuh"
#include "multiply_by_cuda.cuh"
#include "sample_graph_cuda.cuh"

#include "cuda_utils.h"
#include "sin_cuda.cuh"
#include "sqrt_cuda.cuh"
#include "subtract_by_cuda.cuh"

namespace cuda_poc::sample_graph {
void regular_approach(float* d_data, int n, int block_size, cudaStream_t stream) {
  using namespace cuda_poc::inplace;

  int num_blocks = (n + block_size - 1) / block_size;
  size_t step = block_size * num_blocks;

  sin_kernel<<<num_blocks, block_size, 0, stream>>>(d_data, n, step);
  add_one_kernel<<<num_blocks, block_size, 0, stream>>>(d_data, n, step);
  sqrt_kernel<<<num_blocks, block_size, 0, stream>>>(d_data, n, step);
  multiply_by_kernel<<<num_blocks, block_size, 0, stream>>>(d_data, 2.0f, n, step);
  subtract_by_kernel<<<num_blocks, block_size, 0, stream>>>(d_data, 1.0f, n, step);
}
void create_graph(cudaGraphExec_t* graph_exec, float* d_data, int n, int block_size) {
  cudaStream_t capture_stream;
  cudaGraph_t graph;

  CUDA_CHECK(cudaStreamCreate(&capture_stream));

  // Begin capture graph
  CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));

  // Capture the sequence of kernels
  regular_approach(d_data, n, block_size, capture_stream);

  // End capture and instantiate
  CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0));

  // Clean up
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(capture_stream));
}
void execute_graph(cudaGraphExec_t graph_exec, cudaStream_t stream) {
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
}
}  // namespace cuda_poc::sample_graph