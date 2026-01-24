//
// Created by ali on 1/25/26.
//

#pragma once
#include "add_cuda.cuh"

namespace cuda_poc::sample_graph {
void regular_approach(float* d_data, int n, int block_size, cudaStream_t stream);
void create_graph(cudaGraphExec_t* graph_exec, float* d_data, int n, int block_size);
void execute_graph(cudaGraphExec_t graph_exec, cudaStream_t stream);
}  // namespace cuda_poc::sample_graph