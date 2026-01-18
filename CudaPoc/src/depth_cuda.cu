//
// Created by Albert Li on 1/18/2026.
//

#include "depth_cuda.cuh"

#include <cstdio>

namespace cuda_poc::depth {
    __global__ void depth_kernel(int current_depth, int max_depth) {
        printf("Depth %d: Block %d Thread %d \n", current_depth, blockIdx.x, threadIdx.x);
        if (current_depth < max_depth) {
            const int next_depth = current_depth + 1;
            depth_kernel<<<2, 2>>>(next_depth, max_depth);
        }
    }

    void depth_visit(int current_depth, int max_depth) {
        depth_kernel<<<1, 1>>>(current_depth, max_depth);
    }
} // namespace cuda_poc::depth
