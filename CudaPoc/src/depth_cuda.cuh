//
// Created by Albert Li on 1/18/2026.
//
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda_poc::depth {
void depth_visit(int current_depth, int max_depth);
}  // namespace cuda_poc::depth
