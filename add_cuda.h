#pragma once

#include <cuda_runtime.h>

// C++ callable wrapper function
void launch_add_kernel(float *c, const float *a, const float *b, int n, dim3 grid_dim, dim3 block_dim);
