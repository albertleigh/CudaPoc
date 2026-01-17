//
// Created by Albert Li on 1/16/2026.
//

#include "transpose_cuda.cuh"
#include "cuda_fp16.h"

namespace cuda_poc {
    template<typename T>
    __global__ void transpose_kernel(T *output, const T *input, const int rows, const int cols) {
        __shared__ T tile[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM + 1]; // +1 to avoid bank conflicts

        // Calculate global indices for reading from input (rows x cols)
        int x_in = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.x; // column in input
        int y_in = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.y; // row in input

        // Load data into shared memory (coalesced read)
        if (x_in < cols && y_in < rows) {
            tile[threadIdx.y][threadIdx.x] = input[y_in * cols + x_in];
        }

        __syncthreads();

        // Calculate global indices for writing to output (cols x rows)
        int x_out = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.x; // column in output (was row in input)
        int y_out = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.y; // row in output (was column in input)

        // Write data from shared memory to output (coalesced write)
        if (x_out < rows && y_out < cols) {
            output[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
        }
    }

    // C++ callable wrapper function
    template<typename T>
    void transpose(T *output, const T *input, int rows, int cols, dim3 grid_dim, dim3 block_dim) {
        // Launch kernel with statically allocated shared memory
        // Shared memory size is determined at compile time: TRANSPOSE_BLOCK_DIM x (TRANSPOSE_BLOCK_DIM + 1) x sizeof(T)
        // No need to specify dynamic shared memory parameter (3rd argument)
        // Input: rows x cols, Output: cols x rows
        transpose_kernel<T><<<grid_dim, block_dim>>>(output, input, rows, cols);
    }

    // Explicit template instantiations
    template void transpose<float>(float *output, const float *input, int rows, int cols, dim3 grid_dim,
                                   dim3 block_dim);

    template void transpose<double>(double *output, const double *input, int rows, int cols, dim3 grid_dim,
                                    dim3 block_dim);

    template void transpose<int>(int *output, const int *input, int rows, int cols, dim3 grid_dim, dim3 block_dim);

    template void transpose<half>(half *output, const half *input, int rows, int cols, dim3 grid_dim, dim3 block_dim);
} // namespace cuda_poc
