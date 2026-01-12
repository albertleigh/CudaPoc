template<typename T>
__global__ void add_kernel_inner_loop(T *c, const T *a, const T *b, size_t n, size_t step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

// C++ callable wrapper function
void vector_add(float *c, const float *a, const float *b, int n, dim3 grid_dim, dim3 block_dim) {
    size_t step = block_dim.x * grid_dim.x;
    add_kernel_inner_loop<float><<<grid_dim, block_dim>>>(c, a, b, n, step);
}
