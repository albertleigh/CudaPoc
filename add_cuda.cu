template<typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// C++ callable wrapper function
void launch_add_kernel(float *c, const float *a, const float *b, int n, dim3 grid_dim, dim3 block_dim) {
    add_kernel<float><<<grid_dim, block_dim>>>(c, a, b, n);
}
