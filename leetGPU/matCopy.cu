__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < N * N) {
        B[x] = A[x];
    }
}
