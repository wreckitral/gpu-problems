__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(col < N) {
        C[col] = A[col] + B[col];
    }

}