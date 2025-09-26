__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < M && col < K) {
        float sum = 0;

        for(int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[K * i + col];

        }

        C[row * K + col] = sum;
    }
}