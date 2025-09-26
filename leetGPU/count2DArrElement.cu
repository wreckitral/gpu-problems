__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(y < N && x < M && input[y * M + x] == K) {
        atomicAdd(output, 1);
    }
}
