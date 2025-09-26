__global__ void relu_kernel(const float* input, float* output, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < N) {
        output[x] = fmaxf(input[x], 0.0f);
    }
}
