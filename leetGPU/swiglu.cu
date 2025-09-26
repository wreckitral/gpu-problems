__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < halfN) {
        float i = input[x];
        float sigmoid = 1.0f/ (1.0f+ expf(-i));
        float s = i * sigmoid;

        output[x] = input[halfN + x] * s;
    }
}
