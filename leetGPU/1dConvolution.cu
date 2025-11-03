__global__ void convolution_1d_kernel(const float* input, const float* kernel,
                                      float* output, int input_size, int kernel_size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < input_size - kernel_size + 1) {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++) {
            sum += input[x + i] * kernel[i];
        }
        output[x] = sum;
    }
}
