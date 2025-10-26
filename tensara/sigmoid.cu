#include <cuda_runtime.h>

__global__ void sigmoid_kernel(const float* input, float* output, size_t n, size_t m) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = n * m;

    if(x < total) {
        float i = input[x];
        output[x] = 1.0f / (1.0f + expf(-i));
    }
}

// input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
