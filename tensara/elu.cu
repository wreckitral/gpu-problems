#include <cuda_runtime.h>

__global__ void kernel(const float* input, float* output, size_t n, size_t m, float alpha) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * m;


    if (x < total) {
        output[x] = input[x] > 0 ? input[x] : alpha * (expf(input[x]) - 1);
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m, alpha);
    cudaDeviceSynchronize();
}
