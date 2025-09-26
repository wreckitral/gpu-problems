#include <cmath>
__global__ void silu_kernel(const float* input, float* output, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < N) {
        float i = input[x];
        float sigmoid = 1.0f/ (1.0f+ expf(-i));
        output[x] = i * sigmoid;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
