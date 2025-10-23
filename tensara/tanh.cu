//Problem Description
//
// Perform the Tanh activation function on an input matrix:
// $$
// C[i][j] = \text{tanh}(A[i][j])
// $$
//
// The Tanh function is defined as:
// $$
// \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
// $$
//
// ## Input:
// - Matrix $A$ of size $M \times N$ containing floating-point values
//
// ## Output:
// - Matrix $C$ of size $M \times N$ containing the Tanh activation values
//
// ## Notes:
// - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
// - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/22_Tanh.py)


#include <cuda_runtime.h>

__global__ void kernel(const float* input, float* output, size_t n, size_t m) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(y < m && x < n) {
        output[y * n + x] = tanhf(input[y * n + x]);
    }
}

__global__ void kernel_faster(const float* input, float* output, size_t n, size_t m) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(y < m && x < n) {
        output[y * n + x] = tanhf(input[y * n + x]);
    }
}

// Note: input, output are all device pointers to float16 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
    kernel<<<gridDim, blockDim>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
