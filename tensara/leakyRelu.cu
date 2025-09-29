//Problem Description
//
// Perform the Leaky ReLU (Leaky Rectified Linear Unit) activation function on an input matrix:
// $$
// \text{C}[i][j] = \max(\alpha \cdot \text{A}[i][j], \text{A}[i][j])
// $$
// where $\alpha$ is a small positive constant (e.g. 0.01)
//
// The Leaky ReLU function is defined as:
// $$
// f(x) = \begin{cases}
// x & \text{if } x > 0 \\
// \alpha x & \text{if } x \leq 0
// \end{cases}
// $$
//
// ## Input:
// - Matrix $\text{A}$ of size $M \times N$
// - $\alpha$ value (slope for negative values)
//
// ## Output:
// - Matrix $\text{C}$ of size $M \times N$
//
// ## Notes:
// - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
// - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/20_LeakyReLU.py)
#include <cuda_runtime.h>

__global__ void kernel(const float* input, float alpha, float* output, size_t n, size_t m) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;


    if(y < m && x < n) {
        int idx = y * n + x;
        output[idx] = input[idx] <= 0 ? alpha * input[idx] : input[idx];
    }
}

// Note: input, output are all device pointers to float16 arrays
extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
    kernel<<<gridDim, blockDim>>>(input, alpha, output, n, m);
    cudaDeviceSynchronize();
}
