//Problem Description
//
// Perform the ReLU (Rectified Linear Unit) activation function on an input matrix:
// $$
// C[i][j] = \max(0, A[i][j])
// $$
//
// The ReLU function is defined as:
// $$
// f(x) = \begin{cases}
// x & \text{if } x > 0 \\
// 0 & \text{if } x \leq 0
// \end{cases}
// $$
//
// ## Input:
// - Matrix $A$ of size $M \times N$ containing floating-point values
//
// ## Output:
// - Matrix $C$ of size $M \times N$ containing the ReLU activation values
//
// ## Notes:
// - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
// - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/19_ReLU.py)
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, size_t n, size_t m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m && y < n) {
        int idx = y * m + x;
        float val = fmax(input[idx], 0);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    relu_kernel<<<gridDim, blockDim>>>(input, output, n, m);
    cudaDeviceSynchronize();
}

