//Problem Description
//
// Perform 1D convolution between an input signal and a kernel:
// $$
// \text{C}[i] = \sum_{j=0}^{K-1} \text{A}[i + j] \cdot \text{B}[j]
// $$
//
// The convolution operation slides the kernel over the input signal, computing the sum of element-wise multiplications at each position. Zero padding is used at the boundaries.
//
// ## Input:
// - Vector $\text{A}$ of size $\text{N}$ (input signal)
// - Vector $\text{B}$ of size $\text{K}$ (convolution kernel)
//
// ## Output:
// - Vector $\text{C}$ of size $\text{N}$ (convolved signal)
//
// ## Notes:
// - $\text{K}$ is odd and smaller than $\text{N}$
// - Use zero padding at the boundaries where the kernel extends beyond the input signal
// - The kernel is centered at each position, with $(K-1)/2$ elements on each side
// - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/67_conv_standard_1D.py)

#include <cuda_runtime.h>

__global__ void kernel(const float* A, const float* B, float* C, size_t N, size_t K) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int half = K / 2;

    if(x < N) {
        float sum = 0;

        for(int i = 0; i < K; i ++) {
            int idx = x + i - half;
            if(idx >= 0 && idx < N) {
                sum += A[idx] * B[i];
            }
        }

        C[x] = sum;
    }
}

// Note: A, B, C are all device pointers to float16 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, K);

    cudaDeviceSynchronize();
}
