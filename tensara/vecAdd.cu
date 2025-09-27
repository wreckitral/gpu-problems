//Problem Description
//
// Perform element-wise addition of two vectors:
// $$
// c_i = a_i + b_i
// $$
//
// ## Input
// - Vectors $a$ and $b$ of length $N$
//
// ## Output
// - Vector $c$ of length $N$ containing the element-wise sum


#include <cuda_runtime.h>

__global__ void kernel(const float* A, const float* B, float* C, size_t N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < N) {
        C[x] = A[x] + B[x];
    }
}

// Note: d_input1, d_input2, d_output are all device pointers to float16 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n);

    cudaDeviceSynchronize();
}
