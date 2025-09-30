//Problem Description
//
// Perform multiplication of a matrix and a vector:
// $$
// C[i] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k]
// $$
//
// ## Input:
// - Matrix $A$ of size $M \times K$
// - Vector $B$ of size $K \times 1$
//
// ## Output:
// - Vector $C = AB$ of size $M \times 1$
//
// ## Notes:
// - Matrix $\text{A}$ is stored in row-major order
// - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/4_Matrix_vector_multiplication_.py)

#include <cuda_runtime.h>

__global__ void kernel(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < m) {
        float sum = 0;
        for (int i = 0; i < k; i++) {
            sum += input_a[x * k + i] * input_b[i];
        }

        output_c[x] = sum;
    }

}

// Note: input_a, input_b, output_c are all device pointers to float16 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernel<<<blocksPerGrid, threadsPerBlock>>>(input_a, input_b, output_c, m, k);

    cudaDeviceSynchronize();
}
