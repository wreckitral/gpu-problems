//Problem Description
//
// Perform the GELU (Gaussian Error Linear Unit) activation function on an input matrix:
// $$
// C[i][j] = \text{GELU}(A[i][j])
// $$
//
// The GELU function is defined as:
// $$
// \text{GELU}(x) = x \cdot \Phi(x)
// $$
//
// where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.
//
// A common approximation for GELU is:
// $$
// \text{GELU}(x) \approx 0.5x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715x^3)))
// $$
//
// ## Input:
// - Matrix $A$ of size $M \times N$ containing floating-point values
//
// ## Output:
// - Matrix $C$ of size $M \times N$ containing the GELU activation values
//
// ## Notes:
// - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
// - You should implement the approximation formula for GELU defined above
// - GELU is commonly used in modern transformer-based neural networks like BERT and GPT

#include <cuda_runtime.h>

__device__ __host__ inline float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}
__global__ void kernel(const float* input, float* output, size_t n, size_t m) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < n && y < m) {
        output[y * n + x] = gelu(input[y * n + x]);
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
