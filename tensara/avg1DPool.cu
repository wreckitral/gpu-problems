#include <cuda_runtime.h>

__global__ void kernel(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int Hout = ((H + 2 * padding - kernel_size) / stride) + 1;

    if(x < Hout) {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++) {
            int idx = stride * x + i - padding;
            if(idx >= 0 && idx < H) sum += input[idx];
        }

        output[x] = sum / kernel_size;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    dim3 threadsPerBlock(256);
    dim3 blockPerGrid((H + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernel<<<blockPerGrid, threadsPerBlock>>>(input, kernel_size, stride, padding, output, H);

    cudaDeviceSynchronize();
}
