#include <cuda_runtime.h>

__global__ void conv_kernel(const float* input, const float* kernel,
                       float* output, int input_rows, int input_cols, int kernel_rows,
                       int kernel_cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    if(row < output_rows && col < output_cols) {
        float sum = 0.0f;

        for (int i = 0; i < kernel_rows; i++) {
            for(int j = 0; j < kernel_cols; j++) {
                sum += input[(row + i) * input_cols + (col + j)] * kernel[i * kernel_cols + j];
            }
        }

        output[row * output_cols + col] = sum;
    }

}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 blockDim(16, 16);
    dim3 gridDim((output_cols + blockDim.x - 1) / blockDim.x,
                 (output_rows + blockDim.y - 1) / blockDim.y);

    conv_kernel<<<gridDim, blockDim>>>(input, kernel, output,
                                              input_rows, input_cols,
                                              kernel_rows, kernel_cols);
}
