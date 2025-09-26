__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if(j < cols && i < rows) {
        output[j * rows + i] = input[i * cols + j];
    }
}