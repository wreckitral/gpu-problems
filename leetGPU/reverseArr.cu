#include <__clang_cuda_builtin_vars.h>
__global__ void reverse_array(float* input, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    float temp;

    if (x < (N + 1) / 2) {
        temp = input[x];
        input[x] = input[N - 1 - x];
        input[N - 1 - x] = temp;
    }
}

// 0, 1, 2, 3, 4, 5 lets say this is a index of an array
// 5, 4, 3, 2, 1, 0
