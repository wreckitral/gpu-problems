#include <__clang_cuda_builtin_vars.h>
__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < N && input[x] == K) {
        atomicAdd(output, 1);
    }
}
