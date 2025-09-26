__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < width * height) {
        int startIdx = x * 4;
        image[startIdx] = 255 - image[startIdx];
        image[startIdx + 1] = 255 - image[startIdx + 1];
        image[startIdx + 2] = 255 - image[startIdx + 2];
    }
}
