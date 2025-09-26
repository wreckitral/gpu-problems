# [LeetGPU](https://leetgpu.com/challenges)

1. [Vector Addition](https://leetgpu.com/challenges/vector-addition) - adds 2 vectors `vecAdd.cu`
2. [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) - naive matrix multiplication `matMul.cu`
3. [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) - transpose the matrix A[i][j] -> B[j][i]       `matTrans.cu`
4. [Color Inversion](https://leetgpu.com/challenges/color-inversion) - invert color of an image
5. [1D Convolution](https://leetgpu.com/challenges/1d-convolution) - implement 1D convolution operation
6. [Reverse Array](https://leetgpu.com/challenges/reverse-array) - reverse an array in-place
7. [ReLU](https://leetgpu.com/challenges/relu) - implement a program that performs
the rectified Linier Unit (ReLU) activation function
(im trying to avoid divergence, so i use CUDA's math function fmaxf(), which return the largest of two floats,
this translate to only one instruction, no branching)
8. [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) - same with the relu but has additional constant, 0.01
9. [Rainbow Table](https://leetgpu.com/challenges/rainbow-table) - hash an array of integer, per element R rounds of hashing
9. [Matrix Copy](https://leetgpu.com/challenges/matrix-copy) - copy matrix A to B, i just use 1D mapping, but i dont know whether this is optimal or not, i'll benchmark this later
10. [Count Array Element](https://leetgpu.com/challenges/count-array-element) - count the number of element in a array
11. [Count 2D Array Element](https://leetgpu.com/challenges/count-2d-array-element) - count the number of element in a matrix/2d arrayarray
11. [Sigmoid Linear Unit](https://leetgpu.com/challenges/sigmoid-linear-unit) - implement SiLU activation function to a 1D vector
12. [Swish-Gated Linear Unit](https://leetgpu.com/challenges/swish-gated-linear-unit) - implement SWiGLU activation function to a 1D vector
