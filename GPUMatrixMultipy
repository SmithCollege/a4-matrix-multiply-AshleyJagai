#include <stdio.h>
#include <stdlib.h>

// Function to multiply two matrices on the GPU
__global__ void matrixMultiply(float *A, float *B, float *C, int M, int N, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < P) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * P + j];
        }
        C[i * P + j] = sum;
    }
}

int main() {
    int M = 100; // Number of rows in matrix A
    int N = 100; // Number of columns in matrix A and rows in matrix B
    int P = 100; // Number of columns in matrix B

    // Allocate memory for matrices A, B, and C
    float *A = (float *)malloc(M * N * sizeof(float));
    float *B = (float *)malloc(N * P * sizeof(float));
    float *C = (float *)malloc(M * P * sizeof(float));

    // Initialize matrices A and B
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            B[i * P + j] = i + j;
        }
    }

    // Allocate memory on the GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * P * sizeof(float));
    cudaMalloc((void **)&d_C, M * P * sizeof(float));

    // Copy matrices A and B to the GPU
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((M + blockSize - 1) / blockSize, (P + blockSize - 1) / blockSize
