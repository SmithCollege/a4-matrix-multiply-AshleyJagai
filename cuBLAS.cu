#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

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

    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set the matrix dimensions
    int lda = N;
    int ldb = P;
    int ldc = P;

    // Set the alpha and beta values
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform the matrix multiplication using cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, P, N, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            printf("%f ", C[i * P + j]);
        }
        printf("\n");
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy the cuBLAS handle
    cublasDestroy(handle);

    return 0;
}
