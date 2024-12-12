#include <stdio.h>
#include <stdlib.h>

// Function to multiply two matrices
void matrixMultiply(float *A, float *B, float *C, int M, int N, int P) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            C[i * P + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * P + j] += A[i * N + k] * B[k * P + j];
            }
        }
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

    // Multiply matrices A and B
    matrixMultiply(A, B, C, M, N, P);

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

    return 0;
}
