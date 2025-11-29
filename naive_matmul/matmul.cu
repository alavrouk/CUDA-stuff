// #include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__
void matmul_kernel(float* A_d, float* B_d, int d1_A, int shared_dim, int d2_B, float* output_d) {
    // get the indices of this particular thread
    int A_row = blockIdx.x * blockDim.x + threadIdx.x;
    int B_col = blockIdx.y * blockDim.y + threadIdx.y;

    // check for out of bounds since the tiling can go over
    if (A_row >= d1_A || B_col >= d2_B) {
        return;
    }

    // do the dot product
    float dot_product = 0;
    for (int i = 0; i < shared_dim; ++i) {
        float a = A_d[A_row * shared_dim + i]; // one row of A
        float b = B_d[i * d2_B + B_col]; // one col of B
        dot_product += a * b;
    }

    // index into the correct output position
    output_d[A_row * d2_B + B_col] = dot_product;
}

void matmul(const float* A_h, int d1_A, int d2_A, const float* B_h, int d1_B, int d2_B, float* output_h) {

    if (d2_A != d1_B) {
        fprintf(stderr, "Dimension Mismatch");
        return;
    }
    int sizeof_A = d1_A * d2_A * sizeof(float);
    int sizeof_B = d1_B * d2_B * sizeof(float);
    int sizeof_output = d1_A * d2_B * sizeof(float);

    float* A_d;
    cudaMalloc((void**) &A_d, sizeof_A);
    cudaMemcpy(A_d, A_h, sizeof_A, cudaMemcpyHostToDevice);
    float* B_d;
    cudaMalloc((void**) &B_d, sizeof_B);
    cudaMemcpy(B_d, B_h, sizeof_B, cudaMemcpyHostToDevice);
    float* output_d;
    cudaMalloc((void**) &output_d, sizeof_output);

    // Typical pattern, tiling over the output with 16x16 blocks
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(d1_A / float(blockDim.x)), ceil(d2_B / float(blockDim.y)));

    matmul_kernel<<<gridDim, blockDim>>>(A_d, B_d, d1_A, d2_A, d2_B, output_d);

    cudaMemcpy(output_h, output_d, sizeof_output, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(output_d);
}

// Claude driver code
int main() {
    float A[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };
    float B[3][2] = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    float C[2][2];
    float expected[2][2] = {
        {58, 64},
        {139, 154}
    };

    matmul((float*)A, 2, 3, (float*)B, 3, 2, (float*)C);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(C[i][j] - expected[i][j]) > 1e-5) {
                printf("FAIL at [%d][%d]: got %f, expected %f\n", i, j, C[i][j], expected[i][j]);
                return 1;
            }
        }
    }
    printf("PASS\n");
    return 0;
}
