#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <string> 
using namespace  std;

#define CUDA_CHECK_RETURN(value) {                                                                           \
    cudaError_t error = value;                                                                               \
                                                                                                             \
    if (error != cudaSuccess) {                                                                              \
        fprintf(stderr, "Error %s at line %d at file %s\n", cudaGetErrorString(error), __LINE__, __FILE__);  \
        exit(1);                                                                                             \
    }                                                                                                        \
}

#define VECTOR_SIZE (100000000)
#define BLOCK_SIZE (2048)
#define GRID_SIZE ((VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);


__global__ void B_vec_GPU(int *data, int length) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < length) { 
        data[i] = 1 - length;
    }
}

__global__ void vector_difference(int *vectorA, int *vectorB, int *vectorC, int length) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length) {
        vectorC[i] = vectorA[i] - vectorB[i];
    }
}


void task1() {
    int *A = (int*) malloc(sizeof(int) * VECTOR_SIZE);
    int *B = (int*) malloc(sizeof(int) * VECTOR_SIZE);
    int *C = (int*) malloc(sizeof(int) * VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        A[i] = INT_MAX;
        B[i] = 1 - VECTOR_SIZE;
        C[i] = 0;
    }

    for (int i = 0; i < VECTOR_SIZE; i++) {
        C[i] = A[i] - B[i];
    }

    // for (int i = 0; i < VECTOR_SIZE; i++) {
    //     cout << to_string(A[i]) + " " + to_string(B[i]) + " " + to_string(C[i]) + "\n";
    // }

    free(A);
    free(B);
    free(C);
}

void task2() {
    int *A = (int*) malloc(sizeof(int) * VECTOR_SIZE);
    int *C = (int*) malloc(sizeof(int) * VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        A[i] = INT_MAX;
        C[i] = 0;
    }

    int *AD, *BD, *CD;
    CUDA_CHECK_RETURN(cudaMalloc(&AD, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMalloc(&BD, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMalloc(&CD, sizeof(int) * VECTOR_SIZE));

    CUDA_CHECK_RETURN(cudaMemcpy(AD, A, sizeof(int) * VECTOR_SIZE, cudaMemcpyHostToDevice));

    int block_size = BLOCK_SIZE;
    int grid_size = GRID_SIZE
    B_vec_GPU<<<grid_size, block_size>>>(BD, VECTOR_SIZE);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    vector_difference<<<grid_size, block_size>>>(AD, BD, CD, VECTOR_SIZE);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(C, CD, sizeof(int) * VECTOR_SIZE, cudaMemcpyDeviceToHost));

    // int* B = (int*) malloc(sizeof(int) * VECTOR_SIZE);
    // CUDA_CHECK_RETURN(cudaMemcpy(B, BD, sizeof(int) * VECTOR_SIZE, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < VECTOR_SIZE; i++) {
    //     cout << to_string(A[i]) + " " + to_string(B[i]) + " " + to_string(C[i]) + "\n";
    // }

    cudaFree(AD);
    cudaFree(BD);
    cudaFree(CD);

    free(A);
    free(C);
}

void task3() {
    int *A, *B, *C;
    CUDA_CHECK_RETURN(cudaMallocManaged(&A, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMallocManaged(&B, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMallocManaged(&C, sizeof(int) * VECTOR_SIZE));

    for (int i = 0; i < VECTOR_SIZE; i++) {
        A[i] = INT_MAX;
        B[i] = 1 - VECTOR_SIZE;
        C[i] = 0;
    }
    int block_size = BLOCK_SIZE;
    int grid_size = GRID_SIZE
    vector_difference<<<grid_size, block_size>>>(A, B, C, VECTOR_SIZE);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // for (int i = 0; i < VECTOR_SIZE; i++) {
    //     cout << to_string(A[i]) + " " + to_string(B[i]) + " " + to_string(C[i]) + "\n";
    // }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main(int, char**) {
    // unsigned int start_time =  clock();
    // task1();
    // unsigned int end_time = clock();
    // unsigned int search_time = end_time - start_time;
    // cout << "task1: " +  to_string(search_time) + "ms\n";
    
    //start_time = clock();
    //task2();
    //end_time = clock();
    //search_time = end_time - start_time;
    //cout << "task2: " + to_string(search_time) + "ms\n";
    
    // start_time = clock();
    // task3();
    // end_time = clock();
    // search_time = end_time - start_time;
    // cout << "task3: " +  to_string(search_time) + "ms\n";
}