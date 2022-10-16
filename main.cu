#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
using namespace  std;

#define CUDA_CHECK_RETURN(value) {                                                                           \
    cudaError_t error = value;                                                                               \
                                                                                                             \
    if (error != cudaSuccess) {                                                                              \
        fprintf(stderr, "Error %s at line %d at file %s\n", cudaGetErrorString(error), __LINE__, __FILE__);  \
        exit(1);                                                                                             \
    }                                                                                                        \
}

#define VECTOR_SIZE (1000000u)
#define BLOCK_SIZE (256)
#define GRID_SIZE ((VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);


__global__ void initBVec(int *data, int length) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < length) { 
        data[i] = 1 - length;
    }
}

__global__ void computeCVect(int *vectorA, int *vectorB, int *vectorC, int length) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length) {
        vectorC[i] = vectorA[i] - vectorB[i];
    }
}


void task1() {
    int *A_data = (int*) malloc(sizeof(int) * VECTOR_SIZE);
    int *B_data = (int*) malloc(sizeof(int) * VECTOR_SIZE);
    int *C_data = (int*) malloc(sizeof(int) * VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        A_data[i] = INT_MAX;
        B_data[i] = 1 - VECTOR_SIZE;
        C_data[i] = 0;
    }

    for (int i = 0; i < VECTOR_SIZE; i++) {
        C_data[i] = A_data[i] - B_data[i];
    }

    free(A_data);
    free(B_data);
    free(C_data);
}

void task2() {
    int *A_data = (int*) malloc(sizeof(int) * VECTOR_SIZE);
    int *C_data = (int*) malloc(sizeof(int) * VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        A_data[i] = INT_MAX;
        C_data[i] = 0;
    }

    int *AD_data, *BD_data, *CD_data;
    CUDA_CHECK_RETURN(cudaMalloc(&AD_data, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMalloc(&BD_data, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMalloc(&CD_data, sizeof(int) * VECTOR_SIZE));

    CUDA_CHECK_RETURN(cudaMemcpy(AD_data, A_data, sizeof(int) * VECTOR_SIZE, cudaMemcpyHostToDevice));

    int block_size = BLOCK_SIZE;
    int grid_size = GRID_SIZE
    initBVec<<<grid_size, block_size>>>(BD_data, VECTOR_SIZE);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    computeCVect<<<grid_size, block_size>>>(AD_data, BD_data, CD_data, VECTOR_SIZE);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(C_data, CD_data, sizeof(int) * VECTOR_SIZE, cudaMemcpyDeviceToHost));

    cudaFree(AD_data);
    cudaFree(BD_data);
    cudaFree(CD_data);

    free(A_data);
    free(C_data);
}

void task3() {
    int *A_data, *B_data, *C_data;
    CUDA_CHECK_RETURN(cudaMallocManaged(&A_data, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMallocManaged(&B_data, sizeof(int) * VECTOR_SIZE));
    CUDA_CHECK_RETURN(cudaMallocManaged(&C_data, sizeof(int) * VECTOR_SIZE));

    for (int i = 0; i < VECTOR_SIZE; i++) {
        A_data[i] = INT_MAX;
        B_data[i] = 1 - VECTOR_SIZE;
        C_data[i] = 0;
    }
    int block_size = BLOCK_SIZE;
    int grid_size = GRID_SIZE
    computeCVect<<<grid_size, block_size>>>(A_data, B_data, C_data, VECTOR_SIZE);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaFree(A_data);
    cudaFree(B_data);
    cudaFree(C_data);
}

int main(int, char**) {
    unsigned int start_time =  clock();
    task1();
    unsigned int end_time = clock();
    unsigned int search_time = end_time - start_time;
    cout << search_time;
    //task2();
    //task3();
}