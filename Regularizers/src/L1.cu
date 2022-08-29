#include "headers/L1.cuh"
#include<stdio.h>

#define dabs(x) ((x) < 0.0 ? -(x) : (x))
#define sign(x) ((x) > 0 ? 1 : -1)
#ifndef div2ceil
#define div2ceil(x) (((x) + 1) / 2)
#endif

__global__ void L1_norm(double* w, int size, double* temp){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        temp[idx] = dabs(w[idx]);
    }
}

__global__ void grad(double* w, int size, double* temp, double* alpha){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        temp[idx] = sign(w[idx]) * alpha[0];
    }
}

__global__ void sum_temp(double* temp, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < ((int)size/2)){
        temp[idx] = temp[idx] + temp[size-idx-1];
    }
}

__global__ void mul_alpha(double* temp, double* alpha){
    temp[0] = temp[0] * alpha[0];
}

L1::L1(double alpha, int max_size){
    cudaError_t err = cudaMalloc((void**)&temp, sizeof(double) * max_size);
    if(err != cudaSuccess)printf("Error allocating temp memory\n");
    err = cudaMalloc((double **)&dev_alpha, sizeof(double));
    if(err != cudaSuccess)printf("Error allocating alpha memory\n");
    err = cudaMemcpy(dev_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)printf("Error copying alpha\n");
    err = cudaMemset(temp, 0, sizeof(double) * max_size);
    if(err != cudaSuccess)printf("Error setting temp memory\n");
}

L1::~L1(){
    cudaFree(temp);
    cudaFree(dev_alpha);
}

double* L1::norm(double* weights, int size){
    L1_norm<<<size+1, 1>>>(weights, size, this->temp);
    while(size > 1){
        sum_temp<<<size+1, 1>>>(this->temp, size);
        size = div2ceil(size);
    }
    mul_alpha<<<1, 1>>>(this->temp, dev_alpha);
    return this->temp;                            /// Only the first element is used
}

double* L1::calc_gradient(double* weights, int size){
    grad<<<size+1, 1>>>(weights, size, this->temp, dev_alpha);
    return this->temp;
}