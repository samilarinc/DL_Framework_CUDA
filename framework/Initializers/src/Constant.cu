#include "headers/Constant.cuh"
#include <stdio.h>

__global__ void fillwithcons(double *in, double c, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size)
        in[idx] = c;
}

Constant::Constant(double constant){
    this->c = constant;
}

double* Constant::initialize(int size){
    double *temp;
    cudaError_t err = cudaMalloc((double **)&temp, sizeof(double) * size);
    if(err != cudaSuccess)printf("Error allocating memory to initialize.\n");
    fillwithcons<<<size+1,1>>>(temp, this->c, size);
    return temp;
}