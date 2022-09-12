#include "headers/Sigmoid.cuh"
#include <stdio.h>
#include <math.h>

Sigmoid::Sigmoid(){
    activ = NULL;
}

Sigmoid::~Sigmoid(){
    cudaFree(activ);
}

__global__ void sig_forw(double* out, double* in, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        out[idx] = 1 / (1 + exp(-in[idx]));
    }
}

__global__ void sig_back(double* out, double* err, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        out[idx] = out[idx] * (1 - out[idx]) * err[idx];
    }
}

double* Sigmoid::forward(double *input_tensor, int size){
    if(activ == NULL){
        cudaError_t err;
        err = cudaMalloc((double **)&activ, size*sizeof(double));
        if(err != cudaSuccess) printf("Activation allocation failed!");
    }
    sig_forw<<<size+1, 1>>>(activ, input_tensor, size);

    return activ;
}

double* Sigmoid::backward(double *error_tensor, int size){
    sig_back<<<size+1, 1>>>(activ, error_tensor, size);
    
    return activ;
}