#include "headers/SGD.cuh"
#include <stdio.h>


__global__ void sgd_update(double *w, double *g, int size, double lr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        w[idx] -= lr * g[idx];
    }
}

__global__ void sgd_update_momentum(double *w, double *g, double *v, int size, double lr, double momentum){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        v[idx] = momentum * v[idx] + lr * g[idx];
        w[idx] -= v[idx];
    }
}

SGD::SGD(double learning_rate, int weight_size, double momentum, Regularizer* regularizer): Optimizer(regularizer) {
    this->lr = learning_rate;
    this->momentum = momentum;
    if(momentum > 0){
        cudaError_t err = cudaMalloc((double **)&v, sizeof(double) * weight_size);
        if(err != cudaSuccess)printf("Error allocating memory for v\n");
        err = cudaMemset(v, 0, sizeof(double) * weight_size);
        if(err != cudaSuccess)printf("Error setting memory for v\n");
    }
    else{
        this->v = NULL;
    }
    this->size = weight_size;
}

SGD::~SGD() {
    if(v != NULL)
        cudaFree(v);
}

void SGD::step(double* weights, double* gradients) {
    if(this->regularizer != NULL){
        double* temp = regularizer->calc_gradient(weights, this->size);
        sgd_update<<<this->size+1, 1>>>(weights, temp, this->size, this->lr);
    }
    if(v != NULL){
        sgd_update_momentum<<<this->size+1, 1>>>(weights, gradients, v, this->size, this->lr, this->momentum);
    }
    else{
        sgd_update<<<this->size+1, 1>>>(weights, gradients, this->size, this->lr);
    }
}