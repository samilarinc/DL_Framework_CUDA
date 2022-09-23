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

SGD::SGD(double learning_rate, double momentum, Regularizer* regularizer): Optimizer(regularizer) {
    this->lr = learning_rate;
    this->momentum = momentum
    this->v = NULL;
}

SGD::SGD(SGD& sgd): Optimizer(sgd.regularizer) {
    this->lr = sgd.lr;
    this->momentum = sgd.momentum;
    if(momentum > 0){
        cudaError_t err = cudaMalloc((double **)&v, sizeof(double) * size);
        if(err != cudaSuccess)printf("Error allocating memory for v\n");
        err = cudaMemset(v, 0, sizeof(double) * size);
        if(err != cudaSuccess)printf("Error setting memory for v\n");
    }
    else{
        this->v = NULL;
    }
}

SGD::~SGD() {
    if(v != NULL)
        cudaFree(v);
}

void SGD::step(double* weights, double* gradients, int weight_size) {
    if(v == NULL && momentum > 0){
        cudaError_t err = cudaMalloc((double **)&v, sizeof(double) * weight_size);
        if(err != cudaSuccess)printf("Error allocating memory for v\n");
        err = cudaMemset(v, 0, sizeof(double) * weight_size);
        if(err != cudaSuccess)printf("Error setting memory for v\n");
    }
    if(this->regularizer != NULL){
        double* temp = regularizer->calc_gradient(weights, weight_size);
        sgd_update<<<weight_size+1, 1>>>(weights, temp, weight_size, this->lr);
    }
    if(v != NULL){
        sgd_update_momentum<<<weight_size+1, 1>>>(weights, gradients, v, weight_size, this->lr, this->momentum);
    }
    else{
        sgd_update<<<weight_size+1, 1>>>(weights, gradients, weight_size, this->lr);
    }
}