#include "headers/SGD.cuh"
#include <stdio.h>


__global__ void sgd_update(double *w, double *g, int size, double lr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        w[idx] -= lr * g[idx];
    }
}

SGD::SGD(double learning_rate, Regularizer* regularizer): Optimizer(regularizer) {
        this->lr = learning_rate;
    }

void SGD::step(double* weights, double* gradients, int size) {
    if(this->regularizer != NULL){
        double* temp = regularizer->calc_gradient(weights, size);
        sgd_update<<<size+1, 1>>>(weights, temp, size, this->lr);
    }
    sgd_update<<<size+1, 1>>>(weights, gradients, size, this->lr);
}