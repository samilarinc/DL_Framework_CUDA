#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>
#include "headers/CrossEntropyLoss.cuh"

CrossEntropyLoss::CrossEntropyLoss() = default;
CrossEntropyLoss::~CrossEntropyLoss() = default;

__global__ void logLoss(double *y, double *y_hat, double *loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        loss[i] = -y[i] * log(y_hat[i] + 1e-8);
    }
}

__global__ void sumOver(double *loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        for(int j = 1; j < n; j *= 2){
            if(i % (2 * j) == 0){
                loss[i] += loss[i + j];
            }
        }
    }
}

__global__ void calculateBackward(double *label, double *pred, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        pred[i] = -(label[i] / (pred[i] + 1e-8));
    }
}

double CrossEntropyLoss::forward(double *label, double *pred, int size){
    this->last_input = pred;
    cudaError_t err;
    if(this->loss == NULL){
        err = cudaMalloc((void**)&this->loss, size * sizeof(double));
        if(err != cudaSuccess)printf("Error allocating memory for loss");
    }
    double doubleLoss;
    logLoss<<<size+1,1>>>(label, pred, this->loss, size);
    sumOver<<<size+1,1>>>(this->loss, size);
    err = cudaMemcpy(&doubleLoss, loss, sizeof(double), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)printf("Error copying loss to host");
    return -doubleLoss;
}

double* CrossEntropyLoss::backward(double *label, int size){
    calculateBackward<<<size+1,1>>>(label, this->last_input, size);
    return this->last_input;
}