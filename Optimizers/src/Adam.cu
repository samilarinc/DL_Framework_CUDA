#include "headers/Adam.cuh"
#include <math.h>
#include <stdio.h>

__global__ void adam_update(double *w, double *g, int size, double lr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        w[idx] -= lr * g[idx];
    }
}

__global__ void calc_rv(double rho, double mu, double *r, double *v, double *g, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        v[idx] = mu * v[idx] + (1 - mu) * g[idx];      
        r[idx] = rho * r[idx] + (1 - rho) * g[idx] * g[idx];
    }
}

__global__ void divConstant(double *in, double c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        in[idx] /= c;
    }
}

__global__ void divConstantNoChange(double *in, double c, int size, double *out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        out[idx] = in[idx] / c;
    }
}

__global__ void calculateAdamGrad(double *v_hat, double *r_hat, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        v_hat[idx] = v_hat[idx] / (sqrt(r_hat[idx]) + 1e-8);
    }
}

Adam::Adam(double learning_rate, double mu, double rho, Regularizer* regularizer) : Optimizer(regularizer) {
    this->lr = learning_rate;
    // this->weight_size = weight_size;
    this->k = 1;
    cudaError_t err;
    err = cudaMalloc((double **)&this->v, weight_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating v\n");
    err = cudaMalloc((double **)&this->r, weight_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating r\n");
    err = cudaMalloc((double **)&this->v_hat, weight_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating v_hat\n");
    err = cudaMalloc((double **)&this->r_hat, weight_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating r_hat\n");    

    this->mu = mu;
    this->rho = rho;
    this->regularizer = regularizer;
}

Adam::~Adam(){
    cudaFree(this->v);
    cudaFree(this->r);
    cudaFree(this->v_hat);
    cudaFree(this->r_hat);
}

void Adam::step(double* weights, double* gradients, int weight_size){
    if(this->regularizer != NULL){
        double* temp = regularizer->calc_gradient(weights, weight_size);
        adam_update<<<weight_size+1, 1>>>(weights, temp, weight_size, this->lr);
    }
    calc_rv<<<weight_size+1, 1>>>(rho, mu, this->r, this->v, gradients, weight_size);
    double temp_divider = 1 - pow(this->mu, this->k);
    divConstantNoChange<<<weight_size+1, 1>>>(this->v, temp_divider, weight_size, this->v_hat);
    temp_divider = 1 - pow(this->rho, this->k);
    divConstantNoChange<<<weight_size+1, 1>>>(this->r, temp_divider, weight_size, this->r_hat);
    this->k += 1;
    calculateAdamGrad<<<weight_size+1, 1>>>(this->v_hat, this->r_hat, weight_size);
    adam_update<<<weight_size+1, 1>>>(weights, this->v_hat, weight_size, this->lr);    
}