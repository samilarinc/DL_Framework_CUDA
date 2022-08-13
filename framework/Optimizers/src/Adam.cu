#include "headers/Adam.cuh"

__global__ void adam_update(double *w, double *g, int size, double lr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        w[idx] -= lr * g[idx];
    }
}

__global__ void calc_v(double mu, double *v, double *g, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        v[idx] = mu * v[idx] + (1 - mu) * g[idx];
    }
}

__global__ void calc_r(double rho, double *r, double *g, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        r[idx] = rho * r[idx] + (1 - rho) * g[idx] * g[idx];
    }
}

__global__ void divCpnstant(double *in, double c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        in[idx] /= c;
    }
}

Adam::Adam(double learning_rate, int weight_size, double mu, double rho, Regularizer* regularizer) : Optimizer(regularizer) {
    this->learning_rate = learning_rate;
    this->weight_size = weight_size;
    this->k = 1;
    cudaError_t err;
    err = cudaMalloc((double **)&this->v, weight_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating v\n");
    err = cudaMalloc((double **)&this->r, weight_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating r\n");
    this->mu = mu;
    this->rho = rho;
    this->regularizer = regularizer;
}

Adam::~Adam(){
    cudaFree(this->v);
    cudaFree(this->r);
}

void Adam::step(double* weights, double* gradients){
    if(this->regularizer != NULL){
        double* temp = regularizer->calc_gradient(weights, size);
        adam_update<<<size+1, 1>>>(weights, temp, size, this->lr);
    }
    calc_v<<<size+1, 1>>>(this->mu, this->v, gradients, size);
    calc_r<<<size+1, 1>>>(this->rho, this->r, gradients, size);
    divCpnstant<<<size+1, 1>>>(this->v, this->k, size);
}