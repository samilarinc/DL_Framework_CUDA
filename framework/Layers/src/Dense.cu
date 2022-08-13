#include "headers/Dense.cuh"
#include <stdio.h>

__global__ void initialize(double *A, int N, double constant) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = constant;
    }
}

__global__ void dense_forward(double* input, double* weight, double* bias, double* out, int in_h, int in_w, int w_h){ // N is the number of columns of A
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < in_h && col < w_h){
        double sum = 0;
        for(int k=0; k < in_w; k++) {
            sum += input[row*in_w+k] * weight[k*w_h+col];
        }
        out[row*in_w+col] = sum + bias[col];
    }
}

Dense::Dense(int in_size, int out_size) : BaseLayer(true){
    this->in_size = in_size;
    this->out_size = out_size;
    cudaError_t err;
    err = cudaMalloc((double **)&this->weights, in_size * out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error allocating memory for weights\n");
    initialize<<<128, 1>>>(this->weights, in_size * out_size, 1);
    err = cudaMalloc((double **)&this->bias, out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error allocating memory for biases\n");
    initialize<<<128, 1>>>(this->bias, out_size, 0.1);
    err = cudaMalloc((double **)&this->output, out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error allocating memory for output\n");
    err = cudaMemset(this->output, 0, out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error setting output to 0\n");
}

Dense::~Dense() {
    cudaFree(this->weights);
    cudaFree(this->bias);
    cudaFree(this->output);
}

double* Dense::forward(double *input){
    dim3 dimGrid(1, 512); //change 1 to batch size
    dim3 dimBlock(1, 512);
    dense_forward<<<dimBlock, dimGrid>>>(input, this->weights, this->bias, this->output, 1, this->in_size, this->out_size); // 1 is the number of rows of A, might depend on the batch size
    return this->output;
}