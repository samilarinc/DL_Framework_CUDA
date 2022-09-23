#include "headers/Dense.cuh"
#include <stdio.h>

__global__ void initialize(double *A, int N, double constant) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = constant;
    }
}

__global__ void dense_forward(double* input, double* weight, double* bias, double* out, int in_h, int in_w, int w_w){ // N is the number of columns of A
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < in_h && col < w_w){
        double sum = 0;
        for(int k = 0; k < in_w; k++) {
            sum += input[row*in_w+k] * weight[k*w_w+col];
        }
        out[row*w_w+col] = sum + bias[col];
    }
}

__global__ void dot_T_first(double *error, double *weight, double *out, int in_h, int in_w, int w_h, int w_w){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < in_w && col < w_w){
        double sum = 0;
        for(int k = 0; k < in_h; k++) {
            sum += error[row*in_h + k] * weight[col*w_h + k];
        }
        out[row*w_w+col] = sum;
    }
}

__global__ void dot_T_sec(double *error, double *weight, double *out, int in_h, int in_w, int w_h, int w_w){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < in_h && col < w_h){
        double sum = 0;
        for(int k = 0; k < in_w; k++) {
            sum += error[row*in_w+k] * weight[col*w_w+k];
        }
        out[row*w_h+col] = sum;
    }
}

__global__ void sum_bias(double *error, double *db, int in_h, int in_w){ // CAN BE OPTIMIZED!!!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < in_h){
        double sum = 0;
        for(int i = 0; i < in_w; i++){
            sum += error[idx*in_w+i];
        }
        db[idx] = sum;
    }
}

Dense::Dense(int in_size, int out_size, Optimizer *optimizer) : BaseLayer(true){
    this->in_size = in_size;
    this->out_size = out_size;
    this->weight_size = in_size * out_size;
    this->w_optimizer = optimizer;
    Optimizer *temp_opt = new SGD(optimizer);
    Regularizer *temp_reg = new L2(optimizer->regularizer->alpha, optimizer->regularizer->max_size);
    temp_opt->set_regularizer(temp_reg);
    this->b_optimizer = temp_opt;
    cudaError_t err;
    err = cudaMalloc((double **)&this->weights, in_size * out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error allocating memory for weights\n");
    initialize<<<128, 1>>>(this->weights, in_size, 1);
    err = cudaMalloc((double **)&this->bias, out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error allocating memory for biases\n");
    initialize<<<128, 1>>>(this->bias, out_size, 0.1);
    err = cudaMalloc((double **)&this->output, out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error allocating memory for output\n");
    err = cudaMemset(this->output, 0, out_size * sizeof(double));
    if (err != cudaSuccess)printf("Error setting output to 0\n");
    err = cudaMalloc((double **)&last_input, in_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating memory for last input\n");
    err = cudaMalloc((double **)&dW, in_size * out_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating memory for dw\n");
    err = cudaMalloc((double **)&dx, in_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating memory for dx\n");
    err = cudaMalloc((double **)&db, out_size * sizeof(double));
    if(err != cudaSuccess)printf("Error allocating memory for db\n");
}

Dense::~Dense() {
    cudaFree(this->weights);
    cudaFree(this->bias);
    cudaFree(this->output);
    cudaFree(this->last_input);
    cudaFree(this->dW);
    cudaFree(this->dx);
    cudaFree(this->db);
}

double* Dense::forward(double *input){
    dim3 dimGrid(32, 32); //change 1 to batch size
    dim3 dimBlock(32, 32);
    cudaError_t err = cudaMemcpy(this->last_input, input, in_size * sizeof(double), cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess)printf("Error copying input to last input\n");
    dense_forward<<<dimBlock, dimGrid>>>(input, this->weights, this->bias, this->output, 1, this->in_size, this->out_size); // 1 is the number of rows of A, might depend on the batch size
    return this->output;
}

double* Dense::backward(double *error_tensor){
    dim3 dimGrid(32, 32); 
    dim3 dimBlock(32, 32);
    dot_T_sec<<<dimGrid, dimBlock>>>(error_tensor, this->weights, this->dx, 1, this->out_size, this->in_size, this->out_size);
    dot_T_first<<<dimGrid, dimBlock>>>(this->last_input, error_tensor, this->dW, 1, this->in_size, 1, this->out_size);
    sum_bias<<<dimGrid, dimBlock>>>(error_tensor, this->db, 1, this->out_size);
    if(w_optimizer != NULL){
        w_optimizer->step(this->weights, this->dW, this->weight_size);
        b_optimizer->step(this->bias, this->db, this->out_size);
    }
    return this->dx;
}