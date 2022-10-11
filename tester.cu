#include<stdio.h>
#include "headers/Constant.cuh"
#include "headers/Dense.cuh"
#include "headers/Conv.cuh"
#include "headers/L1.cuh"
#include "headers/L2.cuh"
#include "headers/SGD.cuh"

#define reg_type L1 // l1, l2
#define DEBUG_DENSE

__global__ void fillMatrix(double *input, double num, int h, int w){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < h*w){
        input[i] = i;
    }
}
#ifndef DEBUG_DENSE
void RegTestNorm(double alpha = 0.5, int num_weights = 5, double init_weights = 1);
void RegTestGrad(double alpha = 0.5, int num_weights = 5, double init_weights = 1);
void SGDTest(double momentum = 0, int num_weights = 5, double init_weights = 1, double init_grad = 0.3, double lr = 0.1, double reg_alpha = 0.5);
void InitializerTest(double init_weights = 1);
#endif
void DenseTest();

int main()
{
    DenseTest();
    return 0;
}
#ifndef DEBUG_DENSE
void InitializerTest(double init_weights){
    Constant initializer(init_weights);
    int size = 20;
    double *input = new double[size];
    memset(input, 0, size*sizeof(double));
    double* dev_w = initializer.initialize(size);
    cudaMemcpy(input, dev_w, size*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; i++){
        printf("%f ", input[i]);
    }
    printf("\n");
}

void SGDTest(double momentum, int num_weights, double init_weights, double init_grad, double lr, double reg_alpha){
    L1 *reg = new L1(reg_alpha);
    SGD optimizer(lr, num_weights, 0.5, reg);
    fillMatrix<<<32,1>>>(optimizer.v, 1, num_weights, 1);
    double *weights, *gradients;
    double *host_weights = (double*)malloc(num_weights*sizeof(double));
    cudaError_t err;
    err = cudaMalloc((double **)&weights, num_weights * sizeof(double));
    printf(err == cudaSuccess ? "." : "Memory allocation failed\n");
    err = cudaMalloc((double **)&gradients, num_weights * sizeof(double));
    printf(err == cudaSuccess ? "." : "Memory allocation failed\n");
    fillMatrix<<<16,1>>>(weights, init_weights, num_weights, 1);
    fillMatrix<<<16,1>>>(gradients, init_grad, num_weights, 1);
    optimizer.step(weights, gradients);
    err = cudaMemcpy(host_weights, weights, num_weights * sizeof(double), cudaMemcpyDeviceToHost);
    printf(err == cudaSuccess ? "." : "Memory allocation failed1asd\n");
    printf("\n");
    for(int i = 0; i < num_weights; i++){
        printf("%f ", host_weights[i]);
    }
    printf("\n");
    free(host_weights);
    cudaFree(weights);
    cudaFree(gradients);
}

void RegTestNorm(double alpha, int num_weights, double init_weights){
    reg_type regularizer(alpha);
    double *pseudoWeights;
    double *norm, *host_norm;
    host_norm = (double*)malloc(sizeof(double)*1);
    cudaError_t err;
    err = cudaMalloc((double **)&pseudoWeights, sizeof(double)*num_weights);
    printf(err == cudaSuccess ? "." : "Memory allocation failed\n");
    fillMatrix<<<num_weights+1, 1>>>(pseudoWeights, init_weights, num_weights, 1);
    err = cudaMalloc((double **)&norm, sizeof(double));
    printf(err == cudaSuccess ? "." : "Memory allocation failed\n");
    norm = regularizer.norm(pseudoWeights, num_weights);
    err = cudaMemcpy(host_norm, norm, sizeof(double), cudaMemcpyDeviceToHost);
    printf(err == cudaSuccess ? "." : "Memory copy failed\n");
    printf("\n\n%f\n", *host_norm);
    free(host_norm);
    cudaFree(pseudoWeights);
    cudaFree(norm);
}

void RegTestGrad(double alpha, int num_weights, double init_weights){
    reg_type regularizer(alpha);
    double *pseudoWeights;
    double *norm, *host_norm;
    host_norm = (double*)malloc(sizeof(double)*num_weights);
    cudaError_t err;
    err = cudaMalloc((double **)&pseudoWeights, sizeof(double)*num_weights);
    printf(err == cudaSuccess ? "." : "Memory allocation failed\n");
    fillMatrix<<<num_weights+1, 1>>>(pseudoWeights, init_weights, num_weights, 1);
    err = cudaMalloc((double **)&norm, sizeof(double)*num_weights);
    printf(err == cudaSuccess ? "." : "Memory allocation failed\n");
    norm = regularizer.calc_gradient(pseudoWeights, num_weights);
    err = cudaMemcpy(host_norm, norm, sizeof(double)*num_weights, cudaMemcpyDeviceToHost);
    printf(err == cudaSuccess ? "." : "Memory copy failed\n");
    for(int i = 0; i < num_weights; i++){
        printf("\n%f", host_norm[i]);
    }
    printf("\n\n");
    free(host_norm);
    cudaFree(pseudoWeights);
    cudaFree(norm);
}
#endif
void DenseTest(){
    cudaError_t err;
    double *mat, *output;
    double *layer_output;
    double *backward_output;
    int in = 8, out = 5;
    double *temp_input = (double*) malloc(in*sizeof(double));
    
    err = cudaMalloc((void**)&mat, in*sizeof(double));
    if(err != cudaSuccess)printf("Error allocating memory for mat\n");
    output = (double*)malloc(out*sizeof(double));
    fillMatrix<<<128, 1>>>(mat, 1, in, 1);
    Dense layer(in, out);
    layer_output = layer.forward(mat);
    backward_output = layer.backward(layer_output);
    err = cudaMemcpy(output, layer_output, out*sizeof(double), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)printf("Error copying output\n");
    err = cudaMemcpy(temp_input, backward_output, in*sizeof(double), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)printf("Error copying input\n");
    for(int i = 0; i < in; i++){
        printf("%f ", temp_input[i]);
    }
    printf("\n");
    for(int i = 0; i < out; i++){
        printf("%f\n", output[i]);
    }
    free(output);
    cudaFree(mat);
}