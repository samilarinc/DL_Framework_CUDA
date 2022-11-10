#include "headers/Tensor.cuh"
#include <stdio.h>

__global__ void transposeKernel(double *d_in, double *d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int in = x + y * width;
    int out = y + x * height;
    d_out[out] = d_in[in];
}

Tensor::Tensor(int h, int w) {
    this->h = h;
    this->w = w;
    cudaError_t err = cudaMalloc((void**)&data, h*w*sizeof(double));
    if(err != cudaSuccess)printf("Error allocating memory for tensor");
}

Tensor::~Tensor() {
    cudaFree(data);
}

Tensor::transpose() {
    Tensor* t = new Tensor(w, h);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((w + threadsPerBlock.x - 1) / threadsPerBlock.x, (h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transposeKernel<<<numBlocks, threadsPerBlock>>>(data, t->data, w, h);
    return t;
}