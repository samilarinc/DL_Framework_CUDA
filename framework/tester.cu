#include<stdio.h>
#include "headers/Dense.cuh"

__global__ void fillMatrix(double *input, double num, int h, int w){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < h*w){
        input[i] = num;
    }
}

int main()
{
    return 0;
}