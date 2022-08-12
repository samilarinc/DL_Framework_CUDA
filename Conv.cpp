#include<cuda_runtime.h>
#include<stdio.h>
#include<algorithm>
#include <thrust/fill.h>

__global__ void conv2d(double *input, double *kernel, double *output, int in_h, int in_w, int ker_h, int ker_w)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < in_h-ker_h+1 && j < in_w-ker_w+1)
    {
        double sum = 0;
        for (int k = 0; k < ker_h; k++)
        {
            for (int l = 0; l < ker_w; l++)
            {
                sum += input[(i + k) * in_w + j + l] * kernel[k * ker_w + l];
            }
        }
        output[i * in_w + j] = sum;
    }
}

__global__ void matrix(double *input, double num, int h, int w)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < h*w)
    {
        input[i] = num;
    }
}

int main()
{
    cudaError_t err;
    double *mat, *kernel, *output;
    int in_h = 5, in_w = 5;
    int kh = 3, kw = 3;
    int out_h = in_h - kh + 1;
    int out_w = in_w - kw + 1;
    int num_elements = in_h * in_w;
    int num_elements_ker = kh * kw;
    int num_elements_out = out_h * out_w;

    int size = sizeof(double) * num_elements;
    int size_ker = sizeof(double) * num_elements_ker;
    int size_out = sizeof(double) * num_elements_out;

    double *temp = (double *)malloc(size_out);

    err = cudaMalloc((void**)&mat, size);
    if(err != cudaSuccess)printf("Failed1!\n");
    err = cudaMalloc((void**)&kernel, size_ker);
    if(err != cudaSuccess)printf("Failed2!\n");
    err = cudaMalloc((void**)&output, size_out);
    if(err != cudaSuccess)printf("Failed3!\n");

    matrix<<<num_elements, 1>>>(mat, 1, in_h, in_w);
    matrix<<<num_elements_ker, 1>>>(kernel, 2, kh, kw);

    dim3 dimGrid(64, 64);
    dim3 dimBlock(1, 1);
    for(int t = 0; t < 100; t++)
    {
        conv2d<<<dimGrid, dimBlock>>>(mat, kernel, output, in_h, in_w, kh, kw);
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess)printf("Failed4!\n");
    err = cudaMemcpy(temp, output, size_out, cudaMemcpyDeviceToHost);
    for (int i = 0; i < out_h; i++)
    {
        for (int j = 0; j < out_w; j++)
        {
            printf("%f ", temp[i * out_w + j]);
        }
        printf("\n");
    }
}