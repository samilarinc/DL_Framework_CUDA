#include<cuda_runtime.h>
#include<stdio.h>
#include<algorithm>
#include <thrust/fill.h>

__global__ void conv3d(double *input, double *kernel, double *output, int in_h, int in_w, int ker_h, int ker_w, int stride = 1, int pad = 0){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int out_h = (in_h - ker_h + 2 * pad) / stride + 1;
    int out_w = (in_w - ker_w + 2 * pad) / stride + 1;
    if (idx < out_w && idy < out_h && idz < out_h)
    {
        int in_idx = idx * stride;
        int in_idy = idy * stride;
        int in_idz = idz * stride;
        double sum = 0;
        for (int k = 0; k < ker_h; k++)
        {
            for (int l = 0; l < ker_w; l++)
            {
                int in_idx_ = in_idx + k;
                int in_idy_ = in_idy + l;
                int in_idz_ = in_idz + k;
                if (in_idx_ < in_w && in_idy_ < in_h && in_idz_ < in_h)
                {
                    sum += input[in_idx_ + in_idy_ * in_w + in_idz_ * in_w * in_h] * kernel[k + l * ker_h];
                }
            }
        }
        output[idx + idy * out_w + idz * out_w * out_h] = sum;
    }
}

__global__ void conv2d(double *input, double *kernel, double *output, int in_h, int in_w, int ker_h, int ker_w, int stride = 1, int pad = 0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    i = i * stride + pad;
    j = j * stride + pad;
    // int out_h = (in_h - ker_h + 2 * pad) / stride + 1;
    int out_w = (in_w - ker_w + 2 * pad) / stride + 1;
    if(i < in_h && j < in_w){
        double sum = 0;
        for(int k = 0; k < ker_h; k++){
            for(int l = 0; l < ker_w; l++){
                sum += input[(i + k) * in_w + j + l] * kernel[k * ker_w + l];
            }
        }
        output[(i-pad)/stride + (j-pad)/stride * out_w] = sum;
    }
}

__global__ void matrix(double *input, double num, int h, int w)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < h*w){
        input[i] = num;
    }
}

int main()
{
    cudaError_t err;
    double *mat, *kernel, *output;
    int in_h = 4, in_w = 4;
    int kh = 2, kw = 2;
    int stride = 2, pad = 1;
    int out_h = (in_h - kh + 2 * pad) / stride + 1;
    int out_w = (in_w - kw + 2 * pad) / stride + 1;
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

    dim3 dimGrid(16, 16);
    dim3 dimBlock(num_elements / dimGrid.x, num_elements / dimGrid.y);
    for(int t = 0; t < 1; t++)
    {
        conv2d<<<dimBlock, dimGrid>>>(mat, kernel, output, in_h, in_w, kh, kw, stride, pad);
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess)printf("Failed4!\n");
    err = cudaMemcpy(temp, output, size_out, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)printf("Failed5!\n");
    
    for(int i = 0; i < out_h; i++){
        for(int j = 0; j < out_w; j++){
            printf("%f ", temp[i * out_w + j]);
        }
        printf("\n");
    }
}