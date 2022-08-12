#include<cuda_runtime.h>
#include<stdio.h>

#define ceil_x_over_y(x,y) (x/y + ((x%y)?1:0))

__global__ void matrix(double *input, double num, int h, int w){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < h*w){
        input[i] = num;
    }
}

__global__ void conv2d(double *input, double *kernel, double *output, int in_h, int in_w, int ker_h, int ker_w, int stride){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    i = i * stride;
    j = j * stride;
    
    int out_w = (in_w - ker_w) / stride + 1;
    if(i < in_h && j < in_w){
        double sum = 0;
        for(int k = 0; k < ker_h; k++){
            for(int l = 0; l < ker_w; l++){
                sum += input[(i + k) * in_w + j + l] * kernel[k * ker_w + l];
            }
        }
        output[i/stride + j/stride * out_w] = sum;
    }
}

__global__ void padding(double *input, double *output, int in_h, int in_w, int left, int right){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < in_h && j < in_w){
        output[(i + left) * (in_w + left + right) + (j + left)] = input[i * in_w + j];
    }
}

class Conv{
public:
    Conv(int in_h, int in_w, int ker_h, int ker_w, int stride = 1, char pad = 'V'){
        cudaError_t err;
        this->stride = stride;
        this->pad = pad;
        this->ker_h = ker_h;
        this->ker_w = ker_w;
        this->in_h = in_h;
        this->in_w = in_w;
        double *weights, *bias;
        err = cudaMalloc((double **)&weights, sizeof(double) * ker_h * ker_w);
        if(err != cudaSuccess)printf("Weights malloc error\n");
        err = cudaMalloc((double **)&bias, sizeof(double));
        if(err != cudaSuccess)printf("Bias malloc error\n");
        matrix<<<1024, 1>>>(weights, 1, ker_h * ker_w, 1);
        matrix<<<1024, 1>>>(bias, 0.1, 1, 1);
        this->weights = weights;
        this->bias = bias;
        if(this->pad == 'S'){
            int left = (ker_w-1) / 2;
            int right = ker_w - 1 - left;
            cudaError_t err = cudaMalloc((double **)&pad_input, sizeof(double) * (in_h + ker_h - 1) * (in_w + ker_w - 1));
            if(err != cudaSuccess)printf("Pad input malloc error\n");
            this->left_up = left;
            this->right_down = right;
            this->out_w = ceil_x_over_y(in_w, stride);
            this->out_h = ceil_x_over_y(in_h, stride);
        }
        else if(this->pad == 'V'){
            cudaError_t err = cudaMalloc((double **)&pad_input, sizeof(double) * in_h * in_w);
            this->left_up = 0;
            this->right_down = 0;
            this->out_w = (in_w - ker_w) / stride + 1;
            this->out_h = (in_h - ker_h) / stride + 1;
        }
        double *output;
        err = cudaMalloc((double **)&output, sizeof(double) * out_h * out_w);
        if(err != cudaSuccess)printf("Output malloc error\n");
        this->output = output;
    }

    ~Conv(){
        cudaFree(weights);
        cudaFree(bias);
        cudaFree(pad_input);
        cudaFree(output);
    }

    double* forward(double *input){
        int num_elements = (this->in_h + this->ker_h - 1) * (this->in_w + this->ker_w - 1);
        dim3 dimGrid(16, 16);
        dim3 dimBlock(num_elements / dimGrid.x, num_elements / dimGrid.y);
        matrix<<<1024, 1>>>(this->pad_input, 0, this->in_h + this->ker_h - 1, this->in_w + this->ker_w - 1);
        padding<<<dimBlock, dimGrid>>>(input, this->pad_input, this->in_h, this->in_w, this->left_up, this->right_down);
        
        
        conv2d<<<dimBlock, dimGrid>>>(this->pad_input, this->weights, this->output, in_h + ker_h - 1, in_w + ker_w - 1,
                                        this->ker_h, this->ker_w, this->stride);
        return this->output;
    }

protected:
    double *weights;
    double *bias;
    double *pad_input;
    double *output;
    int stride;
    char pad;
    int ker_h;
    int ker_w;
    int in_h;
    int in_w;
    int left_up;
    int right_down;
    int out_w;
    int out_h;
};


int main()
{
    cudaError_t err;
    double *mat, *kernel;// *output, *padded;
    int in_h = 4, in_w = 4;
    int ker_h = 2, ker_w = 2;
    Conv layer(in_h, in_w, ker_h, ker_w, 2, 'S');
    err = cudaMalloc((double **)&mat, sizeof(double) * in_h * in_w);
    if(err != cudaSuccess)printf("Input malloc error\n");
    err = cudaMalloc((double **)&kernel, sizeof(double) * ker_h * ker_w);
    if(err != cudaSuccess)printf("Kernel malloc error\n");

    matrix<<<1024, 1>>>(mat, 2, in_h * in_w, 1);
    matrix<<<1024, 1>>>(kernel, 1, ker_h * ker_w, 1);
    layer.forward(mat);

    return 0;
}