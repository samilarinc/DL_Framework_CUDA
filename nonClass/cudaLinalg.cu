#include<stdio.h>
#include<cuda_runtime.h>
#define loop(type,i,a,b) for(type i=a;i<b;i++)

__global__ void T(double *a, double *b, size_t row, size_t col)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i < row && j < col){
        a[j*col+i] = b[i*col+j];
    }
}

__global__ void add(double *a, double *b, double *c, size_t N)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N){
        c[i] = a[i] + b[i];
    }
}

__global__ void sub(double *a, double *b, double *c, size_t N)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N){
        c[i] = a[i] - b[i];
    }
}

__global__ void mul(double *a, double *b, double *c, size_t N)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N){
        c[i] = a[i] * b[i];
    }
}

__global__ void div(double *a, double *b, double *c, size_t N)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N){
        c[i] = a[i] / b[i];
    }
}

__global__ void matmul(double *a, double *b, double *c, size_t row, size_t col)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i < row && j < col){
        c[j*col+i] = 0;
        loop(size_t,k,0,col){
            c[j*col+i] += a[j*col+k] * b[k*col+i];
        }
    }
}

__global__ void norm(double* a, double* b, size_t N) // b is a double pointer not an array! Set b=0 before using this function; Returns the square of norm!
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N){
        *b += a[i]*a[i];
    }
}

__device__ void det(double *a, double *determinant, size_t N) // pass N as NxN matrix, *det 1xN vector
{
    if(N == 1){
        *determinant = a[0];
    }
    else if(N == 2){
        *determinant = a[N*(N-1)-2]*a[N*N-1] - a[N*(N-1)-1]*a[N*N-2];
    }
    else{
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i<N){
            double *temp;
            cudaMalloc((double**)&temp, sizeof(double)*(N-1)*(N-1));
            size_t j = 0;
            for(int k = N; k < N*N; k++){
                if(k%N == i){
                    j++;
                }
                else{
                    temp[k-N-j] = a[k];
                }
            }
            double tempdet;
            det(temp, &tempdet, N-1);
            int sign = i%2==0?1:-1;
            determinant[i] = sign * a[i] * (tempdet);
            cudaFree(temp);
        }
    }
}

__global__ void callDet(double *a, double *determinant, size_t N)
{
    det(a, determinant, N);
}

int main()
{
    cudaError_t cudaStatus = cudaSuccess;
    size_t h = 3;
    size_t w = 3;
    size_t N = h*w;
    double *dev_a = NULL;
    double *dev_det = NULL;

    cudaStatus = cudaMalloc((double **)&dev_a, N*sizeof(double));
    if(cudaStatus != cudaSuccess){printf("cudaMalloc failed!\n");}
    cudaStatus = cudaMalloc((double **)&dev_det, h*sizeof(double));
    if(cudaStatus != cudaSuccess){printf("cudaMalloc failed!\n");}
    double *host_a = new double[N] {1,2,3,4,5,6,7,8,10};
    double *host_det = new double[h] {0,0,0};

    cudaStatus = cudaMemcpy(dev_a, host_a, N*sizeof(double), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess){printf("cudaMemcpy failed!\n");}
    cudaStatus = cudaMemcpy(dev_det, host_det, h*sizeof(double), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess){printf("cudaMemcpy failed!\n");}
    
    cudaStatus = cudaMemcpy(host_a, dev_a, N*sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess){printf("cudaMemcpy failed!\n");}

    for(int i = 0; i < 1000; i++){
        callDet<<<1, N*N>>>(dev_a, dev_det, h);
    }
    cudaStatus = cudaMemcpy(host_det, dev_det, h*sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess){printf("cudaMemcpy failed!\n");}
    double determinant = host_det[0] + host_det[1] + host_det[2];
    cudaFree(dev_a);
    cudaFree(dev_det);
    delete[] host_a;
    delete[] host_det;

    printf("%f\n", determinant);
    cudaDeviceReset();
}


















    // cudaStatus = cudaMalloc((double **)&c, N*sizeof(double));
    // if(cudaStatus != cudaSuccess){printf("cudaMalloc failed!\n");}
    // double *d = (double *)malloc(N*sizeof(double));
    // double *e = (double *)malloc(N*sizeof(double));
    // double *f = (double *)malloc(N*sizeof(double));
    // if(d == NULL || e == NULL || f == NULL){printf("malloc failed!\n");}
    
    // loop(size_t,i,0,N){d[i] = i;e[i] = i;}

    // cudaStatus = cudaMemcpy(a, d, N*sizeof(double), cudaMemcpyHostToDevice);
    // if(cudaStatus != cudaSuccess){printf("cudaMemcpy failed!\n");}
    // cudaStatus = cudaMemcpy(b, e, N*sizeof(double), cudaMemcpyHostToDevice);
    // if(cudaStatus != cudaSuccess){printf("cudaMemcpy failed!\n");}

    // dim3 grid(h, w, 1);
    // dim3 threads(h, w, 1);

    // det<<<grid, threads>>>();
    // cudaStatus = cudaMemcpy(f, c, N*sizeof(double), cudaMemcpyDeviceToHost);

    // // loop(size_t,i,0,1){
    // //     T<<<grid, threads>>>(a,b,h,w);
    // // }

    // // cudaStatus = cudaMemcpy(f, a, N*sizeof(double), cudaMemcpyDeviceToHost);
    // // if(cudaStatus != cudaSuccess){printf("cudaMemcpy failed!\n");}
    
    // loop(size_t,i,0,h){
    //     loop(size_t,j,0,w){
    //         printf("%f ", f[i*w+j]);
    //     }
    //     printf("\n");
    // }
// }