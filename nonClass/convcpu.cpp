#include<stdio.h>
#include<cstdlib>

int main()
{
    int in_h = 500, in_w = 500;
    int kernel_h = 100, kernel_w = 100;
    int out_h = in_h - kernel_h + 1;
    int out_w = in_w - kernel_w + 1;

    double *a = (double *)malloc(in_h*in_w*sizeof(double));
    double *b = (double *)malloc(kernel_h*kernel_w*sizeof(double));
    double *c = (double *)malloc(out_h*out_w*sizeof(double));

    for(int i = 0; i < in_h*in_w; i++)
        a[i] = 1;
    
    for(int i = 0; i < kernel_h*kernel_w; i++)
        b[i] = 2;

    for(int t = 0; t < 1; t++)
    for(int i = 0; i < in_h-kernel_h+1; i++){
        for(int j = 0; j < in_w-kernel_w+1; j++){
            double sum = 0;
            for(int k = 0; k < kernel_h; k++){
                for(int l = 0; l < kernel_w; l++){
                    sum += a[(i+k)*in_w+(j+l)] * b[k*kernel_w+l];
                }
            }
            c[i*out_w+j] = sum;
        }
    }

    // for(int i = 0; i < out_h; i++){
    //     for(int j = 0; j < out_w; j++){
    //         printf("%f ", c[i*out_w+j]);
    //     }
    //     printf("\n");
    // }
}