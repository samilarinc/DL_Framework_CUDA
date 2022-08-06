#include<stdio.h>
#include<cstring>
#include<cstdlib>
#define loop(type,i,a,b) for(type i=a;i<b;i++)

void T(double *a, double *b, size_t h, size_t w)
{
    loop(size_t,i,0,h)
        loop(size_t,j,0,w)
            a[j*w+i]=b[i*w+j];
}

int main()
{
    size_t h = 300;
    size_t w = 300;
    size_t N = h*w;
    double *a = (double *)malloc(N*sizeof(double));
    double *b = (double *)malloc(N*sizeof(double));
    double *c = (double *)malloc(N*sizeof(double));
    
    double *d = (double *)malloc(N*sizeof(double));
    double *e = (double *)malloc(N*sizeof(double));
    double *f = (double *)malloc(N*sizeof(double));
    if(a == NULL || b == NULL || c == NULL || d == NULL || e == NULL || f == NULL){printf("Error allocating memory\n");}
    
    loop(size_t,i,0,N){d[i] = i;e[i] = i;}

    memcpy(a, d, N*sizeof(double));
    memcpy(b, e, N*sizeof(double));

    loop(size_t,i,0,100000){
        T(a,b,h,w);
    }

    memcpy(f, a, N*sizeof(double));
    
    // loop(size_t,i,0,N){
    //     if(f[i] != i+i)printf("error!\n");
    // }
}