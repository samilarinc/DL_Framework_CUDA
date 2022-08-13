#ifndef DENSE_H
#define DENSE_H

#include "headers/Base.cuh"

class Dense : public BaseLayer {
public:
    Dense(int in_size, int out_size);
    ~Dense();
    double* forward(double *input);
    int in_size;
    int out_size;
    double *weights;
    double *bias;
    double *output;
};

#endif