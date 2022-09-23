#ifndef DENSE_H
#define DENSE_H

#include "headers/Base.cuh"

class Dense : public BaseLayer {
public:
    Dense(int in_size, int out_size);
    ~Dense();
    double* forward(double *input);
    double* backward(double *error_tensor);
    int in_size;
    int out_size;
    int weight_size;

    double *last_input;
    double *weights;
    double *bias;
    double *output;
    double *dx;
    double *dW;
    double *db;
};

#endif