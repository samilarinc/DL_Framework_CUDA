#ifndef DENSE_H
#define DENSE_H

#include "headers/Base.cuh"
#include "headers/Optimizer.cuh"

class Dense : public BaseLayer {
public:
    Dense(int in_size, int out_size, Optimizer* optimizer = NULL);
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
    Optimizer *w_optimizer;
    Optimizer *b_optimizer;
};

#endif