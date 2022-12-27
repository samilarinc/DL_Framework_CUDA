#ifndef DENSE_H
#define DENSE_H

#include "headers/Base.cuh"
#include "headers/Optimizer.cuh"
#include "Tensor.cuh"

class Dense : public BaseLayer {
public:
    Dense(int in_size, int out_size, Optimizer* optimizer);
    Dense(int in_size, int out_size) : Dense(in_size, out_size, NULL) {}
    ~Dense();
    double* forward(const double *input);
    Tensor forward(Tensor input) { return Tensor(forward(input.data), input.batch_size, input.h, input.w); }
    double* backward(double *error_tensor);
    // Tensor backward(Tensor error_tensor);
    int in_size;
    int out_size;
    int weight_size;

    double *last_input = NULL;
    double *weights = NULL;
    double *bias = NULL;
    double *output = NULL;
    double *dx = NULL;
    double *dW = NULL;
    double *db = NULL;
    Optimizer *w_optimizer = NULL;
    Optimizer *b_optimizer = NULL;
};

#endif