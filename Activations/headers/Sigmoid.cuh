#ifndef SIGMOID_H
#define SIGMOID_H

#include "headers/Activation.cuh"

class Sigmoid : public Activation {
public:
    Sigmoid();
    ~Sigmoid();
    double* forward(double *input_tensor, int size) override;
    double* backward(double *error_tensor, int size) override;
    
    double *activ;
};

#endif // SIGMOID_H
