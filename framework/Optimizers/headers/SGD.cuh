#ifndef SGD_H
#define SGD_H

#include "headers/Optimizer.cuh"
#include "headers/Regularizer.cuh"

class SGD: public Optimizer {
public:
    SGD(double learning_rate, Regularizer* regularizer = NULL);
    void step(double* weights, double* gradients, int size);

    double lr;
};

#endif // SGD_H