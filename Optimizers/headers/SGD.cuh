#ifndef SGD_H
#define SGD_H

#include "headers/Optimizer.cuh"
#include "headers/Regularizer.cuh"

class SGD: public Optimizer {
public:
    SGD(double learning_rate, int weight_size, double momentum = 0, Regularizer* regularizer = NULL);
    SGD(SGD& sgd);
    ~SGD();
    void step(double* weights, double* gradients);

    double lr;
    double momentum;
    double *v;
    int size;
};

#endif // SGD_H