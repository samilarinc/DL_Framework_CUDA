#ifndef ADAM_H
#define ADAM_H

#include "headers/Optimizer.h"
#include "headers/Regularizer.cuh"

class Adam : public Optimizer {
public:
    Adam(double learning_rate, int weight_size, double mu, double rho, Regularizer* regularizer = NULL);
    ~Adam();
    void step(double* weights, double* gradients);

    double lr;
    double mu;
    double rho;
    double *v;
    double *r;
    double k;
    int weight_size;
    Regularizer* regularizer;
};

#endif // ADAM_H