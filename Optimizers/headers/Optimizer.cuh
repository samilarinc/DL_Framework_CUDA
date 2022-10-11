#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "headers/Regularizer.cuh"

class Optimizer{
public:
    Optimizer(Regularizer* regularizer = NULL);
    virtual Optimizer* clone() const = 0;
    ~Optimizer();
    void set_regularizer(Regularizer* regularizer);
    virtual void step(double* weights, double* gradients, int weight_size) = 0;
    
    Regularizer* regularizer;
};


#endif // OPTIMIZER_H