#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "headers/Regularizer.cuh"

class Optimizer{
public:
    Optimizer(Regularizer* regularizer = NULL);
    ~Optimizer();
    void set_regularizer(Regularizer* regularizer);
    
    Regularizer* regularizer;
};


#endif // OPTIMIZER_H