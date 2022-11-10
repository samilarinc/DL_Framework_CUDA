#ifndef SGD_H
#define SGD_H

#include "headers/Optimizer.cuh"
#include "headers/Regularizer.cuh"

class SGD: public Optimizer {
public:
    SGD(double learning_rate, double momentum = 0, Regularizer* regularizer = NULL);
    virtual SGD* clone() const override { return new SGD(*this); }
    ~SGD();
    void step(double* weights, double* gradients, int weight_size) override;

    double lr;
    double momentum;
    double *v;
    int size;
};

#endif // SGD_H