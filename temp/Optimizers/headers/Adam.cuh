#ifndef ADAM_H
#define ADAM_H

#include "headers/Optimizer.cuh"
#include "headers/Regularizer.cuh"

class Adam : public Optimizer {
public:
    Adam(double learning_rate, double mu, double rho, Regularizer* regularizer = NULL);
    virtual Adam* clone() const override { return new Adam(*this); }
    ~Adam();
    void step(double* weights, double* gradients, int weight_size) override;

    double lr;
    double mu;
    double rho;
    double *v;
    double *v_hat;
    double *r;
    double *r_hat;
    double k;
    int weight_size;
    Regularizer* regularizer;
};

#endif // ADAM_H