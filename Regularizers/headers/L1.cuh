#ifndef L1_H
#define L1_H

#include "headers/Regularizer.cuh"

class L1 : public Regularizer {
public:
    L1(double alpha, int max_size = 10000);
    ~L1();
    double* norm(double* weights, int size) override;
    double* calc_gradient(double* weights, int size) override;

    double *dev_alpha;
    double alpha;
    int max_size;
    double *temp;
};

#endif // L1_H