#ifndef L2_H
#define L2_H

#include "headers/Regularizer.cuh"

class L2 : public Regularizer {
public:
    L2(double alpha, int max_size = 10000);
    ~L2();
    double* norm(double* weights, int size);
    double* calc_gradient(double* weights, int size);

    double *dev_alpha;
    double *temp;
};

#endif // L2_H