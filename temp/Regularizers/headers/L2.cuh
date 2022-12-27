#ifndef L2_H
#define L2_H

#include "headers/Regularizer.cuh"

class L2 : public Regularizer {
public:
    L2(double alpha, int max_size = 10000);
    Regularizer* clone() const { return new L2(*this); }
    ~L2();
    double* norm(double* weights, int size) override;
    double* calc_gradient(double* weights, int size) override;

    double *dev_alpha;
    double alpha;
    int max_size;
    double *temp;
};

#endif // L2_H