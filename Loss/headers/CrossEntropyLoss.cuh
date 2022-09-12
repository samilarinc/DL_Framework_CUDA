#ifndef CROSSENTROPYLOSS_H
#define CROSSENTROPYLOSS_H
#include "headers/Loss.cuh"

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();
    double forward(double *label, double *pred, int size) override;
    double* backward(double *label, int size) override;

    double *last_input = NULL;
    double *loss = NULL;
};

#endif // CROSSENTROPYLOSS_H