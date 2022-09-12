#ifndef LOSS_H
#define LOSS_H

class Loss{
public:
    Loss();
    virtual ~Loss();
    virtual double forward(double *label, double *pred, int size) = 0;
    virtual double* backward(double *label, int size) = 0;
};

#endif // LOSS_H