#ifndef ACTIVATION_H
#define ACTIVATION_H

class Activation {
public:
    Activation();
    virtual ~Activation();
    virtual double* forward(double *input_tensor, int size) = 0;
    virtual double* backward(double *error_tensor, int size) = 0;
};

#endif // ACTIVATION_H