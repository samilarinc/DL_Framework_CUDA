#ifndef REGULARIZER_H
#define REGULARIZER_H

class Regularizer{
public:
    Regularizer();
    virtual Regularizer* clone() const = 0;
    virtual ~Regularizer();
    virtual double* norm(double* weights, int size) = 0;
    virtual double* calc_gradient(double* weights, int size) = 0;

    double *dev_alpha;
    double *temp;
    double max_size;
    double alpha;
};

#endif // REGULARIZER_H