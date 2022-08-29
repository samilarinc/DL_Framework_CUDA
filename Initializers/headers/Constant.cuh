#ifndef CONSTANT_H
#define CONSTANT_H

class Constant : public Initializer{
public:
    Constant(double constant);
    double* initialize(int size);

    double c;
};

#endif // CONSTANT_H