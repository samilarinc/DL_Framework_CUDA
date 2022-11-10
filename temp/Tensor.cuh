#ifndef TENSOR_H
#define TENSOR_H

class Tensor
{
public:
    Tensor();
    Tensor(int h, int w);
    ~Tensor();
    void transpose();

    int h;
    int w;
    double *data;
};

#endif // TENSOR_H
