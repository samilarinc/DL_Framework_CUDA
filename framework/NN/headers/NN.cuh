#ifndef NN_H
#define NN_H

#include "headers/Initializer.h"
#include "headers/Optimizer.h"
#include "headers/Base.cuh"

class NN{
public:
    NN(Optimizer opt, Initializer weight_init, Initializer bias_init);
    double* forward();
    void append_layer(BaseLayer* layer);
    double* test(double* input);

    Optimizer opt;
    Initializer weight_init;
    Initializer bias_init;  
    std::vector<BaseLayer*> layers;
};

#endif //NN_H