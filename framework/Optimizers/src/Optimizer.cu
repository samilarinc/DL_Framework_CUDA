#include "headers/Optimizer.cuh"

Optimizer::Optimizer(Regularizer* regularizer) {
    this->regularizer = regularizer;
}

Optimizer::~Optimizer() = default;

void Optimizer::set_regularizer(Regularizer* regularizer){
    this->regularizer = regularizer;
}

