#include "headers/NN.cuh"

NN::NN(Optimizer opt, Initializer weight_init, Initializer bias_init){
    this->opt = opt;
    this->weight_init = weight_init;
    this->bias_init = bias_init;
}

double *NN::forward(){  // Regularization loss is not implemented yet
    double *output, *labels;
    output = data_layer->forward();
    labels = data_layer->labels();
    for(int i = 0; i < layers.size(); i++){
        output = layers[i]->forward(output);
    }
    double* loss = loss_layer->forward(output, labels);
    return loss;
}