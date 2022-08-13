#include "headers/Base.cuh"

BaseLayer::BaseLayer(bool trainable){
    this->trainable = trainable;
    this->test_mode = false;
}