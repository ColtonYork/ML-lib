#ifndef RELU_H
#define RELU_H

#include "layer.h"
#include "tensor.h"

class ReLU : public Layer {
    Tensor* saved_input = nullptr;
    void free_gradients() override;

public:
    Tensor* forward(Tensor* input) override;
    Tensor* backward(Tensor* grad_output) override;
};

#endif