#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"
#include "tensor.h"

class Softmax : public Layer {
    Tensor* saved_output = nullptr;
public:
    Tensor* forward(Tensor* input) override;
    Tensor* backward(Tensor* grad_output) override;
};

#endif