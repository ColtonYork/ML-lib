#ifndef SOFTMAX_CE_LOSS_H
#define SOFTMAX_CE_LOSS_H

#include "loss.h"
#include "../tensor.h"

class SoftmaxCELoss : public Loss {
    Tensor* saved_output = nullptr;
public:
    Tensor* forward(Tensor* output, Tensor* target) override;
    Tensor* backward(Tensor* output, Tensor* target) override;
};

#endif
