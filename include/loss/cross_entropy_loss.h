#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "loss.h"

class CrossEntropyLoss : public Loss {
public:
    Tensor* forward(Tensor* output, Tensor* target) override;
    Tensor* backward(Tensor* output, Tensor* target) override;
};

#endif
