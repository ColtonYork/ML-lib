#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

class Loss {
public:
    virtual Tensor* forward(Tensor* output, Tensor* target) = 0;
    virtual Tensor* backward(Tensor* output, Tensor* target) = 0;
    virtual ~Loss() {}
};

#endif