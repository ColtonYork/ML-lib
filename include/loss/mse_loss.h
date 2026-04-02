#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "loss.h"

class MSELoss : public Loss {
public:
    Tensor* forward(Tensor* output, Tensor* target) override;
    Tensor* backward(Tensor* output, Tensor* target) override;
};

#endif
