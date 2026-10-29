#ifndef MOMENTUM_H
#define MOMENTUM_H

#include "optimizer.h"
#include "../tensor.h"

class Momentum : public Optimizer {
    std::vector<Tensor*> velocity;   // one velocity tensor per param, same shape as the param
    float momentum;                  // typically 0.9
public:
    Momentum(std::vector<Layer*>* layers, float lr, float momentum = 0.9f);
    ~Momentum() override;            // frees velocity tensors
    void step() override;
};

#endif
