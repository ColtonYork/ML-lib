#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"
#include "../tensor.h"

class Adam : public Optimizer {
    std::vector<Tensor*> m;          // first moment (running mean of gradients), per param
    std::vector<Tensor*> v;          // second moment (running mean of squared gradients), per param
    float beta1, beta2, eps;
    int t;                           // step counter, used for bias correction
public:
    Adam(std::vector<Layer*>* layers, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    ~Adam() override;                // frees m and v tensors
    void step() override;
};

#endif
