#ifndef OPTIMIZER_H
#define OPTIMIZER_H

class Optimizer {
public:
    virtual void update(float* weights, float* grads, int n) = 0;
    float lr;
};

#endif
