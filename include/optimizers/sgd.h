#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

class SGD : public Optimizer {
public:
    SGD(std::vector<Layer*>* layers, float lr) : Optimizer(layers, lr) {}
    void step() override;
};

#endif
