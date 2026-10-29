#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include "../layers/layer.h"

class Optimizer {
protected:
    std::vector<Layer*>* layers;   // borrowed pointer to the network's layer list
    float lr;
public:
    Optimizer(std::vector<Layer*>* layers, float lr) : layers(layers), lr(lr) {}
    virtual void step() = 0;
    virtual ~Optimizer() {}
};

#endif

