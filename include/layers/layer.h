


#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "../tensor.h"
#include "../cuda/cublas_ops.h"

class Layer {
public:
    virtual Tensor* forward(Tensor* input) = 0;

    // grad_output: gradient flowing in from the layer AHEAD (closer to loss)
    // grad_input:  gradient we compute and pass back to the layer BEHIND (closer to input)
    virtual Tensor* backward(Tensor* grad_output) = 0;
    
    virtual void free_gradients() {}
    virtual void set_cublas(CublasContext& ctx) {}
    virtual ~Layer() {}


    virtual std::vector<Tensor*> params() { return {}; }
    virtual std::vector<Tensor*> grads() { return {}; }

};

#endif
