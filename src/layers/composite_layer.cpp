#include "layers/composite_layer.h"

// the network calls this once in add_layer(); pass the context down so inner Linears can matmul
void CompositeLayer::set_cublas(CublasContext& ctx)
{
    for (Layer* child : children)
        child->set_cublas(ctx);
}

// every trainable tensor of every child, in child order.
// the order must never change: Adam/Momentum match their m/v/velocity buffers to params by index.
std::vector<Tensor*> CompositeLayer::params()
{
    std::vector<Tensor*> all;
    for (Layer* child : children) {
        std::vector<Tensor*> p = child->params();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

// gradients of every child, in exactly the same order as params(), so params()[i] and grads()[i]
// always refer to the same tensor. only valid between backward() and the end of the step.
std::vector<Tensor*> CompositeLayer::grads()
{
    std::vector<Tensor*> all;
    for (Layer* child : children) {
        std::vector<Tensor*> g = child->grads();
        all.insert(all.end(), g.begin(), g.end());
    }
    return all;
}

// end-of-step cleanup (called by the network's cleanup() after the optimizer step).
// children go first so they drop their pointers to our internals before we delete them.
void CompositeLayer::free_gradients()
{
    for (Layer* child : children)
        child->free_gradients();

    for (Tensor* t : internals)
        delete t;
    internals.clear();
}

// the block owns its children, so it deletes them.
// free_gradients() first, in case the block is destroyed mid-step with internals still allocated.
CompositeLayer::~CompositeLayer()
{
    free_gradients();

    for (Layer* child : children)
        delete child;
}



