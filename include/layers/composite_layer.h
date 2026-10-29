#ifndef COMPOSITE_LAYER_H
#define COMPOSITE_LAYER_H

#include <vector>
#include "layer.h"

// a layer built out of other layers (e.g. a transformer block).
// handles the bookkeeping every composite needs: passing cublas to its children, collecting their
// params/grads for the optimizer, freeing their state each step, and deleting them.
// subclasses only write the constructor, forward() and backward().
class CompositeLayer : public Layer {
protected:
    std::vector<Layer*> children;

    // tensors created inside forward() that never leave this layer (outputs of inner sublayers).
    // the network only frees the final output, so these are freed here in free_gradients().
    std::vector<Tensor*> internals;

    // registers a sublayer; it's owned by this layer and deleted in the destructor
    template <typename L>
    L* add_child(L* layer) { children.push_back(layer); return layer; }

    // marks a tensor to be freed at the end of the training step
    Tensor* keep(Tensor* t) { internals.push_back(t); return t; }

public:
    void set_cublas(CublasContext& ctx) override;
    std::vector<Tensor*> params() override;
    std::vector<Tensor*> grads() override;
    void free_gradients() override;
    ~CompositeLayer() override;
};

#endif
