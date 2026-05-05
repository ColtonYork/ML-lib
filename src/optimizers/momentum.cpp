#include "optimizers/momentum.h"
#include "cuda/kernels.h"

Momentum::Momentum(std::vector<Layer*>* layers, float lr, float momentum)
    : Optimizer(layers, lr), momentum(momentum)
{
    for (Layer* layer : *layers) {
        std::vector<Tensor*> ps = layer->params();
        for (Tensor* p : ps) {
            Tensor* vel = new Tensor(p->shape, p->ndim, false);
            vel->to_gpu();
            velocity.push_back(vel);
        }
    }
}

Momentum::~Momentum()
{
    for (Tensor* vel : velocity) delete vel;
}

void Momentum::step()
{
    int idx = 0;
    for (Layer* layer : *layers) {
        std::vector<Tensor*> ps = layer->params();
        std::vector<Tensor*> gs = layer->grads();
        for (size_t i = 0; i < ps.size(); i++) {
            if (gs[i] == nullptr) { idx++; continue; }
            launch_momentum_update(ps[i]->data, gs[i]->data, velocity[idx]->data,
                                   lr, momentum, ps[i]->num_elements());
            idx++;
        }
    }
}
