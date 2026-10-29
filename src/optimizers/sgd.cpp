#include "optimizers/sgd.h"
#include "cuda/kernels.h"
#include "tensor.h"

void SGD::step()
{
    for (Layer* layer : *layers) {
        std::vector<Tensor*> ps = layer->params();
        std::vector<Tensor*> gs = layer->grads();
        for (size_t i = 0; i < ps.size(); i++) {
            if (gs[i] == nullptr) continue;
            launch_sgd_update(ps[i]->data, gs[i]->data, lr, ps[i]->num_elements());
        }
    }
}
