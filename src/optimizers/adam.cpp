#include "optimizers/adam.h"
#include "../../include/cuda/kernels.h"
#include <cmath>

Adam::Adam(std::vector<Layer*>* layers, float lr, float beta1, float beta2, float eps)
    : Optimizer(layers, lr), beta1(beta1), beta2(beta2), eps(eps), t(0)
{
    // walk every layer, allocate one m and one v tensor per parameter,
    // matching the param's shape. CPU constructor zero-inits, then we move to GPU.
    for (Layer* layer : *layers) {
        std::vector<Tensor*> ps = layer->params();
        for (Tensor* p : ps) {
            Tensor* mi = new Tensor(p->shape, p->ndim, false);
            Tensor* vi = new Tensor(p->shape, p->ndim, false);
            mi->to_gpu();
            vi->to_gpu();
            m.push_back(mi);
            v.push_back(vi);
        }
    }
}

Adam::~Adam()
{
    for (Tensor* mi : m) delete mi;
    for (Tensor* vi : v) delete vi;
}

void Adam::step()
{
    t++;

    // bias-correction denominators -- computed once per step on the CPU
    // and passed down to the kernel, since they're scalars shared by all elements.
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);

    int idx = 0;
    for (Layer* layer : *layers) {
        std::vector<Tensor*> ps = layer->params();
        std::vector<Tensor*> gs = layer->grads();
        for (size_t i = 0; i < ps.size(); i++) {
            if (gs[i] == nullptr) { idx++; continue; }
            launch_adam_update(ps[i]->data, gs[i]->data,
                               m[idx]->data, v[idx]->data,
                               lr, beta1, beta2, eps, bc1, bc2,
                               ps[i]->num_elements());
            idx++;
        }
    }
}
