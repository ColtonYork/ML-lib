#include "../include/cross_entropy_loss.h"
#include <cmath>

Tensor* CrossEntropyLoss::forward(Tensor* output, Tensor* target)
{
    output->to_cpu();
    target->to_cpu();

    int n = output->num_elements();
    int shape[] = {1, 1};
    Tensor* loss = new Tensor(shape, 2, false);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        // clamp to avoid log(0) which is -infinity
        float p = output->data[i];
        if (p < 1e-7f) p = 1e-7f;
        sum += target->data[i] * logf(p);
    }

    loss->data[0] = -sum;
    return loss;
}

Tensor* CrossEntropyLoss::backward(Tensor* output, Tensor* target)
{
    output->to_cpu();
    target->to_cpu();

    int n = output->num_elements();
    int shape[] = {1, n};
    Tensor* grad = new Tensor(shape, 2, false);

    // dL/d(output[i]) = -target[i] / output[i]
    for (int i = 0; i < n; i++) {
        float p = output->data[i];
        if (p < 1e-7f) p = 1e-7f;
        grad->data[i] = -target->data[i] / p;
    }

    return grad;
}
