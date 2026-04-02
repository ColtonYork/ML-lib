#include "loss/mse_loss.h"
#include <cmath>

Tensor* MSELoss::forward(Tensor* output, Tensor* target)
{
    output->to_cpu();
    target->to_cpu();

    int n = output->num_elements();
    int shape[] = {1, 1};
    Tensor* loss = new Tensor(shape, 2, false);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = output->data[i] - target->data[i];
        sum += diff * diff;
    }
    loss->data[0] = sum / n;

    return loss;
}

Tensor* MSELoss::backward(Tensor* output, Tensor* target)
{
    output->to_cpu();
    target->to_cpu();

    int n = output->num_elements();

    // match the shape of output exactly, not [1, n]
    Tensor* grad = new Tensor(output->shape, output->ndim, false);

    for (int i = 0; i < n; i++)
        grad->data[i] = (2.0f * (output->data[i] - target->data[i])) / n;

    return grad;
}
