#include "loss/mse_loss.h"
#include "cuda/kernels.h"

Tensor* MSELoss::forward(Tensor* output, Tensor* target)
{
    if (!output->on_gpu) output->to_gpu();
    if (!target->on_gpu) target->to_gpu();

    int shape[] = {1, 1};
    Tensor* loss = new Tensor(shape, 2, true);
    launch_mse_loss(output->data, target->data, loss->data, output->num_elements());

    return loss;
}

Tensor* MSELoss::backward(Tensor* output, Tensor* target)
{
    if (!output->on_gpu) output->to_gpu();
    if (!target->on_gpu) target->to_gpu();

    Tensor* grad = new Tensor(output->shape, output->ndim, true);
    launch_mse_backward(output->data, target->data, grad->data, output->num_elements());

    return grad;
}
