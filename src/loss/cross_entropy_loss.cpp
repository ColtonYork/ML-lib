#include "loss/cross_entropy_loss.h"
#include "cuda/kernels.h"

// expects output to already be probabilities (e.g. after a Softmax layer), shape [batch, classes]
Tensor* CrossEntropyLoss::forward(Tensor* output, Tensor* target)
{
    if (!output->on_gpu) output->to_gpu();
    if (!target->on_gpu) target->to_gpu();

    int shape[] = {1, 1};
    Tensor* loss = new Tensor(shape, 2, true);
    launch_cross_entropy_loss(output->data, target->data, loss->data,
                              output->num_elements(), output->shape[0]);

    return loss;
}

Tensor* CrossEntropyLoss::backward(Tensor* output, Tensor* target)
{
    if (!output->on_gpu) output->to_gpu();
    if (!target->on_gpu) target->to_gpu();

    Tensor* grad = new Tensor(output->shape, output->ndim, true);
    launch_cross_entropy_backward(output->data, target->data, grad->data,
                                  output->num_elements(), output->shape[0]);

    return grad;
}
