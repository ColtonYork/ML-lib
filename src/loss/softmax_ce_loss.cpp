#include "loss/softmax_ce_loss.h"
#include "cuda/kernels.h"

SoftmaxCELoss::~SoftmaxCELoss()
{
    delete saved_output;
}

// output is raw logits [batch, classes]; softmax is applied here, so don't put a Softmax layer before this
Tensor* SoftmaxCELoss::forward(Tensor* output, Tensor* target)
{
    if (!output->on_gpu) output->to_gpu();
    if (!target->on_gpu) target->to_gpu();

    int batch_size  = output->shape[0];
    int num_classes = output->shape[1];

    // probs from the previous step are no longer needed
    delete saved_output;
    saved_output = new Tensor(output->shape, output->ndim, true);
    launch_softmax_rows(output->data, saved_output->data, batch_size, num_classes);

    int shape[] = {1, 1};
    Tensor* loss = new Tensor(shape, 2, true);
    launch_cross_entropy_loss(saved_output->data, target->data, loss->data,
                              saved_output->num_elements(), batch_size);

    return loss;
}

Tensor* SoftmaxCELoss::backward(Tensor* output, Tensor* target)
{
    if (!target->on_gpu) target->to_gpu();

    Tensor* grad = new Tensor(saved_output->shape, saved_output->ndim, true);
    launch_softmax_ce_backward(saved_output->data, target->data, grad->data,
                               saved_output->num_elements(), saved_output->shape[0]);

    return grad;
}
