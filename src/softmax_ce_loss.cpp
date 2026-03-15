#include "../include/softmax_ce_loss.h"
#include <cmath>

Tensor* SoftmaxCELoss::forward(Tensor* output, Tensor* target)
{
    output->to_cpu();
    target->to_cpu();

    int batch_size  = output->shape[0];
    int num_classes = output->shape[1];

    int probs_shape[] = {batch_size, num_classes};
    Tensor* probs = new Tensor(probs_shape, 2, false);

    // softmax per sample
    for (int i = 0; i < batch_size; i++) {
        float* row = &output->data[i * num_classes];

        float max_val = row[0];
        for (int j = 1; j < num_classes; j++)
            if (row[j] > max_val) max_val = row[j];

        float sum = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            probs->data[i * num_classes + j] = expf(row[j] - max_val);
            sum += probs->data[i * num_classes + j];
        }
        for (int j = 0; j < num_classes; j++)
            probs->data[i * num_classes + j] /= sum;
    }

    saved_output = probs;

    // cross entropy loss averaged over batch
    int loss_shape[] = {1, 1};
    Tensor* loss = new Tensor(loss_shape, 2, false);
    float ce = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            float p = probs->data[i * num_classes + j];
            if (p < 1e-7f) p = 1e-7f;
            ce += target->data[i * num_classes + j] * logf(p);
        }
    }
    loss->data[0] = -ce / batch_size;

    return loss;
}

Tensor* SoftmaxCELoss::backward(Tensor* output, Tensor* target)
{
    target->to_cpu();

    int batch_size  = saved_output->shape[0];
    int num_classes = saved_output->shape[1];

    int shape[] = {batch_size, num_classes};
    Tensor* grad = new Tensor(shape, 2, false);

    float scale = 1.0f / batch_size;
    for (int i = 0; i < batch_size * num_classes; i++)
        grad->data[i] = (saved_output->data[i] - target->data[i]) * scale;

    grad->to_gpu();
    return grad;
}
