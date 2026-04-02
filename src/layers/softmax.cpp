#include "layers/softmax.h"
#include <cmath>

Tensor* Softmax::forward(Tensor* input)
{
    input->to_cpu();

    int n = input->num_elements();
    int shape[] = {input->shape[0], input->shape[1]};
    Tensor* output = new Tensor(shape, 2, false);

    // find max for numerical stability
    float max_val = input->data[0];
    for (int i = 1; i < n; i++)
        if (input->data[i] > max_val)
            max_val = input->data[i];

    // e^x for each element
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output->data[i] = expf(input->data[i] - max_val);
        sum += output->data[i];
    }

    // divide by sum
    for (int i = 0; i < n; i++)
        output->data[i] /= sum;

    saved_output = output;
    return output;
}

Tensor* Softmax::backward(Tensor* grad_output)
{
    grad_output->to_cpu();

    int n = saved_output->num_elements();
    int shape[] = {saved_output->shape[0], saved_output->shape[1]};
    Tensor* grad_input = new Tensor(shape, 2, false);

    // dL/dx[i] = s[i] * (grad[i] - sum(grad[j] * s[j]))
    float dot = 0.0f;
    for (int i = 0; i < n; i++)
        dot += grad_output->data[i] * saved_output->data[i];

    for (int i = 0; i < n; i++)
        grad_input->data[i] = saved_output->data[i] * (grad_output->data[i] - dot);

    return grad_input;
}
