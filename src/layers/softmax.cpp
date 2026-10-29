#include "layers/softmax.h"
#include "cuda/kernels.h"

// softmax over the last dimension, independently for each row
Tensor* Softmax::forward(Tensor* input)
{
    if (!input->on_gpu) input->to_gpu();

    int cols = input->shape[input->ndim - 1];
    int rows = input->num_elements() / cols;

    Tensor* output = new Tensor(input->shape, input->ndim, true);
    launch_softmax_rows(input->data, output->data, rows, cols);

    saved_output = output;  // owned by the network's intermediates, not by this layer
    return output;
}

Tensor* Softmax::backward(Tensor* grad_output)
{
    if (!grad_output->on_gpu) grad_output->to_gpu();

    int cols = saved_output->shape[saved_output->ndim - 1];
    int rows = saved_output->num_elements() / cols;

    Tensor* grad_input = new Tensor(saved_output->shape, saved_output->ndim, true);
    launch_softmax_rows_backward(grad_output->data, saved_output->data, grad_input->data, rows, cols);

    return grad_input;
}

void Softmax::free_gradients()
{
    saved_output = nullptr;
}
