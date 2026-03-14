#include "relu.h"
#include "cuda/kernels.h"


Tensor* ReLU::forward(Tensor* input)
{
    if (!input->on_gpu) input->to_gpu();

    saved_input = input;  // save reference before relu zeroes values

    Tensor* output = new Tensor(input->shape, input->ndim, true);
    launch_relu(input->data, output->data, input->num_elements());

    return output;
}

Tensor* ReLU::backward(Tensor* grad_output)
{
    if (!grad_output->on_gpu) grad_output->to_gpu();

    Tensor* grad_input = new Tensor(saved_input->shape, saved_input->ndim, true);

    launch_relu_backward(grad_output->data, saved_input->data, grad_input->data, grad_output->num_elements());
    
    return grad_input;
}

void ReLU::free_gradients()
{
    saved_input = nullptr;
}

