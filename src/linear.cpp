#include "../include/linear.h"
#include "../include/cuda/cublas_ops.h"
#include "../include/cuda/kernels.h"
#include <cstdlib>
#include <cstdio>

Linear::Linear(int in_features, int out_features, InitType init)
{
    int w_shape[] = {in_features, out_features};
    int b_shape[] = {1, out_features};

    weights = new Tensor(w_shape, 2, false);
    bias    = new Tensor(b_shape, 2, false);

    float scale = 0.01f;
    if (init == InitType::HE)
        scale = sqrt(2.0f / in_features);
    else if (init == InitType::XAVIER)
        scale = sqrt(6.0f / (in_features + out_features));

    int total = weights->num_elements();
    for (int i = 0; i < total; i++)
        weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;

    weights->to_gpu();
    bias->to_gpu();
}

Tensor* Linear::forward(Tensor* input)
{
    if (!input->on_gpu) input->to_gpu();

    saved_input = input;

    int out_shape[] = {input->shape[0], weights->shape[1]};
    Tensor* output = new Tensor(out_shape, 2, true);

    cublas->matmul(input, weights, output);

    launch_add_bias(output->data, bias->data, input->shape[0], weights->shape[1]);

    return output;
}

//p
Tensor* Linear::backward(Tensor* grad_output)
{
    if (!grad_output->on_gpu) grad_output->to_gpu();

    int batch = saved_input->shape[0];
    int in_features = weights->shape[0];
    int out_features = weights->shape[1];

    // grad_input = grad_output * W^T  [batch, out] * [out, in] = [batch, in]
    Tensor* grad_input = new Tensor(saved_input->shape, saved_input->ndim, true);
    cublas->matmul_nt(grad_output, weights, grad_input);

    // grad_weights = input^T * grad_output  [in, batch] * [batch, out] = [in, out]
    grad_weights = new Tensor(weights->shape, weights->ndim, true);
    cublas->matmul_tn(saved_input, grad_output, grad_weights);

    // grad_bias = sum of grad_output rows  [batch, out] -> [1, out]
    int b_shape[] = {1, out_features};
    grad_bias = new Tensor(b_shape, 2, true);
    launch_sum_rows(grad_output->data, grad_bias->data, batch, out_features);

    return grad_input;
}

void Linear::update_weights(float lr)
{
    if (grad_weights == nullptr) return;

    launch_sgd_update(weights->data, grad_weights->data, lr, weights->num_elements());
    launch_sgd_update(bias->data,    grad_bias->data,    lr, bias->num_elements());
}

void Linear::set_cublas(CublasContext& ctx)
{
    cublas = &ctx;
}

void Linear::free_gradients()
{
    delete grad_weights;
    delete grad_bias;
    grad_weights = nullptr;
    grad_bias = nullptr;
    saved_input = nullptr;
}

