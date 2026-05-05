#include "cuda/kernels.h"
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;




////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void relu_kernel(float* input, float* output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

void launch_relu(float* input, float* output, int size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void add_bias_kernel(float* output, float* bias, int batch_size, int out_features)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * out_features)
    {
        int col = i % out_features;
        output[i] += bias[col];
    }
}

void launch_add_bias(float* output, float* bias, int batch_size, int out_features)
{
    int size = batch_size * out_features;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias_kernel<<<blocks, BLOCK_SIZE>>>(output, bias, batch_size, out_features);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void relu_backward_kernel(float* grad_output, float* saved_input, float* grad_input, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        grad_input[i] = saved_input[i] > 0.0f ? grad_output[i] : 0.0f;
}

void launch_relu_backward(float* grad_output, float* saved_input, float* grad_input, int size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_backward_kernel<<<blocks, BLOCK_SIZE>>>(grad_output, saved_input, grad_input, size);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


//p////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void sum_rows_kernel(float* input, float* output, int batch_size, int out_features)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < out_features)
    {
        float sum = 0.0f;
        for (int row = 0; row < batch_size; row++)
            sum += input[row * out_features + col];
        output[col] = sum;
    }
}

void launch_sum_rows(float* input, float* output, int batch_size, int out_features)
{
    int blocks = (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_rows_kernel<<<blocks, BLOCK_SIZE>>>(input, output, batch_size, out_features);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void sgd_update_kernel(float* weights, float* grad_weights, float lr, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        weights[i] -= lr * grad_weights[i];
}

void launch_sgd_update(float* weights, float* grads, float lr, int size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sgd_update_kernel<<<blocks, BLOCK_SIZE>>>(weights, grads, lr, size);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void adam_update_kernel(float* p, const float* g, float* m, float* v,
                                   float lr, float beta1, float beta2, float eps,
                                   float bc1, float bc2, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float gi = g[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;
        p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void launch_adam_update(float* p, float* g, float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        float bc1, float bc2, int size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_update_kernel<<<blocks, BLOCK_SIZE>>>(p, g, m, v, lr, beta1, beta2, eps, bc1, bc2, size);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void momentum_update_kernel(float* p, const float* g, float* v,
                                       float lr, float momentum, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        v[i] = momentum * v[i] + g[i];
        p[i] -= lr * v[i];
    }
}

void launch_momentum_update(float* p, float* g, float* v,
                            float lr, float momentum, int size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    momentum_update_kernel<<<blocks, BLOCK_SIZE>>>(p, g, v, lr, momentum, size);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////

