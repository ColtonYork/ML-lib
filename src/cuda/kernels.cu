#include "cuda/kernels.h"
#include "cuda/check.h"
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
    CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////////////
// block-wide reductions. every thread in the block must call these, and blockDim.x must be BLOCK_SIZE.
__device__ float block_reduce_sum(float val)
{
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;

    shared[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float result = shared[0];
    __syncthreads();  // so the next reduction can safely reuse shared
    return result;
}

__device__ float block_reduce_max(float val)
{
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;

    shared[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    float result = shared[0];
    __syncthreads();
    return result;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// one block per row
__global__ void softmax_rows_kernel(float* input, float* output, int cols)
{
    float* x = input  + blockIdx.x * cols;
    float* y = output + blockIdx.x * cols;

    // subtract the row max for numerical stability
    float max_val = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        max_val = fmaxf(max_val, x[j]);
    max_val = block_reduce_max(max_val);

    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = expf(x[j] - max_val);
        y[j] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum);

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        y[j] /= sum;
}

void launch_softmax_rows(float* input, float* output, int rows, int cols)
{
    softmax_rows_kernel<<<rows, BLOCK_SIZE>>>(input, output, cols);
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// one block per row: dL/dx[j] = s[j] * (grad[j] - sum_k(grad[k] * s[k]))
__global__ void softmax_rows_backward_kernel(float* grad_output, float* saved_output, float* grad_input, int cols)
{
    float* g  = grad_output  + blockIdx.x * cols;
    float* s  = saved_output + blockIdx.x * cols;
    float* gi = grad_input   + blockIdx.x * cols;

    float dot = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        dot += g[j] * s[j];
    dot = block_reduce_sum(dot);

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        gi[j] = s[j] * (g[j] - dot);
}

void launch_softmax_rows_backward(float* grad_output, float* saved_output, float* grad_input,
                                  int rows, int cols)
{
    softmax_rows_backward_kernel<<<rows, BLOCK_SIZE>>>(grad_output, saved_output, grad_input, cols);
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// loss = mean((output - target)^2). each block reduces its chunk and atomically adds into loss.
__global__ void mse_loss_kernel(float* output, float* target, float* loss, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < size) {
        float diff = output[i] - target[i];
        val = diff * diff;
    }

    float block_sum = block_reduce_sum(val);
    if (threadIdx.x == 0)
        atomicAdd(loss, block_sum / size);
}

void launch_mse_loss(float* output, float* target, float* loss, int size)
{
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mse_loss_kernel<<<blocks, BLOCK_SIZE>>>(output, target, loss, size);
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void mse_backward_kernel(float* output, float* target, float* grad, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        grad[i] = 2.0f * (output[i] - target[i]) / size;
}

void launch_mse_backward(float* output, float* target, float* grad, int size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mse_backward_kernel<<<blocks, BLOCK_SIZE>>>(output, target, grad, size);
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// loss = -sum(target * log(probs)) / batch_size, with probs clamped to avoid log(0)
__global__ void cross_entropy_loss_kernel(float* probs, float* target, float* loss, int size, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < size)
        val = -target[i] * logf(fmaxf(probs[i], 1e-7f));

    float block_sum = block_reduce_sum(val);
    if (threadIdx.x == 0)
        atomicAdd(loss, block_sum / batch_size);
}

void launch_cross_entropy_loss(float* probs, float* target, float* loss, int size, int batch_size)
{
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_entropy_loss_kernel<<<blocks, BLOCK_SIZE>>>(probs, target, loss, size, batch_size);
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dL/dp = -target / p / batch_size
__global__ void cross_entropy_backward_kernel(float* probs, float* target, float* grad, int size, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        grad[i] = -target[i] / fmaxf(probs[i], 1e-7f) / batch_size;
}

void launch_cross_entropy_backward(float* probs, float* target, float* grad, int size, int batch_size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_entropy_backward_kernel<<<blocks, BLOCK_SIZE>>>(probs, target, grad, size, batch_size);
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// softmax + cross entropy combined: dL/dlogits = (probs - target) / batch_size
__global__ void softmax_ce_backward_kernel(float* probs, float* target, float* grad, int size, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        grad[i] = (probs[i] - target[i]) / batch_size;
}

void launch_softmax_ce_backward(float* probs, float* target, float* grad, int size, int batch_size)
{
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_ce_backward_kernel<<<blocks, BLOCK_SIZE>>>(probs, target, grad, size, batch_size);
    CUDA_CHECK_LAUNCH();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
