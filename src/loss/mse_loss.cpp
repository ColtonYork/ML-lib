#include "loss/mse_loss.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

static void copy_to_host(Tensor* t, std::vector<float>& host)
{
    int total = t->num_elements();
    host.resize(total);
    if (t->on_gpu)
        cudaMemcpy(host.data(), t->data, total * sizeof(float), cudaMemcpyDeviceToHost);
    else
        memcpy(host.data(), t->data, total * sizeof(float));
}

Tensor* MSELoss::forward(Tensor* output, Tensor* target)
{
    int n = output->num_elements();

    std::vector<float> out_host;
    std::vector<float> tgt_host;
    copy_to_host(output, out_host);
    copy_to_host(target, tgt_host);

    int shape[] = {1, 1};
    Tensor* loss = new Tensor(shape, 2, false);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = out_host[i] - tgt_host[i];
        sum += diff * diff;
    }
    loss->data[0] = sum / n;

    return loss;
}

Tensor* MSELoss::backward(Tensor* output, Tensor* target)
{
    int n = output->num_elements();

    std::vector<float> out_host;
    std::vector<float> tgt_host;
    copy_to_host(output, out_host);
    copy_to_host(target, tgt_host);

    Tensor* grad = new Tensor(output->shape, output->ndim, false);

    for (int i = 0; i < n; i++)
        grad->data[i] = (2.0f * (out_host[i] - tgt_host[i])) / n;

    grad->to_gpu();   // hand off to the next backward step on GPU
    return grad;
}














// #include "loss/mse_loss.h"
// #include <cmath>

// Tensor* MSELoss::forward(Tensor* output, Tensor* target)
// {
//     output->to_cpu();
//     target->to_cpu();

//     int n = output->num_elements();
//     int shape[] = {1, 1};
//     Tensor* loss = new Tensor(shape, 2, false);

//     float sum = 0.0f;
//     for (int i = 0; i < n; i++) {
//         float diff = output->data[i] - target->data[i];
//         sum += diff * diff;
//     }
//     loss->data[0] = sum / n;

//     return loss;
// }

// Tensor* MSELoss::backward(Tensor* output, Tensor* target)
// {
//     output->to_cpu();
//     target->to_cpu();

//     int n = output->num_elements();

//     // match the shape of output exactly, not [1, n]
//     Tensor* grad = new Tensor(output->shape, output->ndim, false);

//     for (int i = 0; i < n; i++)
//         grad->data[i] = (2.0f * (output->data[i] - target->data[i])) / n;

//     return grad;
// }
