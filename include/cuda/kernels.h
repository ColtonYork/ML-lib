#ifndef KERNELS_H
#define KERNELS_H

void launch_relu(float* data, float* output, int size);

void launch_add_bias(float* output, float* bias, int batch_size, int out_features);

void launch_relu_backward(float* grad_output, float* saved_input, float* grad_input, int size);

void launch_sum_rows(float* input, float* output, int batch_size, int out_features);

void launch_sgd_update(float* weights, float* grads, float lr, int size);

void launch_adam_update(float* p, float* g, float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        float bc1, float bc2, int size);
                        
void launch_momentum_update(float* p, float* g, float* v,
                            float lr, float momentum, int size);

#endif