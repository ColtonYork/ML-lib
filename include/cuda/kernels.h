#ifndef KERNELS_H
#define KERNELS_H

void launch_relu(float* data, float* output, int size);

void launch_add_bias(float* output, float* bias, int batch_size, int out_features);

void launch_relu_backward(float* grad_output, float* saved_input, float* grad_input, int size);

void launch_sum_rows(float* input, float* output, int batch_size, int out_features);

void launch_sgd_update(float* weights, float* grads, float lr, int size);


#endif