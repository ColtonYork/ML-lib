#ifndef GRAD_CHECK_H
#define GRAD_CHECK_H

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "tensor.h"
#include "layers/layer.h"
#include "cuda/check.h"

// copy a tensor's values into a host vector without changing where the tensor lives
inline std::vector<float> to_host(Tensor* t)
{
    std::vector<float> host(t->num_elements());
    if (t->on_gpu)
        CUDA_CHECK(cudaMemcpy(host.data(), t->data, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    else
        host.assign(t->data, t->data + host.size());
    return host;
}

// overwrite one element of a tensor in place (GPU or CPU)
inline void set_element(Tensor* t, int i, float value)
{
    if (t->on_gpu)
        CUDA_CHECK(cudaMemcpy(t->data + i, &value, sizeof(float), cudaMemcpyHostToDevice));
    else
        t->data[i] = value;
}

// random GPU tensor with values in [-1, -0.1] or [0.1, 1].
// keeping values away from 0 stops the finite difference from stepping across ReLU's kink.
inline Tensor* random_tensor(int* shape, int ndim)
{
    Tensor* t = new Tensor(shape, ndim, false);
    for (int i = 0; i < t->num_elements(); i++) {
        float mag = 0.1f + 0.9f * ((float)rand() / RAND_MAX);
        t->data[i] = (rand() % 2) ? mag : -mag;
    }
    t->to_gpu();
    return t;
}

// the scalar we differentiate: L = sum(output * weights).
// dL/doutput is just weights, so weights is what gets passed into backward().
inline double loss_of(Layer* layer, Tensor* input, const std::vector<float>& weights)
{
    Tensor* output = layer->forward(input);
    std::vector<float> out = to_host(output);
    delete output;
    layer->free_gradients();  // layer may hold a pointer to output; clear it

    double sum = 0.0;
    for (size_t i = 0; i < out.size(); i++)
        sum += (double)out[i] * weights[i];
    return sum;
}

// central difference estimate of dL/dt for every element of t: (L(t + h) - L(t - h)) / 2h.
// t can be the input or one of the layer's params.
inline std::vector<float> numeric_grad(Layer* layer, Tensor* input, Tensor* t,
                                       const std::vector<float>& weights, float h)
{
    std::vector<float> values = to_host(t);
    std::vector<float> grad(values.size());

    for (int i = 0; i < (int)values.size(); i++) {
        set_element(t, i, values[i] + h);
        double plus = loss_of(layer, input, weights);

        set_element(t, i, values[i] - h);
        double minus = loss_of(layer, input, weights);

        set_element(t, i, values[i]);  // restore
        grad[i] = (float)((plus - minus) / (2.0 * h));
    }
    return grad;
}

// compares analytic vs numeric gradients, prints one line, returns true if they match
inline bool report(const char* name, const char* what,
                   const std::vector<float>& analytic, const std::vector<float>& numeric, float tol)
{
    if (analytic.size() != numeric.size()) {
        printf("  %-10s %-10s size mismatch (%zu vs %zu)  FAIL\n", name, what, analytic.size(), numeric.size());
        return false;
    }

    double worst = 0.0;
    for (size_t i = 0; i < analytic.size(); i++) {
        double err = fabs(analytic[i] - numeric[i]) / fmax(1.0, fabs(numeric[i]));
        worst = fmax(worst, err);
    }

    bool ok = worst <= tol;
    printf("  %-10s %-10s max err %.2e  %s\n", name, what, worst, ok ? "ok" : "FAIL");
    return ok;
}

// checks layer->backward() against finite differences for the input gradient and every param gradient.
// the layer must be ready to run (e.g. set_cublas already called for Linear).
// keep shapes tiny: this runs 2 forward passes per input element and per param element.
inline bool grad_check(const char* name, Layer* layer, int* input_shape, int ndim,
                       float h = 1e-2f, float tol = 1e-3f)
{
    Tensor* input = random_tensor(input_shape, ndim);

    // analytic: one forward + backward
    Tensor* output      = layer->forward(input);
    Tensor* grad_output = random_tensor(output->shape, output->ndim);
    std::vector<float> weights = to_host(grad_output);

    Tensor* grad_input = layer->backward(grad_output);
    std::vector<float> analytic_input = to_host(grad_input);

    std::vector<std::vector<float>> analytic_params;
    for (Tensor* g : layer->grads())
        analytic_params.push_back(g ? to_host(g) : std::vector<float>());

    delete grad_input;
    delete grad_output;
    delete output;
    layer->free_gradients();

    // numeric: nudge each element and re-run forward
    bool ok = report(name, "input", analytic_input, numeric_grad(layer, input, input, weights, h), tol);

    std::vector<Tensor*> params = layer->params();
    for (size_t p = 0; p < params.size(); p++) {
        char what[32];
        snprintf(what, sizeof(what), "param %zu", p);
        ok = report(name, what, analytic_params[p], numeric_grad(layer, input, params[p], weights, h), tol) && ok;
    }

    delete input;
    return ok;
}

#endif
