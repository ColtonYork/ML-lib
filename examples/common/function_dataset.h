#ifndef FUNCTION_DATASET_H
#define FUNCTION_DATASET_H

#include <functional>
#include "tensor.h"

// n_samples evenly spaced points of fn over [x_min, x_max], as one full batch on the GPU
struct FunctionDataset {
    Tensor* batch_input  = nullptr;
    Tensor* batch_target = nullptr;

    FunctionDataset(std::function<float(float)> fn, float x_min, float x_max, int n_samples)
    {
        int shape[] = {n_samples, 1};
        batch_input  = new Tensor(shape, 2, false);
        batch_target = new Tensor(shape, 2, false);

        for (int i = 0; i < n_samples; i++) {
            float x = x_min + (float)i / (n_samples - 1) * (x_max - x_min);
            batch_input->data[i]  = x;
            batch_target->data[i] = fn(x);
        }

        batch_input->to_gpu();
        batch_target->to_gpu();
    }

    ~FunctionDataset()
    {
        delete batch_input;
        delete batch_target;
    }
};

#endif
