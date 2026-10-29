#ifndef IMPLICIT_DATASET_H
#define IMPLICIT_DATASET_H

#include <functional>
#include <cstdlib>
#include "tensor.h"

struct ImplicitDataset {
    Tensor* batch_input  = nullptr;
    Tensor* batch_target = nullptr;

    // distance_fn(x, y) returns how far the point is from the curve
    // points closer than threshold are labeled 1, everything else 0
    ImplicitDataset(std::function<float(float, float)> distance_fn,
                    float x_min, float x_max,
                    float y_min, float y_max,
                    int n_samples, float threshold = 0.1f)
    {
        int in_shape[]  = {n_samples, 2};
        int out_shape[] = {n_samples, 1};

        batch_input  = new Tensor(in_shape,  2, false);
        batch_target = new Tensor(out_shape, 2, false);

        for (int i = 0; i < n_samples; i++) {
            float x = x_min + ((float)rand() / RAND_MAX) * (x_max - x_min);
            float y = y_min + ((float)rand() / RAND_MAX) * (y_max - y_min);

            batch_input->data[i * 2 + 0] = x;
            batch_input->data[i * 2 + 1] = y;

            float dist = distance_fn(x, y);
            batch_target->data[i] = (dist < threshold) ? 1.0f : 0.0f;
        }

        batch_input->to_gpu();
        batch_target->to_gpu();
    }

    ~ImplicitDataset()
    {
        delete batch_input;
        delete batch_target;
    }
};

#endif
