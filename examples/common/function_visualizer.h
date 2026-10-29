#ifndef FUNCTION_VISUALIZER_H
#define FUNCTION_VISUALIZER_H

#include <vector>
#include <functional>
#include <algorithm>
#include <cstdio>
#include "tensor.h"
#include "neural_network.h"

// plots y_actual (cyan) and y_pred (magenta) as ASCII; overlapping points are white
inline void draw(std::vector<float>& x_vals, std::vector<float>& y_actual, std::vector<float>& y_pred,
                 int width = 80, int height = 24)
{
    float y_min = 1e9f, y_max = -1e9f;
    for (float v : y_actual) { y_min = std::min(y_min, v); y_max = std::max(y_max, v); }
    for (float v : y_pred)   { y_min = std::min(y_min, v); y_max = std::max(y_max, v); }
    float pad = (y_max - y_min) * 0.1f;
    y_min -= pad;
    y_max += pad;

    float x_min = x_vals.front();
    float x_max = x_vals.back();

    std::vector<std::vector<bool>> actual_grid(height, std::vector<bool>(width, false));
    std::vector<std::vector<bool>> pred_grid(height, std::vector<bool>(width, false));

    auto to_col = [&](float x) {
        return (int)((x - x_min) / (x_max - x_min) * (width - 1));
    };
    auto to_row = [&](float y) {
        return (int)((y - y_max) / (y_min - y_max) * (height - 1));
    };

    for (int i = 0; i < (int)x_vals.size(); i++) {
        int col = to_col(x_vals[i]);
        int row = to_row(y_actual[i]);
        if (row >= 0 && row < height && col >= 0 && col < width)
            actual_grid[row][col] = true;
    }

    for (int i = 0; i < (int)x_vals.size(); i++) {
        int col = to_col(x_vals[i]);
        int row = to_row(y_pred[i]);
        if (row >= 0 && row < height && col >= 0 && col < width)
            pred_grid[row][col] = true;
    }

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            bool a = actual_grid[r][c];
            bool p = pred_grid[r][c];
            if      (a && p) printf("\033[97m*\033[0m");
            else if (a)      printf("\033[36m*\033[0m");
            else if (p)      printf("\033[35m*\033[0m");
            else             printf(" ");
        }
        printf("\n");
    }
}

inline void visualize_network_vs_truth(NeuralNetwork& net,
                                       float x_min, float x_max,
                                       std::function<float(float)> truth_fn,
                                       int width = 80, int height = 24)
{
    int n = width * 100;

    std::vector<float> x_vals(n);
    std::vector<float> y_actual(n);
    for (int i = 0; i < n; i++) {
        x_vals[i] = x_min + (float)i / (n - 1) * (x_max - x_min);
        y_actual[i] = truth_fn(x_vals[i]);
    }

    int in_shape[] = {n, 1};
    Tensor* input = new Tensor(in_shape, 2, false);
    for (int i = 0; i < n; i++)
        input->data[i] = x_vals[i];
    input->to_gpu();

    Tensor* output = net.infer(input);

    std::vector<float> y_pred(n);
    for (int i = 0; i < n; i++)
        y_pred[i] = output->data[i];

    delete input;
    delete output;

    draw(x_vals, y_actual, y_pred, width, height);
}

#endif
