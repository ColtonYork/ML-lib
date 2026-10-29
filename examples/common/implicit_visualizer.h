#ifndef IMPLICIT_VISUALIZER_H
#define IMPLICIT_VISUALIZER_H

#include <cstdio>
#include "tensor.h"
#include "neural_network.h"

// evaluates net on a width x height grid of (x, y) points and prints it as an ASCII heatmap
inline void visualize_implicit(NeuralNetwork& net,
                               float x_min, float x_max,
                               float y_min, float y_max,
                               int width = 80, int height = 40)
{
    int n = width * height;

    int in_shape[] = {n, 2};
    Tensor* input = new Tensor(in_shape, 2, false);

    // fill grid of (x, y) points row by row
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int i = r * width + c;
            float x = x_min + (float)c / (width  - 1) * (x_max - x_min);
            float y = y_max  - (float)r / (height - 1) * (y_max - y_min); // flip y
            input->data[i * 2 + 0] = x;
            input->data[i * 2 + 1] = y;
        }
    }

    input->to_gpu();
    Tensor* output = net.infer(input);

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int i = r * width + c;
            float val = output->data[i];

            if      (val > 0.90f) printf("\033[97m@\033[0m"); // strong
            else if (val > 0.75f) printf("\033[97m*\033[0m");
            else if (val > 0.60f) printf("\033[36m*\033[0m");
            else if (val > 0.45f) printf("\033[36m+\033[0m");
            else if (val > 0.30f) printf("\033[35m.\033[0m");
            else if (val > 0.15f) printf("\033[35m,\033[0m");
            else if (val > 0.05f) printf("\033[90m.\033[0m");
            else                  printf(" ");
        }
        printf("\n");
    }

    delete input;
    delete output;
}

#endif
