// Fits a 1D function with an MLP and redraws prediction vs. truth in the terminal while training.

#include "neural_network.h"
#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/mse_loss.h"
#include "optimizers/adam.h"
#include "function_dataset.h"
#include "function_visualizer.h"
#include <cstdio>
#include <cmath>
#include <functional>

// 1 = abs combo
// 2 = piecewise
// 3 = polynomial
// 4 = triple sine
#define FUNCTION 4

int main()
{
    std::function<float(float)> truth;
    float x_min, x_max;

    // f(x) = |x| - |x - 1| + |x + 1|
    if (FUNCTION == 1) {
        truth = [](float x) { return fabsf(x) - fabsf(x - 1.0f) + fabsf(x + 1.0f); };
        x_min = -3.0f; x_max = 3.0f;
    }

    // f(x) = x^2           for x < 0
    //        -x^2 + x      for x >= 0
    else if (FUNCTION == 2) {
        truth = [](float x) { return x < 0 ? x * x : -x * x + x; };
        x_min = -3.0f; x_max = 3.0f;
    }

    // f(x) = x^5 - 4x^3 + 3x
    else if (FUNCTION == 3) {
        truth = [](float x) { return x*x*x*x*x - 4*x*x*x + 3*x; };
        x_min = -2.0f; x_max = 2.0f;
    }

    // f(x) = sin(x) + 0.5*sin(13x) + 0.25*sin(31x)
    else {
        truth = [](float x) { return sinf(x) + 0.5f * sinf(13.0f * x) + 0.25f * sinf(31.0f * x); };
        x_min = -6.28f; x_max = 6.28f;
    }

    NeuralNetwork net;
    net.add_layer(new Linear(1, 32));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(32, 64));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(64, 128));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(128, 256));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(256, 512));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(512, 1024));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(1024, 2048));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(2048, 1024));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(1024, 512));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(512, 256));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(256, 128));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(128, 64));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(64, 32));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(32, 1));
    net.set_loss(new MSELoss());
    net.set_optimizer(new Adam(net.get_layers(), 0.001f));

    FunctionDataset dataset(truth, x_min, x_max, 500);

    int epochs = 20000;
    int draw_every = 10;

    printf("\033[2J");  // clear screen once
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        net.forwardPass(dataset.batch_input, dataset.batch_target);
        net.backwardsPass();
        net.update();

        if (epoch % draw_every == 0)
        {
            printf("\033[H");  // move cursor to top-left, no clear
            printf("Epoch %5d  |  Loss: %.6f\n\n", epoch, net.current_loss);
            visualize_network_vs_truth(net, x_min, x_max, truth, 160, 40);
        }
    }

    return 0;
}
