/*
#include "include/loss.h"
#include "include/softmax_ce_loss.h"
#include "include/tensor.h"
#include "include/cuda/cublas_ops.h"
#include "include/relu.h"
#include "include/linear.h"
#include "include/neural_network.h"
#include "include/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <random>
*/

#include "include/loss.h"
#include "include/softmax_ce_loss.h"
#include "include/tensor.h"
#include "include/cuda/cublas_ops.h"
#include "include/relu.h"
#include "include/linear.h"
#include "include/neural_network.h"
#include "include/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// XOR regression: Linear(2,8) -> ReLU -> Linear(8,1) -> MSELoss
// Loss should drop from ~0.25 toward 0 over 1000 epochs

int main() {
    srand(42);

    float x_raw[] = { 0,0, 0,1, 1,0, 1,1 };
    float t_raw[] = { 0, 1, 1, 0 };

    int x_shape[] = {4, 2};
    int t_shape[] = {4, 1};

    Tensor* x = new Tensor(x_shape, 2, false);
    Tensor* t = new Tensor(t_shape, 2, false);
    for (int i = 0; i < 8; i++) x->data[i] = x_raw[i];
    for (int i = 0; i < 4; i++) t->data[i] = t_raw[i];

    NeuralNetwork net;
    net.add_layer(new Linear(2, 8, InitType::HE));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(8, 1, InitType::HE));
    net.set_loss(new MSELoss());

    float lr;
    net.learning_rate = lr = 0.05f;
    int epochs = 1000;

    printf("Training XOR  (Linear->ReLU->Linear  MSELoss)\n");
    printf("lr=%.3f  epochs=%d\n\n", lr, epochs);

    for (int e = 1; e <= epochs; e++) {
        net.forwardPass(x, t);
        net.backwardsPass();
        net.update();
        if (e % 100 == 0)
            printf("epoch %4d | loss = %.6f\n", e, net.current_loss);
    }

    printf("\nFinal predictions vs targets:\n");
    printf("  input      pred    target\n");
    const char* inputs[4] = {"[0,0]","[0,1]","[1,0]","[1,1]"};

    Tensor* out = net.forwardPass(x, t);
    out->to_cpu();
    for (int i = 0; i < 4; i++)
        printf("  %-8s  %.4f   %.1f\n", inputs[i], out->data[i], t_raw[i]);

    delete x;
    delete t;
    return 0;
}
