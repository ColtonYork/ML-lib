// Learns a 2D shape (a smiley face) from points labeled near/far from the curve,
// and redraws the network's output as an ASCII heatmap while training.

#include "neural_network.h"
#include "layers/linear.h"
#include "layers/relu.h"
#include "loss/mse_loss.h"
#include "optimizers/adam.h"
#include "implicit_dataset.h"
#include "implicit_visualizer.h"
#include <cstdio>
#include <cmath>

// distance from (x, y) to the nearest part of the face
float curve(float x, float y)
{
    // head outline
    float head = fabsf(sqrtf(x*x + y*y) - 1.0f);

    // left eye — filled disk
    float ldx = x + 0.38f, ldy = y - 0.38f;
    float left_eye = fmaxf(0.0f, sqrtf(ldx*ldx + ldy*ldy) - 0.13f);

    // right eye — filled disk
    float rdx = x - 0.38f, rdy = y - 0.38f;
    float right_eye = fmaxf(0.0f, sqrtf(rdx*rdx + rdy*rdy) - 0.13f);

    // left eyebrow — short arc above left eye
    float lbx = x + 0.38f, lby = y - 0.62f;
    float left_brow = fabsf(sqrtf(lbx*lbx + lby*lby) - 0.18f);
    if (x < -0.55f || x > -0.18f) left_brow = 1.0f;

    // right eyebrow
    float rbx = x - 0.38f, rby = y - 0.62f;
    float right_brow = fabsf(sqrtf(rbx*rbx + rby*rby) - 0.18f);
    if (x > 0.55f || x < 0.18f) right_brow = 1.0f;

    // smile — arc, lower half only
    float smx = x, smy = y + 0.15f;
    float smile = fabsf(sqrtf(smx*smx + smy*smy) - 0.55f);
    if (y > -0.15f) smile = 1.0f;
    if (x < -0.45f || x > 0.45f) smile = 1.0f;

    // nose — tiny dot
    float nose = fmaxf(0.0f, sqrtf(x*x + (y-0.05f)*(y-0.05f)) - 0.06f);

    return fminf(head,
           fminf(left_eye,
           fminf(right_eye,
           fminf(left_brow,
           fminf(right_brow,
           fminf(smile, nose))))));
}

int main()
{
    NeuralNetwork net;
    net.add_layer(new Linear(2, 256));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(256, 512));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(512, 512));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(512, 512));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(512, 256));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(256, 128));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(128, 1));
    net.set_loss(new MSELoss());
    net.set_optimizer(new Adam(net.get_layers(), 0.0005f));

    ImplicitDataset dataset(curve, -1.3f, 1.3f, -1.3f, 1.3f, 30000, 0.07f);

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
            printf("\033[H");
            printf("Epoch %5d  |  Loss: %.6f\n\n", epoch, net.current_loss);
            visualize_implicit(net, -1.3f, 1.3f, -1.3f, 1.3f, 100, 35);
        }
    }

    return 0;
}
