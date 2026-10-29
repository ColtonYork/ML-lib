// Trains a fully-connected classifier on MNIST (~97% test accuracy after 30 epochs).
// The dataset is downloaded to ~/mllib/data/mnist/ on first run.

#include "neural_network.h"
#include "layers/linear.h"
#include "layers/relu.h"
#include "datasets/mnist.h"
#include "loss/softmax_ce_loss.h"
#include "optimizers/sgd.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main()
{
    srand(time(NULL));
    MNISTLoader loader(64); // batch size 64

    // 784 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 10
    NeuralNetwork net;
    net.add_layer(new Linear(784, 1024));
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
    net.add_layer(new Linear(32, 16));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(16, 10));
    net.set_loss(new SoftmaxCELoss());
    net.set_optimizer(new SGD(net.get_layers(), 0.01f));

    for (int epoch = 0; epoch < 30; epoch++) {
        float total_loss = 0.0f;
        for (int i = 0; i < loader.num_batches(); i++) {
            net.forwardPass(loader.get_image_batch(i), loader.get_label_batch(i));
            net.backwardsPass();
            net.update();
            total_loss += net.current_loss;
        }
        printf("Epoch %2d | Avg Loss: %.4f\n", epoch + 1, total_loss / loader.num_batches());
    }

    return 0;
}
