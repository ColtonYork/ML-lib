# MLlib

A CUDA-accelerated neural network library built from scratch in C++. Supports GPU-accelerated training via cuBLAS for matrix operations.

## Features
- Linear layers with He and Xavier initialization
- ReLU and Softmax activations
- MSE, Cross-Entropy, and Softmax Cross-Entropy loss functions
- SGD, Momentum, and Adam optimizers
- MNIST dataset loader
- Automatic memory management on GPU

## Requirements
- Linux or WSL2
- NVIDIA GPU
- CUDA Toolkit (tested on CUDA 11+)
- g++
- nvcc

## Building and Installing

```bash
make
sudo make install
```

`make` builds `build/libMLlib.a`. `make install` copies headers to `/usr/local/include/mllib/` and the static library to `/usr/local/lib/libMLlib.a`.

To uninstall:

```bash
sudo make uninstall
```

## Using in a Project

```bash
nvcc main.cpp -L/usr/local/lib -lMLlib -lcublas -lcurl -lz -o myprogram
```

## Example

```cpp
#include <mllib/neural_network.h>
#include <mllib/layers/linear.h>
#include <mllib/layers/relu.h>
#include <mllib/loss/mse_loss.h>
#include <mllib/optimizers/adam.h>

int main() {
    NeuralNetwork net;
    net.add_layer(new Linear(1, 32));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(32, 32));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(32, 1));
    net.set_loss(new MSELoss());
    net.set_optimizer(new Adam(net.get_layers(), 0.001f)); // set after all layers are added

    // training loop
    for (int e = 0; e < 1000; e++) {
        net.forwardPass(x, t);
        net.backwardsPass();
        net.update();
    }

    // forward pass only; returns a new CPU tensor the caller deletes
    Tensor* prediction = net.infer(x);
    delete prediction;

    return 0;
}
```

## Examples

Example programs live in `examples/`, one directory per program. Helpers that are only used by the examples (toy datasets, terminal visualizers) live in `examples/common/` and are not part of the library.

| Example | What it does |
|---|---|
| `function_fit` | Fits a 1D function and draws prediction vs. truth in the terminal while training |
| `implicit_curve` | Learns a 2D shape from near/far-labeled points and draws it as an ASCII heatmap |
| `mnist` | Trains a fully-connected MNIST classifier |

```bash
make examples          # builds every example into build/
make run-function_fit  # builds and runs one example
```

## MNIST Example

A fully-connected network trained with this library achieves **~97% accuracy** on the MNIST test set after 30 epochs. Full source: `examples/mnist/main.cpp`.

```cpp
#include <mllib/neural_network.h>
#include <mllib/layers/linear.h>
#include <mllib/layers/relu.h>
#include <mllib/datasets/mnist.h>
#include <mllib/loss/softmax_ce_loss.h>
#include <mllib/optimizers/sgd.h>

int main() {
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
```

