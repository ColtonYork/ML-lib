# MLlib

A CUDA-accelerated neural network library built from scratch in C++. Supports GPU-accelerated training via cuBLAS for matrix operations.

## Features
- Linear layers with He and Xavier initialization
- ReLU and Softmax activations
- MSE, Cross-Entropy, and Softmax Cross-Entropy loss functions
- SGD optimizer
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
make all
sudo make install
```

This installs headers to `/usr/local/include/mllib/` and the static library to `/usr/local/lib/libMLlib.a`.

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

int main() {
    NeuralNetwork net;
    net.add_layer(new Linear(1, 32));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(32, 32));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(32, 1));
    net.set_loss(new MSELoss());
    net.learning_rate = 0.01f;

    // training loop
    for (int e = 0; e < 1000; e++) {
        net.forwardPass(x, t);
        net.backwardsPass();
        net.update();
    }

    return 0;
}
```

## MNIST Example

A fully-connected network trained with this library achieves **~97% accuracy** on the MNIST test set after 30 epochs.

```cpp
#include <mllib/neural_network.h>
#include <mllib/layers/linear.h>
#include <mllib/layers/relu.h>
#include <mllib/datasets/mnist.h>
#include <mllib/loss/softmax_ce_loss.h>

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
    net.learning_rate = 0.01f;

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

