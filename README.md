# MLlib

A CUDA-accelerated neural network library built from scratch in C++. Supports GPU-accelerated training via cuBLAS for matrix operations.

## Features
- Linear layers with He and Xavier initialization
- ReLU activation
- MSE and Softmax Cross-Entropy loss functions
- SGD optimizer
- Automatic memory management on GPU

## Requirements
- Linux or WSL2
- NVIDIA GPU
- CUDA Toolkit (tested on CUDA 11+)
- g++
- nvcc

## Building the Library
```
make all
make install
```
This installs headers to `~/mllib/include` and the static library to `~/mllib/lib`.

## Using in a Project
Include the headers and link against the library:
```
nvcc main.cpp -I/home/USERNAME/mllib/include -L/home/USERNAME/mllib/lib -lMLlib -lcublas -o myprogram
```

## Example
```cpp
#include "neural_network.h"
#include "linear.h"
#include "relu.h"
#include "mse_loss.h"

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
