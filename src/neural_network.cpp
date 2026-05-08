#include "include/neural_network.h"
#include "include/layers/linear.h"
#include "include/layers/relu.h"
#include "include/layers/softmax.h"
#include <cstdlib>
#include <cstdio>



void NeuralNetwork::add_layer(Layer* layer) 
{
    if (optimizer != nullptr) {
        fprintf(stderr, "Error: cannot add layers after set_optimizer()\n");
        std::abort();
    }

    layer->set_cublas(cublas);
    layers.push_back(layer);
}

    
void NeuralNetwork::set_loss(Loss* loss)
{
    loss_fn = loss;
}

Tensor* NeuralNetwork::forwardPass(Tensor* input, Tensor* target) 
{
    saved_target = target;
    
    for (int i = 0; i < layers.size(); i++) {
        input = layers[i]->forward(input);
        intermediates.push_back(input);
    }
    
    saved_output = input;
    Tensor* loss_tensor = loss_fn->forward(saved_output, target);
    loss_tensor->to_cpu();
    current_loss = loss_tensor->data[0];
    delete loss_tensor;
    return input;
}

void NeuralNetwork::backwardsPass()
{
    Tensor* grad = loss_fn->backward(saved_output, saved_target);
    for (int i = layers.size() - 1; i >= 0; i--) {
        Tensor* prev_grad = grad;
        grad = layers[i]->backward(grad);
        delete prev_grad;
    }
    delete grad;
}

void NeuralNetwork::update()
{
    optimizer->step();
    cleanup();
}

void NeuralNetwork::cleanup()
{
    // free forward intermediates
    for (int i = 0; i < intermediates.size(); i++)
        delete intermediates[i];
    intermediates.clear();

    // free gradients on each layer
    for (int i = 0; i < layers.size(); i++)
        layers[i]->free_gradients();

    saved_output = nullptr; 
}

void NeuralNetwork::set_optimizer(Optimizer* opt)
{
    delete optimizer;
    optimizer = opt;
}

void NeuralNetwork::print_network_configuration()
{
    printf("Neural Network Configuration\n");
    printf("----------------------------\n");
    printf("Layers: %zu\n\n", layers.size());

    for (size_t i = 0; i < layers.size(); i++) {
        if (Linear* lin = dynamic_cast<Linear*>(layers[i])) {
            printf("  [%2zu]  Linear   %4d -> %-4d\n",
                   i, lin->weights->shape[0], lin->weights->shape[1]);
        } else if (dynamic_cast<ReLU*>(layers[i])) {
            printf("  [%2zu]  ReLU\n", i);
        } else if (dynamic_cast<Softmax*>(layers[i])) {
            printf("  [%2zu]  Softmax\n", i);
        } else {
            printf("  [%2zu]  (unknown)\n", i);
        }
    }

    printf("\nArchitecture: ");
    bool first = true;
    for (size_t i = 0; i < layers.size(); i++) {
        Linear* lin = dynamic_cast<Linear*>(layers[i]);
        if (!lin) continue;
        if (first) { printf("%d", lin->weights->shape[0]); first = false; }
        printf(" -> %d", lin->weights->shape[1]);
    }
    printf("\n");
}

