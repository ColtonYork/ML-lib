#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "loss/loss.h"
#include "layers/layer.h"
#include "cuda/cublas_ops.h"
#include "optimizers/optimizer.h"

class NeuralNetwork {
    private:
        std::vector<Layer*> layers = {};
        Loss* loss_fn = nullptr;
        Optimizer* optimizer = nullptr;

        Tensor* saved_output = nullptr;
        Tensor* saved_target = nullptr;

        std::vector<Tensor*> intermediates;

        CublasContext cublas;

        void cleanup();

    public:
        void add_layer(Layer* layer);

        void set_loss(Loss* loss);
        void set_optimizer(Optimizer* opt);

        Tensor* forwardPass(Tensor* input, Tensor* target);
        void backwardsPass();
        void update();
        
        Layer* get_layer(int i) { return layers[i]; }
        std::vector<Layer*>* get_layers() { return &layers; }
        
        ~NeuralNetwork()
            {
                for (Layer* layer : layers) delete layer;
                delete loss_fn;
                delete optimizer;
            }

        void print_network_configuration();

        float current_loss = -1.0f;
};
#endif
