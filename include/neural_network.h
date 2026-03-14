
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "loss.h"
#include "layer.h"
#include "cuda/cublas_ops.h"

class NeuralNetwork {
    private:
        std::vector<Layer*> layers;
        Loss* loss_fn = nullptr;

        Tensor* saved_output = nullptr;
        Tensor* saved_target = nullptr;

        std::vector<Tensor*> intermediates;

        CublasContext cublas;

        void cleanup();

    public:
        void add_layer(Layer* layer);
        void set_loss(Loss* loss);

        Tensor* forwardPass(Tensor* input, Tensor* target);
        void backwardsPass();
        void update();
        
        Layer* get_layer(int i) { return layers[i]; }

        float current_loss = -1.0f;
        float learning_rate = 0.01f;

};
#endif
