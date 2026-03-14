#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"
#include "tensor.h"
#include "cuda/cublas_ops.h"

enum class InitType {
    HE,
    XAVIER,
    ZERO
};

class Linear : public Layer {

private:
    Tensor* saved_input = nullptr;
    CublasContext* cublas;

    void free_gradients() override;

public:
    Tensor* weights;
    Tensor* bias;
    Tensor* grad_weights = nullptr;
    Tensor* grad_bias = nullptr;

    Linear(int in_features, int out_features, InitType init = InitType::HE);
    Tensor* forward(Tensor* input) override;
    Tensor* backward(Tensor* grad_output) override;
    void set_cublas(CublasContext& ctx) override;
    void update_weights(float lr) override;

};

#endif