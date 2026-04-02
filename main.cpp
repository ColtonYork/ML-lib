/*
#include "include/loss/loss.h"
#include "include/loss/softmax_ce_loss.h"
#include "include/tensor.h"
#include "include/cuda/cublas_ops.h"
#include "include/layers/relu.h"
#include "include/layers/linear.h"
#include "include/neural_network.h"
#include "include/loss/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <random>
*/

#include "include/loss/loss.h"
#include "include/loss/softmax_ce_loss.h"
#include "include/tensor.h"
#include "include/cuda/cublas_ops.h"
#include "include/layers/relu.h"
#include "include/layers/linear.h"
#include "include/neural_network.h"
#include "include/loss/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "include/datasets/mnist.h"
#include <stdio.h>

int main() {

    int shape[] = {3, 3};
    Tensor* a = new Tensor(shape, 2, true);

    a->print();
    return 0;
}
