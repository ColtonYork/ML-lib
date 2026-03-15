/*
#include "include/loss.h"
#include "include/softmax_ce_loss.h"
#include "include/tensor.h"
#include "include/cuda/cublas_ops.h"
#include "include/relu.h"
#include "include/linear.h"
#include "include/neural_network.h"
#include "include/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <random>
*/

#include "include/loss.h"
#include "include/softmax_ce_loss.h"
#include "include/tensor.h"
#include "include/cuda/cublas_ops.h"
#include "include/relu.h"
#include "include/linear.h"
#include "include/neural_network.h"
#include "include/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "include/datasets/mnist.h"

#include "include/datasets/mnist.h"
#include <stdio.h>

int main() {
    // test 1: default load (all 60,000)
    printf("=== Test 1: Full dataset ===\n");
    MNISTLoader loader1(64);
    printf("Training batches: %d (expected 937)\n", loader1.num_batches());
    printf("Test batches: %d (expected 156)\n\n", loader1.num_test_batches());

    // test 2: limited samples
    printf("=== Test 2: 1000 samples ===\n");
    MNISTLoader loader2(64, 1000);
    printf("Training batches: %d (expected 15)\n", loader2.num_batches());
    printf("Test batches: %d (expected 156)\n\n", loader2.num_test_batches());

    // test 3: over 60000 gets capped
    printf("=== Test 3: 99999 samples (should cap to 60000) ===\n");
    MNISTLoader loader3(64, 99999);
    printf("Training batches: %d (expected 937)\n", loader3.num_batches());
    printf("Test batches: %d (expected 156)\n\n", loader3.num_test_batches());

    // test 4: check tensor shapes are correct
    printf("=== Test 4: Tensor shapes ===\n");
    MNISTLoader loader4(64);
    Tensor* img = loader4.get_image_batch(0);
    Tensor* lbl = loader4.get_label_batch(0);
    printf("Image batch shape: (%d, %d) (expected 64, 784)\n", img->shape[0], img->shape[1]);
    printf("Label batch shape: (%d, %d) (expected 64, 10)\n", lbl->shape[0], lbl->shape[1]);

    return 0;
}
