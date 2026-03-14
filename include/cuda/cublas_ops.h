// include/cuda/cublas_ops.h
#ifndef CUBLAS_OPS_H
#define CUBLAS_OPS_H

#include <cublas_v2.h>
#include "tensor.h"

struct CublasContext {
    cublasHandle_t handle;

    CublasContext();
    ~CublasContext();

    void matmul(Tensor* A, Tensor* B, Tensor* C);
    void matmul_nt(Tensor* A, Tensor* B, Tensor* C);
    void matmul_tn(Tensor* A, Tensor* B, Tensor* C);
};

#endif