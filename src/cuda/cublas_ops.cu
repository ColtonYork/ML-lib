#include "cuda/cublas_ops.h"
#include "cuda/check.h"

CublasContext::CublasContext()
{
    CUBLAS_CHECK(cublasCreate(&handle));
}

CublasContext::~CublasContext()
{
    CUBLAS_CHECK(cublasDestroy(handle));
}

void CublasContext::matmul(Tensor* A, Tensor* B, Tensor* C)
{
    if (!A->on_gpu) A->to_gpu();
    if (!B->on_gpu) B->to_gpu();
    if (!C->on_gpu) C->to_gpu();

    float alpha = 1.0f;
    float beta = 0.0f;

    int m = A->shape[0];
    int k = A->shape[1];
    int n = B->shape[1];

    // Compute B^T * A^T = (A * B)^T, which gives correct row-major result
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B->data, n, A->data, k, &beta, C->data, n));
}

void CublasContext::matmul_nt(Tensor* A, Tensor* B, Tensor* C)
{
    float alpha = 1.0f, beta = 0.0f;
    int m = A->shape[0];
    int k = A->shape[1];
    int n = B->shape[0];

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B->data, k, A->data, k, &beta, C->data, n));
}

void CublasContext::matmul_tn(Tensor* A, Tensor* B, Tensor* C)
{
    float alpha = 1.0f, beta = 0.0f;
    int m = A->shape[1];
    int k = A->shape[0];
    int n = B->shape[1];

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, B->data, n, A->data, m, &beta, C->data, n));
}
