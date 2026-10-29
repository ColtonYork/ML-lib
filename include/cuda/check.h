#ifndef CHECK_H
#define CHECK_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// wrap any CUDA runtime call: CUDA_CHECK(cudaMalloc(&ptr, bytes));
// prints the failing call, file and line, then aborts
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s\n  at %s:%d\n  in %s\n",            \
                    cudaGetErrorString(err_), __FILE__, __LINE__, #call);       \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// same for cuBLAS calls
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status_ = (call);                                        \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error: %s\n  at %s:%d\n  in %s\n",          \
                    cublasGetStatusString(status_), __FILE__, __LINE__, #call); \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// put right after a kernel<<<...>>> launch to catch bad launch configs.
// errors from inside a running kernel show up at the next CUDA_CHECK'd call (e.g. a cudaMemcpy).
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

#endif
