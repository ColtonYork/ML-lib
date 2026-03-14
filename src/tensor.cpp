#include "../include/tensor.h"
#include <cuda_runtime.h>
#include <cstdio>
void print_recursive(float* data, int* shape, int ndim, int dim, int offset, int indent);

Tensor::Tensor(int* shape, int ndim, bool on_gpu) : on_gpu(on_gpu), ndim(ndim)
{
    this->shape = new int[ndim];
    for (int i = 0; i < ndim; i++)
        this->shape[i] = shape[i];

    if (on_gpu)
        cudaMalloc(&data, num_elements() * sizeof(float));
    else
        data = new float[num_elements()]();
}

Tensor::~Tensor()
{
    if (on_gpu)
        cudaFree(data);
    else
        delete[] data;

    delete[] shape;
}

int Tensor::num_elements()
{
    int count = 1;
    for (int i = 0; i < ndim; i++)
        {
           count *= this->shape[i]; 
        }
    return count;
}

void print_recursive(float* data, int* shape, int ndim, int dim, int offset, int indent)
{
    for (int i = 0; i < indent; i++) printf("  ");

    if (dim == ndim - 1)
    {
        printf("[");
        for (int i = 0; i < shape[dim]; i++)
        {
            printf("%.4f", data[offset + i]);
            if (i < shape[dim] - 1) printf(", ");
        }
        printf("]\n");
        return;
    }

    printf("[\n");
    for (int i = 0; i < shape[dim]; i++)
    {
        int block_size = 1;
        for (int j = dim + 1; j < ndim; j++)
            block_size *= shape[j];
        print_recursive(data, shape, ndim, dim + 1, offset + i * block_size, indent + 1);
    }
    for (int i = 0; i < indent; i++) printf("  ");
    printf("]\n");
}

void Tensor::print()
{
    float* cpu_data;

    if (on_gpu)
    {
        cpu_data = new float[num_elements()];
        cudaMemcpy(cpu_data, data, num_elements() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    else
        cpu_data = data;

    print_recursive(cpu_data, shape, ndim, 0, 0, 0);

    if (on_gpu)
        delete[] cpu_data;
}

void Tensor::to_gpu()
{
    if (on_gpu)
        return;

    //create devptr
    float* gpu_data;

    // allocate and copy data onto gpu
    cudaMalloc(&gpu_data, num_elements() * sizeof(float));
    cudaMemcpy(gpu_data, data, num_elements() * sizeof(float), cudaMemcpyHostToDevice);

    delete[] data;
    data = gpu_data;
    on_gpu = true;
}


void Tensor::to_cpu()
{
    if (!on_gpu) return;

    cudaDeviceSynchronize();
    float* cpu_data = new float[num_elements()];
    cudaMemcpy(cpu_data, data, num_elements() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(data);
    data = cpu_data;
    on_gpu = false;
}

void Tensor::print_shape()
{
    printf("(");
    for (int i = 0; i < ndim; i++) {
        printf("%d", shape[i]);
        if (i < ndim - 1) printf(", ");
    }
    printf(")\n");
}
