// include/tensor.h
#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
public:
    float* data;      // pointer to the actual numbers
    int*   shape;     // e.g. [3, 4] for a 3x4 matrix
    int    ndim;      // number of dimensions (length of shape array)
    bool   on_gpu;    // true if data lives on the GPU

    Tensor(int* shape, int ndim, bool on_gpu = false);
    ~Tensor();

    int num_elements();   // returns total count of numbers (e.g. 3*4 = 12)
    void print();
    void print_shape();

    void to_gpu();
    void to_cpu();

};

#endif