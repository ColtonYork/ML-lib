// include/datasets/mnist.h
#ifndef MNIST_H
#define MNIST_H

#include <string>
#include <vector>
#include "../tensor.h"


class MNISTLoader {
public:
    MNISTLoader(int batch_size, int num_samples = -1);
    ~MNISTLoader();

    int num_batches();
    int num_test_batches();

    Tensor* get_image_batch(int i);
    Tensor* get_label_batch(int i);
    Tensor* get_test_image_batch(int i);
    Tensor* get_test_label_batch(int i);

private:
    int batch_size;
    int num_samples;

    std::vector<Tensor*> image_batches;
    std::vector<Tensor*> label_batches;
    std::vector<Tensor*> test_image_batches;
    std::vector<Tensor*> test_label_batches;


    void download_and_decompress(const std::string& url, const std::string& dest_path);
    void download_if_needed();
    int  swap_endian(int val);
    void read_images(const std::string& path, std::vector<float>& out, int& num_images, bool cap);
    void read_labels(const std::string& path, std::vector<float>& out, int& num_labels, bool cap);
    void load();

};

#endif