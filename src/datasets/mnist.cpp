#include "../../include/datasets/mnist.h"
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <curl/curl.h>
#include <zlib.h>
#include <cstring>

const std::string MNIST_DIR = std::string(getenv("HOME")) + "/mllib/data/mnist/";

const std::string URLS[] = {
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
};

const std::string FILES[] = {
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
};

MNISTLoader::MNISTLoader(int batch_size, int num_samples) 
: batch_size(batch_size), num_samples(num_samples) 
{
    if (num_samples > 60000)
        this->num_samples = 60000;
    if (num_samples < -1)
        this->num_samples = -1;

    load();
}

MNISTLoader::~MNISTLoader() {
    for (int i = 0; i < image_batches.size(); i++)
        delete image_batches[i];
    for (int i = 0; i < label_batches.size(); i++)
        delete label_batches[i];
    for (int i = 0; i < test_image_batches.size(); i++)
        delete test_image_batches[i];
    for (int i = 0; i < test_label_batches.size(); i++)
        delete test_label_batches[i];
}

void MNISTLoader::download_if_needed() {
    std::string home = getenv("HOME");
    mkdir((home + "/mllib").c_str(), 0755);
    mkdir((home + "/mllib/data").c_str(), 0755);
    mkdir((home + "/mllib/data/mnist").c_str(), 0755);

    for (int i = 0; i < 4; i++) {
        std::string file_path = MNIST_DIR + FILES[i];

        struct stat buffer;
        if (stat(file_path.c_str(), &buffer) == 0) {
            printf("Found %s, skipping download.\n", FILES[i].c_str());
            continue;
        }

        printf("Downloading %s...\n", FILES[i].c_str());
        download_and_decompress(URLS[i], file_path);
    }
}

void MNISTLoader::download_and_decompress(const std::string& url, const std::string& dest_path) {
    std::string gz_path = dest_path + ".gz";

    // --- DOWNLOAD ---
    CURL* curl = curl_easy_init();
    FILE* gz_file = fopen(gz_path.c_str(), "wb");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, gz_file);

    curl_easy_perform(curl);

    curl_easy_cleanup(curl);
    fclose(gz_file);

    // --- DECOMPRESS ---
    gzFile in = gzopen(gz_path.c_str(), "rb");
    FILE* out = fopen(dest_path.c_str(), "wb");

    char buffer[4096];
    int bytes_read;
    while ((bytes_read = gzread(in, buffer, sizeof(buffer))) > 0) {
        fwrite(buffer, 1, bytes_read, out);
    }

    gzclose(in);
    fclose(out);

    // --- DELETE .gz ---
    remove(gz_path.c_str());

    printf("Done.\n");
}

void MNISTLoader::read_images(const std::string& path, std::vector<float>& out, int& num_images, bool cap) {
    FILE* f = fopen(path.c_str(), "rb");

    // read header of file (16 bytes)
    int magic, rows, cols;
    fread(&magic, 4, 1, f);
    fread(&num_images, 4, 1, f);
    fread(&rows, 4, 1, f);
    fread(&cols, 4, 1, f);

    // swap endian on all header values
    magic      = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows       = swap_endian(rows);
    cols       = swap_endian(cols);

    if (cap && num_samples != -1)
        num_images = num_samples;

    // read all pixel bytes
    int num_pixels = num_images * rows * cols;
    std::vector<unsigned char> raw(num_pixels);
    fread(raw.data(), 1, num_pixels, f);

    // normalize to [0, 1]
    out.resize(num_images * rows * cols);
    for (int i = 0; i < num_pixels; i++)
        out[i] = raw[i] / 255.0f;

    fclose(f);
}

void MNISTLoader::read_labels(const std::string& path, std::vector<float>& out, int& num_labels, bool cap) {
    FILE* f = fopen(path.c_str(), "rb");

    // read header of file (8 bytes)
    int magic;
    fread(&magic, 4, 1, f);
    fread(&num_labels, 4, 1, f);

    magic      = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    // cap to num_samples if specified
    if (cap && num_samples != -1)
        num_labels = num_samples;

    // read all label bytes
    std::vector<unsigned char> raw(num_labels);
    fread(raw.data(), 1, num_labels, f);

    // one-hot encode
    out.resize(num_labels * 10, 0.0f);
    for (int i = 0; i < num_labels; i++) {
        int label = raw[i];
        out[i * 10 + label] = 1.0f;
    }

    fclose(f);
}

int MNISTLoader::swap_endian(int val) {
    return ((val & 0xFF) << 24) |
           ((val & 0xFF00) << 8) |
           ((val & 0xFF0000) >> 8) |
           ((val >> 24) & 0xFF);
}

void MNISTLoader::load() {
    download_if_needed();

    // --- TRAINING SET ---
    std::vector<float> images;
    std::vector<float> labels;
    int num_images, num_labels;

    read_images(MNIST_DIR + FILES[0], images, num_images, true);
    read_labels(MNIST_DIR + FILES[1], labels, num_labels, true);

    int num_complete_batches = num_images / batch_size;
    for (int i = 0; i < num_complete_batches; i++) {
        int image_offset = i * batch_size * 784;
        int label_offset = i * batch_size * 10;

        int image_shape[] = {batch_size, 784};
        Tensor* image_batch = new Tensor(image_shape, 2, false);
        memcpy(image_batch->data, images.data() + image_offset, batch_size * 784 * sizeof(float));
        image_batch->to_gpu();
        image_batches.push_back(image_batch);

        int label_shape[] = {batch_size, 10};
        Tensor* label_batch = new Tensor(label_shape, 2, false);
        memcpy(label_batch->data, labels.data() + label_offset, batch_size * 10 * sizeof(float));
        label_batch->to_gpu();
        label_batches.push_back(label_batch);
    }

    printf("Loaded %d training batches of size %d\n", num_complete_batches, batch_size);

    // --- TEST SET ---
    std::vector<float> test_images;
    std::vector<float> test_labels;
    int num_test_images, num_test_labels;

    read_images(MNIST_DIR + FILES[2], test_images, num_test_images, false);
    read_labels(MNIST_DIR + FILES[3], test_labels, num_test_labels, false);

    int num_test_complete_batches = num_test_images / batch_size;
    for (int i = 0; i < num_test_complete_batches; i++) {
        int image_offset = i * batch_size * 784;
        int label_offset = i * batch_size * 10;

        int image_shape[] = {batch_size, 784};
        Tensor* image_batch = new Tensor(image_shape, 2, false);
        memcpy(image_batch->data, test_images.data() + image_offset, batch_size * 784 * sizeof(float));
        image_batch->to_gpu();
        test_image_batches.push_back(image_batch);

        int label_shape[] = {batch_size, 10};
        Tensor* label_batch = new Tensor(label_shape, 2, false);
        memcpy(label_batch->data, test_labels.data() + label_offset, batch_size * 10 * sizeof(float));
        label_batch->to_gpu();
        test_label_batches.push_back(label_batch);
    }

    printf("Loaded %d test batches of size %d\n", num_test_complete_batches, batch_size);
}

int MNISTLoader::num_batches() {
    return image_batches.size();
}

int MNISTLoader::num_test_batches() {
    return test_image_batches.size();
}

Tensor* MNISTLoader::get_image_batch(int i) {
    return image_batches[i];
}

Tensor* MNISTLoader::get_label_batch(int i) {
    return label_batches[i];
}

Tensor* MNISTLoader::get_test_image_batch(int i) {
    return test_image_batches[i];
}

Tensor* MNISTLoader::get_test_label_batch(int i) {
    return test_label_batches[i];
}
