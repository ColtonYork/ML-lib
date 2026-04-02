CXX = g++
NVCC = /usr/bin/nvcc

CXXFLAGS = -Iinclude
NVCCFLAGS = -Iinclude

SRCS = src/tensor.cpp src/neural_network.cpp src/datasets/mnist.cpp \
       src/layers/relu.cpp src/layers/linear.cpp src/layers/softmax.cpp \
       src/loss/mse_loss.cpp src/loss/cross_entropy_loss.cpp src/loss/softmax_ce_loss.cpp

CUDA_SRCS = src/cuda/cublas_ops.cu src/cuda/kernels.cu

OBJS = src/tensor.o src/neural_network.o src/datasets/mnist.o \
       src/layers/relu.o src/layers/linear.o src/layers/softmax.o \
       src/loss/mse_loss.o src/loss/cross_entropy_loss.o src/loss/softmax_ce_loss.o

CUDA_OBJS = src/cuda/cublas_ops.o src/cuda/kernels.o

TARGET = mllib

all: $(TARGET)

$(TARGET): main.o $(OBJS) $(CUDA_OBJS)
	$(NVCC) -o $(TARGET) main.o $(OBJS) $(CUDA_OBJS) -lcublas -lcurl -lz

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

src/tensor.o: src/tensor.cpp
	$(CXX) $(CXXFLAGS) -c src/tensor.cpp -o src/tensor.o

src/neural_network.o: src/neural_network.cpp
	$(CXX) $(CXXFLAGS) -c src/neural_network.cpp -o src/neural_network.o

src/datasets/mnist.o: src/datasets/mnist.cpp
	$(CXX) $(CXXFLAGS) -c src/datasets/mnist.cpp -o src/datasets/mnist.o

src/layers/relu.o: src/layers/relu.cpp
	$(CXX) $(CXXFLAGS) -c src/layers/relu.cpp -o src/layers/relu.o

src/layers/linear.o: src/layers/linear.cpp
	$(CXX) $(CXXFLAGS) -c src/layers/linear.cpp -o src/layers/linear.o

src/layers/softmax.o: src/layers/softmax.cpp
	$(CXX) $(CXXFLAGS) -c src/layers/softmax.cpp -o src/layers/softmax.o

src/loss/mse_loss.o: src/loss/mse_loss.cpp
	$(CXX) $(CXXFLAGS) -c src/loss/mse_loss.cpp -o src/loss/mse_loss.o

src/loss/cross_entropy_loss.o: src/loss/cross_entropy_loss.cpp
	$(CXX) $(CXXFLAGS) -c src/loss/cross_entropy_loss.cpp -o src/loss/cross_entropy_loss.o

src/loss/softmax_ce_loss.o: src/loss/softmax_ce_loss.cpp
	$(CXX) $(CXXFLAGS) -c src/loss/softmax_ce_loss.cpp -o src/loss/softmax_ce_loss.o

src/cuda/cublas_ops.o: src/cuda/cublas_ops.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/cublas_ops.cu -o src/cuda/cublas_ops.o

src/cuda/kernels.o: src/cuda/kernels.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels.cu -o src/cuda/kernels.o

clean:
	rm -f main.o $(OBJS) $(CUDA_OBJS) $(TARGET)

install:
	mkdir -p /usr/local/include/mllib/cuda
	mkdir -p /usr/local/include/mllib/datasets
	mkdir -p /usr/local/include/mllib/layers
	mkdir -p /usr/local/include/mllib/loss
	mkdir -p /usr/local/include/mllib/optimizers
	mkdir -p /usr/local/lib
	cp include/*.h /usr/local/include/mllib/
	cp include/cuda/*.h /usr/local/include/mllib/cuda/
	cp include/datasets/*.h /usr/local/include/mllib/datasets/
	cp include/layers/*.h /usr/local/include/mllib/layers/
	cp include/loss/*.h /usr/local/include/mllib/loss/
	cp include/optimizers/*.h /usr/local/include/mllib/optimizers/
	ar rcs /usr/local/lib/libMLlib.a $(OBJS) $(CUDA_OBJS)
