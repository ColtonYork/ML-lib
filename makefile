CXX = g++
NVCC = /usr/bin/nvcc

CXXFLAGS = -Iinclude
NVCCFLAGS = -Iinclude

SRCS = src/tensor.cpp src/relu.cpp src/linear.cpp src/mse_loss.cpp src/softmax.cpp src/cross_entropy_loss.cpp src/softmax_ce_loss.cpp src/neural_network.cpp
CUDA_SRCS = src/cuda/cublas_ops.cu src/cuda/kernels.cu

OBJS = src/tensor.o src/relu.o src/linear.o src/mse_loss.o src/softmax.o src/cross_entropy_loss.o src/softmax_ce_loss.o src/neural_network.o
CUDA_OBJS = src/cuda/cublas_ops.o src/cuda/kernels.o

TARGET = mllib

all: $(TARGET)

$(TARGET): main.o $(OBJS) $(CUDA_OBJS)
	$(NVCC) -o $(TARGET) main.o $(OBJS) $(CUDA_OBJS) -lcublas

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

src/tensor.o: src/tensor.cpp
	$(CXX) $(CXXFLAGS) -c src/tensor.cpp -o src/tensor.o

src/relu.o: src/relu.cpp
	$(CXX) $(CXXFLAGS) -c src/relu.cpp -o src/relu.o

src/linear.o: src/linear.cpp
	$(CXX) $(CXXFLAGS) -c src/linear.cpp -o src/linear.o

src/mse_loss.o: src/mse_loss.cpp
	$(CXX) $(CXXFLAGS) -c src/mse_loss.cpp -o src/mse_loss.o

src/softmax.o: src/softmax.cpp
	$(CXX) $(CXXFLAGS) -c src/softmax.cpp -o src/softmax.o

src/cross_entropy_loss.o: src/cross_entropy_loss.cpp
	$(CXX) $(CXXFLAGS) -c src/cross_entropy_loss.cpp -o src/cross_entropy_loss.o

src/softmax_ce_loss.o: src/softmax_ce_loss.cpp
	$(CXX) $(CXXFLAGS) -c src/softmax_ce_loss.cpp -o src/softmax_ce_loss.o

src/neural_network.o: src/neural_network.cpp
	$(CXX) $(CXXFLAGS) -c src/neural_network.cpp -o src/neural_network.o

src/cuda/cublas_ops.o: src/cuda/cublas_ops.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/cublas_ops.cu -o src/cuda/cublas_ops.o

src/cuda/kernels.o: src/cuda/kernels.cu
	$(NVCC) $(NVCCFLAGS) -c src/cuda/kernels.cu -o src/cuda/kernels.o

clean:
	rm -f main.o $(OBJS) $(CUDA_OBJS) $(TARGET)

install:
	mkdir -p ~/mllib/include/cuda
	mkdir -p ~/mllib/lib
	cp include/*.h ~/mllib/include/
	cp include/cuda/*.h ~/mllib/include/cuda/
	ar rcs ~/mllib/lib/libMLlib.a $(OBJS) $(CUDA_OBJS)