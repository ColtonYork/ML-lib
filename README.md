  # MLlib                                                                                                                          
                                                                                                                                   
  A CUDA-accelerated neural network library built from scratch in C++. Supports GPU-accelerated training via cuBLAS for matrix     
  operations.                                                                                                                      
                  
  ## Features
  - Linear layers with He and Xavier initialization
  - ReLU and Softmax activations                                                                                                   
  - MSE, Cross-Entropy, and Softmax Cross-Entropy loss functions
  - SGD optimizer                                                                                                                  
  - MNIST dataset loader
  - Automatic memory management on GPU                                                                                             
  
  ## Requirements                                                                                                                  
  - Linux or WSL2 
  - NVIDIA GPU
  - CUDA Toolkit (tested on CUDA 11+)
  - g++
  - nvcc

  ## Building and Installing                                                                                                       
  make all
  sudo make install                                                                                                                
  This installs headers to `/usr/local/include/mllib/` and the static library to `/usr/local/lib/libMLlib.a`.
     
  To uninstall:
  sudo make uninstall                                                                                                              
  
  ## Using in a Project                                                                                                            
  nvcc main.cpp -L/usr/local/lib -lMLlib -lcublas -lcurl -lz -o myprogram
                                                                                                                                   
  ## Example
  ```cpp                                                                                                                           
  #include <mllib/neural_network.h>
  #include <mllib/layers/linear.h>                                                                                                 
  #include <mllib/layers/relu.h>
  #include <mllib/loss/mse_loss.h>                                                                                                 
                  
  int main() {
      NeuralNetwork net;
      net.add_layer(new Linear(1, 32));                                                                                            
      net.add_layer(new ReLU());
      net.add_layer(new Linear(32, 32));                                                                                           
      net.add_layer(new ReLU());
      net.add_layer(new Linear(32, 1));
      net.set_loss(new MSELoss());                                                                                                 
      net.learning_rate = 0.01f;
                                                                                                                                   
      // training loop
      for (int e = 0; e < 1000; e++) {
          net.forwardPass(x, t);                                                                                                   
          net.backwardsPass();
          net.update();                                                                                                            
      }           

      return 0;
  }