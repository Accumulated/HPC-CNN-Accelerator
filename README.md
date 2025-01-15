# HPC-CNN-Accelerator

The project focuses on accelerating the inference phase of neural networks by leveraging the CUDA programming model for parallel processing. 
Building on the work previously developed [here](https://github.com/Accumulated/Accelerating-CNN-on-GPU-using-CUDA-C), this continuation 
aims to enhance performance further by optimizing memory management and computation efficiency in GPU environments. 

This version of the project focuses on experimenting with the following optimization methods to achieve better scalability and performance:

- **Computation optimizations**:
  1. Memory Layout Operations on Matrices: Implementing a channel-first memory access pattern (CHW) to improve memory access efficiency, optimizing the data flow for both CPU and GPU.
     
  2. Revisit matrix multiplication implementation: Adapting the matrix multiplication implementation to follow techniques used for data-intensive applications, which can lead to improvements
     in computational throughput and efficiency. For more details, see this [playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k).
  
  3. Adding a Batch Dimension: Integrating a batch dimension for all operations to optimize throughput and better utilize the GPU’s parallel processing capabilities.

- **Application Optimizations**:
  1. Multiple GPGPUs integration
  2. MPI integration

- **Utilities and Tools**:
  1. Profiling Tools: Utilizing profiling tools such as Nsight Systems and Scorpion to analyze performance and identify bottlenecks in computation and memory usage.
     These tools will provide insights into the GPU’s workload and potential areas for optimization.

  2. Debugging with MDB and CUDA-GDB: Integrating tools like [MDB](https://github.com/TomMelt/mdb) to facilitate debugging and using CUDA-GDB along with Valgrind to detect and address memory errors,
     ensuring robust memory management in complex parallel environments.
