#include "CommonInclude.h"



void CheckCudaError(const char* ptr, cudaError err)
{
    if (err == cudaSuccess){
        /* Nothing to do */
    }
    else{

        std::cerr << "CUDA error in " << ptr << ": " << cudaGetErrorString(err) << std::endl;

    }
}
