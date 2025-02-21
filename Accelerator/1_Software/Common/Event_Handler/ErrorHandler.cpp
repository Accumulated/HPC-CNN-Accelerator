#include "CommonInclude.h"


void CheckCudaError(const char* ptr, cudaError err)
{
    if (err == cudaSuccess);
    else
        printf("CUDA error in %s: %s\n", ptr, cudaGetErrorString(err));
}


static void HandleError(cudaError_t err, const char *file, int line) {

    if (err != cudaSuccess) {

        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err)
                  << " \"" << cudaGetErrorString(err) << "\"" << std::endl;
        exit(EXIT_FAILURE);

    }

}