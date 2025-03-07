#pragma once

void CheckCudaError(const char* ptr, cudaError err);

static void HandleError(cudaError_t err, const char *file, int line) {

    if (err != cudaSuccess) {

        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err)
                  << " \"" << cudaGetErrorString(err) << "\"" << std::endl;
        exit(EXIT_FAILURE);

    }
    else{
        /* Nothing to do */
    }
}


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
