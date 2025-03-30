#include "EfficientNet.h"
#include <cuda_runtime.h>
#include "mpi.h"

int main(void){


    MPI_Init(NULL, NULL);


    int rank, size, deviceCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cudaGetDeviceCount(&deviceCount); 


    if (deviceCount == 0) {
        std::cerr << "No GPUs found!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Map MPI rank to a GPU
    int deviceID = rank % deviceCount;
    
    // Assign this GPU to the process
    cudaSetDevice(deviceID);           

    EfficientNet *MyNet = new EfficientNet();

    (*(MyNet))();

    delete MyNet;

    MPI_Finalize();
}