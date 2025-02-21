#include "Typedef.h"
#include "Matrix.h"


Matrix:: Matrix(int depth, int height, int width, float* elements, MatrixType MType):

            // Initialize the layer variables
            depth(depth),
            height(height),
            width(width),
            MType(MType) {

    /* Critical point of design:
     * The elements of the matrix are stored in a 1D array.
     * The elements are stored in a row-major order.
     * The element variable is a pointer to the 1D array.
    */
    if(MType == DefineOnHost) {
        /* If the matrix type should become a host defined matrix:
         * There are 2 senarios for this situation -> Either the host
         * wants to copy the elements from the "elements" pointer to float or
         * the host wants to have a copy from the given "elements" in the pointer.
         *
         * For the moment, the design assumes any host matrix is already defined as a pointer
         * to float that this matrix class should keep track of it with certain CHW configuration.
        */
        this -> elements = elements;

    } else if(MType == DefineOnDevice) {

        /* This is a device matrix that should reside on the GPU device.
         * The elements are copied to the device memory using CUDAmemcpy.
        */
        size_t size = height * width * depth * sizeof(float);

        cudaError err = cudaMemcpy(this -> elements,
                                   elements,
                                   size,
                                   cudaMemcpyHostToDevice);

        CheckCudaError(notification, err);

    }

    else {

        /* */
        std::cout << "Matrix type is not defined" << std::endl;
    }

}



Matrix:: ~Matrix() {

    /* */
    delete[] elements;

}


Matrix:: Matrix_SetDimensions(int depth, int height, int width) {

    /* */
    this -> depth = depth;
    this -> height = height;
    this -> width = width;

}
