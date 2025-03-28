#include "CommonInclude.h"
#include <iostream>
#include <fstream> // Add this line to include the fstream header
#include <unistd.h> // Include this header for getcwd
#include <limits.h> // Include this header for PATH_MAX

using namespace std;


Matrix:: Matrix(int height, int width, int depth, int density, const float* elements_ref, MatrixType MType):

            // Initialize the layer variables
            depth(depth),
            height(height),
            width(width),
            density(density),
            MType(MType) {

    /* Critical point of design:
     * The elements of the matrix are stored in a 1D array.
     * The elements are stored in a row-major order.
     * The element variable is a pointer to the 1D array.
    */
    if(this -> MType == DefineOnHost) {
        /* If the matrix type should become a host defined matrix:
         * There are 2 senarios for this situation -> Either the host
         * wants to copy the elements from the "elements" pointer to float or
         * the host wants to have a copy from the given "elements" in the pointer.
         *
         * For the moment, the design assumes any host matrix is already defined as a pointer
         * to float that this matrix class should keep track of it with certain CHW configuration.
        */
        this -> elements_ref = elements_ref;

    } else if(MType == DefineOnDevice) {

        /* This is a device matrix that should reside on the GPU device.
         * The elements are copied to the device memory using CUDAmemcpy.
        */
        size_t size = height * width * depth * density * sizeof(float);

        /* Keep a reference to the host float elements sent (if needed). */
        this -> elements_ref = elements_ref;

        cudaError err = cudaMalloc((void **)&(this -> elements), size);

        CheckCudaError("Matrix device construction - Memory allocation on device.", err);

        if(this -> elements_ref == NULL) {
            /* If the elements are not defined, then the matrix is initialized with zeros. */
            err = cudaMemset(this -> elements, 0, size);

            CheckCudaError("Matrix device construction - Memory initialization on device.", err);

        }

        else {

            /* If the elements are defined, then copy the elements to the device. */
            err = cudaMemcpy( this -> elements,
                             elements_ref,
                             size,
                             cudaMemcpyHostToDevice);

            CheckCudaError("Matrix device construction - Memory copy on device.", err);

        }
    }
    else {

        std::cout << "Matrix type is not defined" << std::endl;

    }

#ifdef DEBUG
    std:: cout << "Current matrix dimension is: "
                << height << "x" << width << "x" << depth << "x" << density
                << std::endl;
#endif

}



Matrix:: Matrix(int height, int width, int depth, const float* elements_ref, MatrixType MType):

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
    if(this -> MType == DefineOnHost) {
        /* If the matrix type should become a host defined matrix:
         * There are 2 senarios for this situation -> Either the host
         * wants to copy the elements from the "elements" pointer to float or
         * the host wants to have a copy from the given "elements" in the pointer.
         *
         * For the moment, the design assumes any host matrix is already defined as a pointer
         * to float that this matrix class should keep track of it with certain CHW configuration.
        */
        this -> elements_ref = elements_ref;

    } else if(MType == DefineOnDevice) {

        /* This is a device matrix that should reside on the GPU device.
         * The elements are copied to the device memory using CUDAmemcpy.
        */
        size_t size = height * width * depth * sizeof(float);

        /* Keep a reference to the host float elements sent (if needed). */
        this -> elements_ref = elements_ref;

        cudaError err = cudaMalloc((void **)&(this -> elements), size);

        CheckCudaError("Matrix device construction - Memory allocation on device.", err);

        if(this -> elements_ref == NULL) {
            /* If the elements are not defined, then the matrix is initialized with zeros. */
            err = cudaMemset(this -> elements, 0, size);

            CheckCudaError("Matrix device construction - Memory initialization on device.", err);

        }

        else {

            /* If the elements are defined, then copy the elements to the device. */
            err = cudaMemcpy( this -> elements,
                             elements_ref,
                             size,
                             cudaMemcpyHostToDevice);

            CheckCudaError("Matrix device construction - Memory copy on device.", err);

        }
    }
    else {

        std::cout << "Matrix type is not defined" << std::endl;

    }

#ifdef DEBUG
    std:: cout << "Current matrix dimension is: " << height << "x" << width << "x" << depth << std::endl;
#endif
}


Matrix:: ~Matrix() {

    /* If a matrix is on a Device, make sure to free Whatever cudaMalloc allocated. */
    if(this -> MType == DefineOnDevice) {

        cudaError err = cudaFree((void *) this -> elements);

        CheckCudaError("Matrix destruction \n", err);

    } else if(this -> MType == DefineOnHost) {

        /* Nothing to do */
    }

    else {

        /* */
        std::cout << "Matrix type is not defined" << std::endl;
    }
}


void Matrix:: Matrix_SetDimensions(int height, int width, int depth) {

    this -> height = height;
    this -> width = width;
    this -> depth = depth;

#ifdef DEBUG
    std:: cout << "Changed matrix dimension to: " << height << "x" << width << "x" << depth << std::endl;
#endif
}


void Matrix:: Matrix_DumpDeviceMemory(){

    size_t size = this -> density * this -> depth * this -> height * this -> width * sizeof(float);
    std::cout << "Density: " << this->density << ", Depth: " << this->depth << ", Height: " << this->height << ", Width: " << this->width << ", Size: " << size << std::endl;

    float *ptr = new float[size];

    cudaError err = cudaMemcpy(ptr, this -> elements, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }

    show_me_enhanced(ptr, "Memory dump");

    delete[] ptr;

}


void Matrix:: show_me_enhanced(float* ptr, const char* Name)
{
    // Get the current working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Dumping the matrix in: " << cwd << std::endl;
    } else {
        std::cerr << "getcwd() error" << std::endl;
    }

    std::ofstream outfile("matrix_output.txt");

    if (!outfile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    outfile << "height = " << this->height << " width = " <<  this->width <<  " depth = " << this->depth << std::endl;

    outfile << "{" << std::endl;

    for (int i = 0; i < this -> height * this -> width * this -> depth; i++)
    {
        if (i % this->width == 0 && i >= this->width)
            outfile << std::endl;

        if (i % (this->width * this->height) == 0 && i >= (this->width * this->height));
        float tmp = ptr[i];
        outfile << tmp;
        if (i + 1 == this->height * this->width * this->depth);
        else
            outfile << ", ";
    }

    outfile << "}" << std::endl;
    outfile << std::endl;
    outfile.close();
}