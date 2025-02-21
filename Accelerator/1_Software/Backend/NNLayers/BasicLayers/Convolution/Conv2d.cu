#include "CommonInclude.h"
#include "Matrix.h"
#include "kernels.h"
#include "Conv2d.h"


Conv2d:: Conv2d(SupportConvolutionOPs ConvType,
                int kernel_size,
                int stride,
                int padding,
                ActivationTypes activation_type,
                Matrix* weight,
                Matrix* bias,
                Matrix* input,
                Matrix* output):

            // Initialize the layer variables
            ConvType(ConvType),
            kernel_size(kernel_size),
            stride(stride),
            padding(padding),
            activation_type(activation_type),
            weight(weight),
            bias(bias),
            input(input),
            output(output) {


}


Conv2d:: ~Conv2d() {

    /* */

}


void Conv2d::operator()() {

   // The multiplication kernel is used for the 1x1 Conv2d and kxk Conv2d
    if (ConvType == CONV_1x1 || ConvType == CONV_KxK)
    {
        // Get number of blocks
        int nbx = (int) ceil((float)(this -> output -> width) / (THREAD_GRANULARITY_BLOCKS * Tile_GEMM));
        int nby = (int) ceil((float)(this -> output -> height) / Tile_GEMM);
        int num_block_for_phases = (int) ceil((float)(this -> weight -> width) / Tile_GEMM);

        // Check for zero blocks to make sure code runs correctly
        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        dim3 dim_Grid2(nbx, nby, 1);
        dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

        if (this -> bias -> elements != NULL)
        {
            this -> bias -> Matrix_SetDimensions(this -> output -> height, 1, 1);

            // Call shared memory tiled Multiplication  algorithm
            MatrixMulKernel<<<dim_Grid2, dim_Block2>>> (

                this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,

                this -> input -> elements, this -> input -> height, this -> input -> width, this -> input -> depth,

                this -> output -> elements, this -> output -> height, this -> output -> width, this -> output -> depth,

                num_block_for_phases, activation_type,

                BIASED, this -> bias -> elements

            );
        }
        else
        {
            // Call shared memory tiled Multiplication  algorithm
            MatrixMulKernel<<<dim_Grid2, dim_Block2>>> (

                this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,

                this -> input -> elements, this -> input -> height, this -> input -> width, this -> input -> depth,

                this -> output -> elements, this -> output -> height, this -> output -> width, this -> output -> depth,

                num_block_for_phases, this -> activation_type,

                NOT_BIASED, NULL

            );
        }
    }

    // This case is for DWConv2d
    else if (ConvType == CONV_DW)
    {

        int nbx = (int)ceil((float)(this -> output -> width) / Tile_GEMM);
        int nby = (int)ceil((float)(this -> output -> height) / Tile_GEMM);
        int nbz = this -> output -> depth;

        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        // This is the only kernel that runs 3d Grid;
        // Each block in z dimension controls 1 channel
        dim3 dim_Grid2(nbx, nby, nbz);
        dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

        DWConv2d_kernel<<<dim_Grid2, dim_Block2>>> (

            this -> input -> elements, this -> input -> height, this -> input -> width, this -> input -> depth,

            this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,

            this -> output -> elements, this -> output -> height, this -> output -> width, this -> output -> depth,

            this -> stride
        );

    }

    else{

        std::cout << "Unsupported Convolution Operation" << std::endl;

    }

}
