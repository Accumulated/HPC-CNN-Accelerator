#include "CommonInclude.h"
#include "kernels.h"
#include "Conv2d.h"

Conv2d:: Conv2d(SupportConvolutionOPs ConvType,
                int kernel_size,
                int stride,
                int padding,
                float* weight,
                float* bias,
                float* input,
                float* output):

            // Initialize the layer variables
            ConvType(ConvType),
            kernel_size(kernel_size),
            stride(stride),
            padding(padding),
            weight(weight),
            bias(bias),
            input(input),
            output(output) {

    /* */
    this.weight =

}


Conv2d:: ~Conv2d() {

    /* */

}


void Conv2d::operator()() {
/*
   // The multiplication kernel is used for the 1x1 Conv2d and kxk Conv2d
    if (ConvType == CONV_1x1 || ConvType == CONV_KxK)
    {
        // Get number of blocks
        int nbx = (int)ceil((float)out_11 -> width / (THREAD_GRANULARITY_BLOCKS * Tile_GEMM));
        int nby = (int)ceil((float)out_11 -> height / Tile_GEMM);
        int num_block_for_phases = (int)ceil((float)D_1 -> width / Tile_GEMM);

        // Check for zero blocks to make sure code runs correctly
        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        dim3 dim_Grid2(nbx, nby, 1);
        dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

        if (BIASED_CHOISE == BIASED)
        {
          Set_HostMatrix(out_11 -> height, 1, 1, biasMat);

          // Call shared memory tiled Multiplication  algorithm
          MatrixMulKernel <<< dim_Grid2, dim_Block2 >>> (D_1 -> elements, D_1 -> height, D_1 -> width, D_1 -> depth,
                                                         D_2 -> elements, D_2 -> height, D_2 -> width, D_2 -> depth,
                                                         out_11 -> elements, out_11 -> height, out_11 -> width, out_11 -> depth,
                                                         num_block_for_phases, activation_type,
                                                         BIASED_CHOISE, biasMat -> elements);
        }
        else
        {
          MatrixMulKernel <<< dim_Grid2, dim_Block2 >>> (D_1 -> elements, D_1 -> height, D_1 -> width, D_1 -> depth,
                                                         D_2 -> elements, D_2 -> height, D_2 -> width, D_2 -> depth,
                                                         out_11 -> elements, out_11 -> height, out_11 -> width, out_11 -> depth,
                                                         num_block_for_phases, activation_type,
                                                         BIASED_CHOISE, NULL);
         }
    }

    // This case is for DWConv2d
    else if (ConvType == CONV_DW)
    {
        int nbx = (int)ceil((float)out_11 -> width / TileDW);
        int nby = (int)ceil((float)out_11 -> height / TileDW);
        int nbz = out_11 -> depth;

        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        // This is the only kernel that runs 3d Grid;
        // Each block in z dimension controls 1 channel
        dim3 dim_Grid2(nbx, nby, nbz);
        dim3 dim_Block2(TileDW, TileDW, 1);


        DWConv2d_kernel <<< dim_Grid2, dim_Block2 >>> (D_2 -> elements, D_2 -> height, D_2 -> width, D_2 -> depth,
                                                         D_1 -> elements, D_1 -> height, D_1 -> width, D_1 -> depth,
                                                         out_11 -> elements, out_11 -> height, out_11 -> width, out_11 -> depth,
                                                         stride_DW);
    }

    else{
        std::cout << "Unsupported Convolution Operation" << std::endl;
    }

    // Reset the output dimensions to continue in the network
    Set_HostMatrix(ReconstructOutHieght, ReconstructOutWidth, ReconstructOutDepth, out_11); */



}
