#include "CommonInclude.h"
#include "Conv2d.h"

using namespace std;

Conv2d:: Conv2d(SupportConvolutionOPs ConvType,
                int stride,
                int padding,
                ActivationTypes activation_type,
                const ConvDetails * Details,
                Dimension* InputDim):

            // Initialize the layer variables
            ConvType(ConvType),
            stride(stride),
            padding(padding),
            activation_type(activation_type),
            InputDim(InputDim){

    if(Details -> ConvWeights){

        /* Missing output allocation and preperation */
        this -> weight = new Matrix(Details -> FilterHeight,
                                    Details -> FilterWidth,
                                    Details -> FilterDepth,
                                    Details -> FilterDensity,
                                    Details -> ConvWeights,
                                    DefineOnDevice);
    }

    if(Details -> Bias){

        this -> bias = new Matrix(Details -> FilterDensity,
                                    1,
                                    1,
                                    Details -> Bias,
                                    DefineOnDevice);
    }

    // Height and width changes, Only depth remains still
    int OutputHeight = (InputDim -> Height + 2 * padding - Details -> FilterHeight)/stride + 1;
    int OutputWidth = (InputDim -> Width + 2 * padding - Details -> FilterWidth)/stride + 1;
    int OutputDepth = 0;

    if(ConvType == CONV_1x1){

        // Output depth is the number of filters available (Density)
        OutputDepth = Details -> FilterDensity;

        this -> OutputDim = Dimension{
            .Height = OutputHeight,
            .Width = OutputWidth,
            .Depth = OutputDepth
        };

        // Modify Filter Matrix to have dimensions ((K^2 * M) x C x 1); K = 1
        this -> weight -> Matrix_SetDimensions(
            Details -> FilterHeight * Details -> FilterWidth * Details -> FilterDensity,
            Details -> FilterDepth,
            1
        );


        // Modify Output Matrix preprocessing to have dimesions ((K^2 * M) x (H * W) x 1);
        this -> Output = new Matrix(
                            // (K^2 * M)
                            Details -> FilterHeight * Details -> FilterWidth * Details -> FilterDensity,
                            // (H * W)
                            OutputHeight * OutputWidth,
                            // 1 channel
                            1,
                            NULL,
                            DefineOnDevice
                        );

    }
    else if(ConvType == CONV_KxK){

        // Output depth is the number of filters available (Density)
        OutputDepth = Details -> FilterDensity;


        this -> OutputDim = Dimension{
            .Height = OutputHeight,
            .Width = OutputWidth,
            .Depth = OutputDepth
        };


        this -> Conv_InputUnrolling = new InputUnrolling(
                                            InputDim,
                                            Details,
                                            padding,
                                            stride
                                        );

        // Modify Output Matrix preprocessing to have dimesions ((K^2 * M) x (H * W) x 1);
        // Convolution output has dimensions of (M x (H_out * W_out) x 1)
        this -> Output = new Matrix(
                                // (K^2 * M)
                                Details -> FilterDensity,
                                // (H * W)
                                OutputHeight * OutputWidth,
                                // 1 channel
                                1,
                                NULL,
                                DefineOnDevice
                            );

    }
    else if(ConvType == CONV_DW){

        // Output depth is the same as the input depth
        OutputDepth = InputDim -> Depth;

        this -> OutputDim = Dimension{
            .Height = OutputHeight,
            .Width = OutputWidth,
            .Depth = OutputDepth
        };

        this -> Output = new Matrix(
                            OutputHeight,
                            OutputWidth,
                            OutputDepth,
                            NULL,
                            DefineOnDevice
                        );
    }

    else{

        std::cout << "Unsupported Convolution Operation" << std::endl;

    }

    if(padding){

        this -> pad = new PaddingLayer(InputDim, this -> padding);

    }

}


Dimension* Conv2d:: Conv2d_GetOutputDim() {

    return &(this -> OutputDim);

}


Conv2d:: ~Conv2d() {

    /* */

}


Matrix* Conv2d::operator()(Matrix *D_input) {

    /* First: Some preprocessing. */


   // The multiplication kernel is used for the 1x1 Conv2d and kxk Conv2d
    if (ConvType == CONV_1x1)
    {

        /*
            CONV_1x1
            // Modify Input matrix to have dimensions (C x (H * W) x 1)
            Set_HostMatrix(InputIMG -> depth, InputIMG -> height * InputIMG -> width, 1, InputIMG);
        */
        // Modify Input matrix to have dimensions (C x (H * W) x 1)
        D_input->Matrix_SetDimensions(
            D_input -> depth,
            D_input -> height * D_input -> width,
            1
        );

        // Get number of blocks
        int nbx = (int) ceil((float)(this -> Output -> width) / (THREAD_GRANULARITY_BLOCKS * Tile_GEMM));
        int nby = (int) ceil((float)(this -> Output -> height) / Tile_GEMM);
        int num_block_for_phases = (int) ceil((float)(this -> weight -> width) / Tile_GEMM);

        // Check for zero blocks to make sure code runs correctly
        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        dim3 dim_Grid2(nbx, nby, 1);
        dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

        if (this -> bias != NULL)
        {
            this -> bias -> Matrix_SetDimensions(this -> Output -> height, 1, 1);

            // Call shared memory tiled Multiplication  algorithm
            MatrixMulKernel<<<dim_Grid2, dim_Block2>>> (

                this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,

                D_input -> elements, D_input -> height, D_input -> width, D_input -> depth,

                this -> Output -> elements, this -> Output -> height, this -> Output -> width, this -> Output -> depth,

                num_block_for_phases, activation_type,

                BIASED, this -> bias -> elements

            );
        }
        else
        {
            // Call shared memory tiled Multiplication  algorithm
            MatrixMulKernel<<<dim_Grid2, dim_Block2>>> (

                this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,

                D_input -> elements, D_input -> height, D_input -> width, D_input -> depth,

                this -> Output -> elements, this -> Output -> height, this -> Output -> width, this -> Output -> depth,

                num_block_for_phases, this -> activation_type,

                NOT_BIASED, NULL

            );
        }
    }


    else if(ConvType == CONV_KxK){

        Matrix *ptr = D_input;

        if(this -> padding){
            ptr = (*this -> pad)(D_input);
        }

        // 1st phase: Filter unrolling and Input unrolling
        Matrix *FilterUnrolled = (*this -> Conv_InputUnrolling)();
        Matrix *InputUnrolled = (*this -> Conv_InputUnrolling)(ptr);

        // Get number of blocks
        int nbx = (int) ceil((float)(this -> Output -> width) / (THREAD_GRANULARITY_BLOCKS * Tile_GEMM));
        int nby = (int) ceil((float)(this -> Output -> height) / Tile_GEMM);
        int num_block_for_phases = (int) ceil((float)(FilterUnrolled -> width) / Tile_GEMM);

        // Check for zero blocks to make sure code runs correctly
        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        dim3 dim_Grid2(nbx, nby, 1);
        dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

        if (this -> bias != NULL)
        {
            this -> bias -> Matrix_SetDimensions(this -> Output -> height, 1, 1);

            // Call shared memory tiled Multiplication  algorithm
            MatrixMulKernel<<<dim_Grid2, dim_Block2>>> (

                FilterUnrolled -> elements, FilterUnrolled -> height, FilterUnrolled -> width, FilterUnrolled -> depth,

                InputUnrolled -> elements, InputUnrolled -> height, InputUnrolled -> width, InputUnrolled -> depth,

                this -> Output -> elements, this -> Output -> height, this -> Output -> width, this -> Output -> depth,

                num_block_for_phases, activation_type,

                BIASED, this -> bias -> elements

            );
        }
        else
        {
            // Call shared memory tiled Multiplication  algorithm
            MatrixMulKernel<<<dim_Grid2, dim_Block2>>> (

                FilterUnrolled -> elements, FilterUnrolled -> height, FilterUnrolled -> width, FilterUnrolled -> depth,

                InputUnrolled -> elements, InputUnrolled -> height, InputUnrolled -> width, InputUnrolled -> depth,

                this -> Output -> elements, this -> Output -> height, this -> Output -> width, this -> Output -> depth,

                num_block_for_phases, this -> activation_type,

                NOT_BIASED, NULL

            );
        }
    }


    // This case is for DWConv2d
    else if (ConvType == CONV_DW)
    {

        Matrix *ptr = D_input;

        if(this -> padding){
            ptr = (*this -> pad)(D_input);
        }

        int nbx = (int)ceil((float)(this -> Output -> width) / Tile_GEMM);
        int nby = (int)ceil((float)(this -> Output -> height) / Tile_GEMM);
        int nbz = this -> Output -> depth;

        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        // This is the only kernel that runs 3d Grid;
        // Each block in z dimension controls 1 channel
        dim3 dim_Grid2(nbx, nby, nbz);
        dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

        DWConv2d_kernel<<<dim_Grid2, dim_Block2>>> (

            ptr -> elements, ptr -> height, ptr -> width, ptr -> depth,

            this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,

            this -> Output -> elements, this -> Output -> height, this -> Output -> width, this -> Output -> depth,

            this -> stride
        );

    }

    else{

        std::cout << "Unsupported Convolution Operation" << std::endl;

    }

    this -> Output -> height = this -> OutputDim.Height;
    this -> Output -> width = this -> OutputDim.Width;
    this -> Output -> depth = this -> OutputDim.Depth;

    return this -> Output;
}
