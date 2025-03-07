#include "CommonInclude.h"
#include "FC_Layer.h"

    // Warning: Non-generic implementation - should be redesigned for general purpose
    // Warning: Non-generic implementation - should be redesigned for general purpose
    // Warning: Non-generic implementation - should be redesigned for general purpose

FCLayer::FCLayer(const ConvDetails * Details, Dimension* InputDim, ActivationTypes activation_type):
    InputDim(InputDim),
    activation_type(activation_type){

    // Initialize weights (H, W, 1, 1)
    weight = new Matrix(Details->FilterHeight,
                        Details->FilterWidth,
                        1,
                        Details->ConvWeights,
                        DefineOnDevice);

    // (1 x 1 x C)
    bias = new Matrix(1, Details->FilterWidth, 1, Details->Bias, DefineOnDevice);

    // Calculate the output dimensions
    OutputDim = Dimension{
        .Height = InputDim -> Height,
        .Width = Details -> FilterWidth,
        .Depth = 1
    };

    // Initialize the output matrix
    Output = new Matrix(OutputDim.Height, OutputDim.Width, OutputDim.Depth, nullptr, DefineOnDevice);
}


Dimension* FCLayer::FCLayer_GetOutputDim(){

    return &(this -> OutputDim);
}


FCLayer::~FCLayer(){

}


Matrix* FCLayer::operator()(Matrix* D_input){

    // Get number of blocks
    int nbx = (int)ceil((float)this -> Output -> width / (THREAD_GRANULARITY_BLOCKS * Tile_GEMM));
    int nby = (int)ceil((float)this -> Output -> height / Tile_GEMM);
    int num_block_for_phases = (int)ceil((float)D_input -> width / Tile_GEMM);

    // Check for zero blocks to make sure code runs correctly
    if (nbx == 0) nbx = 1;
    if (nby == 0) nby = 1;

    dim3 dim_Grid2(nbx, nby, 1);
    dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);


    // Call shared memory tiled Multiplication  algorithm
    MatrixMulKernel <<< dim_Grid2, dim_Block2 >>> (
        D_input -> elements, D_input -> height, D_input -> depth, D_input -> width,
        this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,
        this -> Output -> elements, this -> Output -> height, this -> Output -> width, this -> Output -> depth,
        num_block_for_phases, activation_type,
        BIASED, this -> bias -> elements
    );

    this -> Output -> Matrix_DumpDeviceMemory();

    return this -> Output;

}