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
    bias = new Matrix(Details->FilterWidth, 1, 1, Details->Bias, DefineOnDevice);

    // Calculate the output dimensions
    OutputDim = Dimension{
        .Height = InputDim -> Height,
        .Width = Details -> FilterWidth,
        .Depth = 1
    };

    // Allocate an array of Matrix pointers
    this -> Output = new Matrix*[this -> numberOfStreams];

    for (int i = 0; i < this -> numberOfStreams; i++) {

        // Initialize the output matrix
        Output[i] = new Matrix(OutputDim.Height, OutputDim.Width, OutputDim.Depth, nullptr, DefineOnDevice);

    }
}


Dimension* FCLayer::FCLayer_GetOutputDim(){

    return &(this -> OutputDim);
}


FCLayer::~FCLayer(){

}


Matrix** FCLayer::operator()(Matrix** D_input){

    // Get number of blocks
    int nbx = (int)ceil((float)this -> Output[0] -> width / (THREAD_GRANULARITY_BLOCKS * Tile_GEMM));
    int nby = (int)ceil((float)this -> Output[0] -> height / Tile_GEMM);
    int num_block_for_phases = (int)ceil((float)D_input[0] -> depth / Tile_GEMM); //?????

    // Check for zero blocks to make sure code runs correctly
    if (nbx == 0) nbx = 1;
    if (nby == 0) nby = 1;

    dim3 dim_Grid2(nbx, nby, 1);
    dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

    for (int i = 0; i < this -> numberOfStreams; i++) {

        // Call shared memory tiled Multiplication  algorithm
        MatrixMulKernel <<< dim_Grid2, dim_Block2, 0, this -> streams[i] >>> (
            D_input[i] -> elements, D_input[i] -> height, D_input[i] -> depth, D_input[i] -> width, //?????
            this -> weight -> elements, this -> weight -> height, this -> weight -> width, this -> weight -> depth,
            this -> Output[i] -> elements, this -> Output[i] -> height, this -> Output[i] -> width, this -> Output[i] -> depth,
            num_block_for_phases, this -> activation_type,
            BIASED, this -> bias -> elements
        );
    }

    int blockSize = 256;
    int numBlocks = (this -> Output[0] -> width + blockSize - 1) / blockSize;

    
    for (int i = 0; i < this -> numberOfStreams; i++) {

        MatrixAddKernel<<<numBlocks, blockSize, 0, this -> streams[i]>>> (this -> Output[i] -> elements, this -> bias -> elements, this -> Output[i] -> width);

    }
    
    return this -> Output;

}