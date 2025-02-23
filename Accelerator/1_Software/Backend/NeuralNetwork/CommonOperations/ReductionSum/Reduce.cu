#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "Reduce.h"


ReductionSum::ReductionSum(Dimension* InputDim)
{

    this -> InputDim = InputDim;

    this -> OutputDim = Dimension{
        .Height = InputDim -> Depth,
        .Width = (int)ceil((double)InputDim -> Height * InputDim -> Width / (2 * BLOCK_SIZE)),
        1
    };

    this -> Output = new Matrix(
                        this -> OutputDim.Height,
                        this -> OutputDim.Width,
                        this -> OutputDim.Depth,
                        NULL,
                        DefineOnDevice
                    );

    this -> TransitionMatrix = new Matrix(
                                    this -> OutputDim.Height,
                                    InputDim -> Height * InputDim -> Width,
                                    this -> OutputDim.Depth,
                                    NULL,
                                    DefineOnDevice
                                );

}


void ReductionSum:: ResetDims(){

    this -> Output -> height = this -> OutputDim.Height;
    this -> Output -> width = this -> OutputDim.Width;
    this -> Output -> depth = this -> OutputDim.Depth;

    this -> TransitionMatrix -> height = this -> OutputDim.Height;
    this -> TransitionMatrix -> width = InputDim -> Height * InputDim -> Width;
    this -> TransitionMatrix -> depth = this -> OutputDim.Depth;

}


ReductionSum::~ReductionSum(){

}


Matrix * ReductionSum:: operator()(Matrix* D_input)
{
    /*
        The mean will be a row vector of 1 x C where C is number of original matrix
        channels.

        WARNING: This operation will change the input matrix elements and dimensions.

    */

    this -> ResetDims();

    size_t size = D_input -> depth * D_input -> height * D_input -> width * sizeof(float);

    cudaMemcpy(this -> TransitionMatrix -> elements, D_input -> elements, size, cudaMemcpyDeviceToDevice);

    /* Starting reduction sum calculations */
    // Note: All blocks are 1D threads. We use Block.x to reduce 1 channel elements'
    // Diffferent block.y to address different number of channels
    int nbx = (int)ceil((float)(this -> TransitionMatrix -> width) / (2 * BLOCK_SIZE));
    int nby = (int)ceil((float)(this -> TransitionMatrix -> height));
    int nbz = 1;


    if (nbx == 0) nbx = 1;
    if (nby == 0) nby = 1;


    // For loop is held to maintain huge number of summations needed
    for (int i = 0; this -> TransitionMatrix -> width != 1; i++)
    {
        dim3 dim_Grid2(nbx, nby, nbz);
        dim3 dim_Block2(BLOCK_SIZE, 1, 1);

        // Make sure to synch between multiple runs
        cudaDeviceSynchronize();

        BN_Kernel_Mean_Reduction <<< dim_Grid2, dim_Block2 >>> (
            this -> TransitionMatrix -> elements,
            this -> TransitionMatrix -> height,
            this -> TransitionMatrix -> width,
            this -> TransitionMatrix -> depth,
            this -> Output -> elements,
            this -> Output -> width
        );

        // Save and copy mean values array into the input array
        size = this -> Output -> height * this -> Output -> width * this -> Output -> depth * sizeof(float);

        cudaMemcpy(
            this -> TransitionMatrix -> elements,
            this -> Output -> elements,
            size,
            cudaMemcpyDeviceToDevice
        );


        // Modify filter width to fit into the new elements width
        this -> TransitionMatrix -> width = nbx;
        this -> TransitionMatrix -> height = nby;


        // Recalculate number of blocks in x direction
        nbx = (int)ceil((float)(this -> TransitionMatrix -> width) / (2 * BLOCK_SIZE));
        nby = (int)ceil((float)(this -> TransitionMatrix -> height));


        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;


        // Set width of mean matrix to the new number of blocks
        this -> Output -> width = nbx;
        this -> Output -> height = nby;

    }


    // Set mean matrix to 1 X C X 1 to ease further calculations
    this -> Output -> height = 1;
    this -> Output -> width = D_input -> depth;
    this -> Output -> depth = 1;


    nbx = (int)ceil((float)this -> Output -> width / 1024);


    dim3 dim_Grid2(nbx, 1, 1);
    dim3 dim_Block2(1024, 1, 1);


    CastingDivision <<<dim_Grid2, dim_Block2>>> (

      this -> Output -> elements,
      this -> Output -> width,
      D_input -> height * D_input ->width

    );

    return this -> Output;
}
