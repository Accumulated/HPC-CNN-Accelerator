#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "Reduce.h"


ReductionSum::ReductionSum(Dimension* InputDim)
{

    this -> InputDim = InputDim;

    /* WARNING: This is only the expected output dimension after execution. */
    this -> OutputDim = Dimension{
                            .Height = 1,
                            .Width = 1,
                            InputDim -> Depth
                        };

    this -> Output = new Matrix(
                        InputDim -> Depth,
                        (int) ceil((double)InputDim -> Height * InputDim -> Width / (2 * BLOCK_SIZE)),
                        1,
                        NULL,
                        DefineOnDevice
                    );

    this -> TransitionMatrix = new Matrix(
                                    this -> InputDim -> Depth,
                                    this -> InputDim -> Height * this -> InputDim -> Width,
                                    1,
                                    NULL,
                                    DefineOnDevice
                                );

}


Dimension* ReductionSum::RS_GetOutputDim(){

    return &(this -> OutputDim);

}


void ReductionSum:: ResetDims(){

    this -> Output -> height = this -> InputDim -> Depth,
    this -> Output -> width = (int) ceil((double) (this -> InputDim -> Height * this -> InputDim -> Width) / (2 * BLOCK_SIZE)),
    this -> Output -> depth = 1;

    this -> TransitionMatrix -> height = this -> InputDim -> Depth;
    this -> TransitionMatrix -> width = this -> InputDim -> Height * this -> InputDim -> Width;
    this -> TransitionMatrix -> depth = 1;

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

    size_t size = this -> TransitionMatrix -> depth * this -> TransitionMatrix -> height * this -> TransitionMatrix -> width * sizeof(float);

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


    // Warning: Remember to pre-process Result_Mean matrix to match 1 x 1 x C as it's
    // the input in this case to the next Conv2d.
    this -> Output -> height = 1;
    this -> Output -> width = 1;
    this -> Output -> depth = D_input -> depth;


    return this -> Output;


}
