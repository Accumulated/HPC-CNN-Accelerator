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

    // Allocate an array of Matrix pointers
    this -> Output = new Matrix*[numberOfStreams];
    this -> TransitionMatrix = new Matrix*[numberOfStreams];

    for (int i = 0; i < this -> numberOfStreams; i++) {

        this -> Output[i] = new Matrix(
                            InputDim -> Depth,
                            (int) ceil((double)InputDim -> Height * InputDim -> Width / (2 * BLOCK_SIZE)),
                            1,
                            NULL,
                            DefineOnDevice
                        );

        this -> TransitionMatrix[i] = new Matrix(
                                        this -> InputDim -> Depth,
                                        this -> InputDim -> Height * this -> InputDim -> Width,
                                        1,
                                        NULL,
                                        DefineOnDevice
                                    );
    }
}


Dimension* ReductionSum::RS_GetOutputDim(){

    return &(this -> OutputDim);

}


void ReductionSum:: ResetDims(){

    for (int i = 0; i < this -> numberOfStreams; i++) {

        this -> Output[i] -> height = this -> InputDim -> Depth,
        this -> Output[i] -> width = (int) ceil((double) (this -> InputDim -> Height * this -> InputDim -> Width) / (2 * BLOCK_SIZE)),
        this -> Output[i] -> depth = 1;

        this -> TransitionMatrix[i] -> height = this -> InputDim -> Depth;
        this -> TransitionMatrix[i] -> width = this -> InputDim -> Height * this -> InputDim -> Width;
        this -> TransitionMatrix[i] -> depth = 1;

    }

}


ReductionSum::~ReductionSum(){

}


Matrix** ReductionSum:: operator()(Matrix** D_input)
{
    /*
        The mean will be a row vector of 1 x C where C is number of original matrix
        channels.

        WARNING: This operation will change the input matrix elements and dimensions.

    */

    this -> ResetDims();

    for (int i = 0; i < this -> numberOfStreams; i++) {

        size_t size = this -> TransitionMatrix[i] -> depth * this -> TransitionMatrix[i] -> height * this -> TransitionMatrix[i] -> width * sizeof(float);

        cudaMemcpyAsync(this -> TransitionMatrix[i] -> elements, D_input[i] -> elements, size, cudaMemcpyDeviceToDevice, streams[i]);

        /* Starting reduction sum calculations */
        // Note: All blocks are 1D threads. We use Block.x to reduce 1 channel elements'
        // Diffferent block.y to address different number of channels
        int nbx = (int)ceil((float)(this -> TransitionMatrix[i] -> width) / (2 * BLOCK_SIZE));
        int nby = (int)ceil((float)(this -> TransitionMatrix[i] -> height));
        int nbz = 1;


        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;


        // For loop is held to maintain huge number of summations needed
        for (int idx = 0; this -> TransitionMatrix[i] -> width != 1; idx++)
        {
            dim3 dim_Grid2(nbx, nby, nbz);
            dim3 dim_Block2(BLOCK_SIZE, 1, 1);

            // Make sure to synch between multiple runs
            cudaDeviceSynchronize();

            BN_Kernel_Mean_Reduction <<< dim_Grid2, dim_Block2, 0, this -> streams[i] >>> (
                this -> TransitionMatrix[i] -> elements,
                this -> TransitionMatrix[i] -> height,
                this -> TransitionMatrix[i] -> width,
                this -> TransitionMatrix[i] -> depth,
                this -> Output[i] -> elements,
                this -> Output[i] -> width
            );

            // Save and copy mean values array into the input array
            size = this -> Output[i] -> height * this -> Output[i] -> width * this -> Output[i] -> depth * sizeof(float);

            cudaMemcpyAsync(
                this -> TransitionMatrix[i] -> elements,
                this -> Output[i] -> elements,
                size,
                cudaMemcpyDeviceToDevice,
                this -> streams[i]
            );


            // Modify filter width to fit into the new elements width
            this -> TransitionMatrix[i] -> width = nbx;
            this -> TransitionMatrix[i] -> height = nby;


            // Recalculate number of blocks in x direction
            nbx = (int)ceil((float)(this -> TransitionMatrix[i] -> width) / (2 * BLOCK_SIZE));
            nby = (int)ceil((float)(this -> TransitionMatrix[i] -> height));


            if (nbx == 0) nbx = 1;
            if (nby == 0) nby = 1;


            // Set width of mean matrix to the new number of blocks
            this -> Output[i] -> width = nbx;
            this -> Output[i] -> height = nby;

        }


        // Set mean matrix to 1 X C X 1 to ease further calculations
        this -> Output[i] -> height = 1;
        this -> Output[i] -> width = D_input[i] -> depth;
        this -> Output[i] -> depth = 1;


        nbx = (int)ceil((float)this -> Output[i] -> width / 1024);


        dim3 dim_Grid2(nbx, 1, 1);
        dim3 dim_Block2(1024, 1, 1);

    
        CastingDivision <<<dim_Grid2, dim_Block2, 0, this -> streams[i]>>> (

            this -> Output[i] -> elements,
            this -> Output[i] -> width,
            D_input[i] -> height * D_input[i] ->width

        );


        // Warning: Remember to pre-process Result_Mean matrix to match 1 x 1 x C as it's
        // the input in this case to the next Conv2d.
        this -> Output[i] -> height = 1;
        this -> Output[i] -> width = 1;
        this -> Output[i] -> depth = D_input[i] -> depth;

    }

    
    return this -> Output;

}
