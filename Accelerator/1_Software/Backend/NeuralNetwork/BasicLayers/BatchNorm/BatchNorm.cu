#include "CommonInclude.h"
#include "BatchNorm.h"


BatchNorm:: BatchNorm(const BatchNorm_Weights* Details, ActivationTypes activation, Dimension* InputDim){


  if(Details -> Mean && Details -> Variance && Details -> Weights && Details -> Bias){

    this -> mean = new Matrix(Details -> size,
                              1,
                              1,
                              Details -> Mean,
                              DefineOnDevice);

    this -> bias = new Matrix(Details -> size,
                              1,
                              1,
                              Details -> Bias,
                              DefineOnDevice);

  this -> weights = new Matrix(Details -> size,
                              1,
                              1,
                              Details -> Weights,
                              DefineOnDevice);

  this -> variance = new Matrix(Details -> size,
                                1,
                                1,
                                Details -> Variance,
                                DefineOnDevice);
  }

  this -> activation = activation;

  this -> OutputDim = Dimension{
                        .Height = InputDim -> Height,
                        .Width = InputDim -> Width,
                        .Depth = InputDim -> Depth
                      };

  // Allocate an array of Matrix pointers
  this -> Output = new Matrix*[this -> numberOfStreams];

  for (int i = 0; i < this -> numberOfStreams; i++) {

    this -> Output[i] = new Matrix(this -> OutputDim.Height,
                                this -> OutputDim.Width,
                                this -> OutputDim.Depth,
                                NULL,
                                DefineOnDevice);
  }


}

Dimension* BatchNorm:: BN_GetOutputDim() {

  return &(this -> OutputDim);

}

Matrix** BatchNorm:: operator()(Matrix **D_input)
{

    /* The D_input matrix is a device matrix */

    /*
      All weights, bias, running mean and running variance
      are pre-defined. Just call the function and use the
      matrices.

      All bias, weights, mean and bariance matrices are 1x1xC

      Output Matrix is modified by the equation
      (y = ((x - Mean) / (sqrt(variance) + epsilon)) * weights + bais)
    */

    int nbx = (int)ceil((float)D_input[0] -> width / Tile_GEMM);
    int nby = (int)ceil((float)D_input[0] -> height / Tile_GEMM);
    int nbz = D_input[0] -> depth;

    if (nbx == 0) nbx = 1;
    if (nby == 0) nby = 1;

    // This is the only kernel that runs 3d Grid;
    // Each block in z dimension controls 1 channel
    dim3 dim_Grid3(nbx, nby, nbz);
    dim3 dim_Block3(Tile_GEMM, Tile_GEMM, 1);

    for (int i = 0; i < this -> numberOfStreams; i++) {

      BatchNormKernel <<< dim_Grid3, dim_Block3, 0, this -> streams[i] >>> (

        D_input[i] -> elements,

        this -> Output[i] -> elements,
        this -> Output[i] -> height,
        this -> Output[i] -> width,
        this -> Output[i] -> depth,

        this -> mean -> elements, this -> variance -> elements,
        this -> weights -> elements, this -> bias -> elements,

        this -> activation

      );
    }

    
    return this -> Output;
}
