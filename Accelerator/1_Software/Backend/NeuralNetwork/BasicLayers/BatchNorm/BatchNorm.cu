#include "CommonInclude.h"
#include "BatchNorm.h"


BatchNorm:: BatchNorm(const BatchNorm_Weights* Details, ActivationTypes activation, Dimension* InputDim){


  if(Details -> Mean && Details -> Variance && Details -> Weights && Details -> Bias){

    /* Missing output allocation and preperation */
    this -> mean = new Matrix(sizeof(Details -> Mean) / sizeof(float),
                              1,
                              1,
                              Details -> Mean,
                              DefineOnDevice);

    this -> bias = new Matrix(sizeof(Details -> Bias) / sizeof(float),
                              1,
                              1,
                              Details -> Bias,
                              DefineOnDevice);

  this -> weights = new Matrix(sizeof(Details -> Weights) / sizeof(float),
                              1,
                              1,
                              Details -> Weights,
                              DefineOnDevice);

  this -> variance = new Matrix(sizeof(Details -> Variance) / sizeof(float),
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

  this -> Output = new Matrix(this -> OutputDim.Height,
                              this -> OutputDim.Width,
                              this -> OutputDim.Depth,
                              NULL,
                              DefineOnDevice);

}


Matrix* BatchNorm:: operator()(Matrix *D_input)
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

    int nbx = (int)ceil((float)D_input -> width / Tile_GEMM);
    int nby = (int)ceil((float)D_input -> height / Tile_GEMM);
    int nbz = D_input -> depth;

    if (nbx == 0) nbx = 1;
    if (nby == 0) nby = 1;

    // This is the only kernel that runs 3d Grid;
    // Each block in z dimension controls 1 channel
    dim3 dim_Grid3(nbx, nby, nbz);
    dim3 dim_Block3(Tile_GEMM, Tile_GEMM, 1);

    BatchNormKernel <<< dim_Grid3, dim_Block3 >>> (

      D_input -> elements,

      this -> Output -> elements,
      this -> Output -> height,
      this -> Output -> width,
      this -> Output -> depth,

      this -> mean -> elements, this -> variance -> elements,
      this -> weights -> elements, this -> bias -> elements,

      this -> activation

    );


    return this -> Output;
}
