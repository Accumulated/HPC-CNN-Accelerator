#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "Reduce.h"
#include "SQueeze.h"
#include "MBConv.h"
#include "Conv2d.h"

SQueeze::SQueeze(const MBConv_SqueezeExcite* Details, Dimension* InputDim): Details(Details), InputDim(InputDim){

  Dimension* MovingDimension = InputDim;

  ReductionSum *Reduction = new ReductionSum(MovingDimension);

  MovingDimension = Reduction -> RS_GetOutputDim();

  Conv2d *Conv1 = new Conv2d(
                    CONV_1x1,                              /*ConvType*/
                    1,                                     /*stride*/
                    0,                                     /*padding*/
                    SWISH_ACTIVATION,                      /*ActivationTypes*/
                    &(Details -> SQ1.Conv),                /*ConvDetails*/
                    MovingDimension
                  );

  MovingDimension = Conv1 -> Conv2d_GetOutputDim();

  Conv2d *Conv2 = new Conv2d(
                    CONV_1x1,                              /*ConvType*/
                    1,                                     /*stride*/
                    0,                                     /*padding*/
                    SIGMOID_ACTIVATION,                    /*ActivationTypes*/
                    &(Details -> SQ2.Conv),                /*ConvDetails*/
                    MovingDimension
                  );

  MovingDimension = Conv2 -> Conv2d_GetOutputDim();

  layers.push_back(Reduction);
  layers.push_back(Conv1);
  layers.push_back(Conv2);

  this -> OutputDim = Dimension{
    .Height = InputDim -> Height,
    .Width = InputDim -> Width,
    .Depth = InputDim -> Depth,
  };

}


Dimension* SQueeze::SQ_GetOutputDim(){

  return &(this -> OutputDim);

}

SQueeze::~SQueeze()
{
  for (auto layer : layers) {
    delete layer;
  }

  layers.clear();
}


Matrix * SQueeze:: operator()(Matrix* D_input){

  /*
  Steps in squeeze and excite layer:
  1. Get mean value for a tensor
  2. pass the mean to the covolution, swish, convolution, sigmoid
  3. the result will be a 1 x 1 x C, multiply elementwise.
  "each element in a channel is multiplied by the result's corresponding channel element"

  Filter Density means #filters used

  Note: All input matrices are device allocated matrices
  */

  // Step 1: Get mean value for a tensor

  /*
    Get mean values for all channels; Dims(1 x InputDepth x 1)
    Note: Mean matrix is a host allocated memory in REDUCTION_SUM;
    It's used to get the final summation from device and
    then divide each element sequentially by total number
    of elements. It's then later copied back to Result_Mean
    Matrix which is a device matrix.
    "This can be later changed"
  */

  Matrix *Result_Mean = layers[0] -> operator()(D_input);

  // Step 2: pass the mean to the covolution, swish, convolution, sigmoid
  Matrix *Result_Conv1 = layers[1] -> operator()(Result_Mean);
  Matrix *Result_Conv2 = layers[2] -> operator()(Result_Conv1);


  /* Redesign later */
  int nbx = (int)ceil((float)D_input -> width / Tile_GEMM);
  int nby = (int)ceil((float)D_input -> height / Tile_GEMM);
  int nbz = D_input -> depth;

  if (nbx == 0) nbx = 1;
  if (nby == 0) nby = 1;


  // This is the only kernel that runs 3d Grid;
  // Each block in z dimension controls 1 channel
  dim3 dim_Grid2(nbx, nby, nbz);
  dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);


  // C then D, the final multiplication is in C matrix
  ConvChannelElementWiseMultiplication <<< dim_Grid2, dim_Block2 >>> (

    D_input -> elements,
    D_input -> height,
    D_input -> width,
    D_input -> depth,
    Result_Conv2 -> elements

  );


  return Result_Conv2;
}
