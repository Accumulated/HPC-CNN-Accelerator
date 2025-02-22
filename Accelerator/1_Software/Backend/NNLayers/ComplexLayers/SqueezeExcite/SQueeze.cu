#include "CommonInclude.h"
#include "SQueeze.h"

SQueeze::SQueeze(
  float *Filter1, float *Filter2,
  int FilterDensity1, int FilterDensity2,
  int input_channels, int output_channels,
  float* First_bias, float* Second_bias
){


}

SQueeze::~SQueeze()
{

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
  /* Redesign later */
  //  Matrix MEAN, Result_Mean;

  //  Set_DeviceMatrix(D_input -> depth,
  //    (int)ceil((double)D_input -> height * D_input -> width / (2 * BLOCK_SIZE)),
  //    1,
  //    &Result_Mean,
  //    "Reesult Mean matrix allocated in device memory");

  //  REDUCTION_SUM(D_input, &MEAN, &Result_Mean);


/*
  // Tmp1 is used as a transition between 2 convolution layers; Dims(1 x 1 x FilterDensity3)
  Matrix tmp1;
  Set_DeviceMatrix(1, 1, FilterDensity1, &tmp1, "Allocating tmp1 in device for transition");

  // tmp2 matrix is the result from sigmoid function: Dims(1 x 1 x FilterDensity4)
  Matrix tmp2;
  Set_DeviceMatrix( 1, 1, FilterDensity2, &tmp2, "Allocating tmp2 in device for final output");

  // Sequence: Conv1x1, swish, Conv1x1, sigmoid
  // Warning: Remember to pre-process Result_Mean matrix to match 1 x 1 x C as it's the input in this case to Conv2d
  Set_HostMatrix(1, 1, D_input -> depth, &Result_Mean);

  Conv2d_Layer(&Result_Mean, this -> FilterBank1, this -> FirstConvOutput, 1, 0, input_channels, output_channels, FilterDensity1,
  CONV_1x1, SWISH_ACTIVATION,
  BIASED, First_bias);

  Conv2d_Layer(this -> FirstConvOutput, this -> FilterBank2, this -> SecondConvOutput, 1, 0, output_channels, input_channels, FilterDensity2,
  CONV_1x1, SIGMOID_ACTIVATION,
  BIASED, Second_bias);
  */

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
    this -> SecondConvOutput -> elements

  );


  return this -> Output;
}
