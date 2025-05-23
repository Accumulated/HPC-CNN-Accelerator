#include "CommonInclude.h"
#include "InputUnrolling.h"

InputUnrolling::InputUnrolling(Dimension* InputDim, const ConvDetails* Conv_Details, int padding, int stride) :
  InputDim(InputDim),
  padding(padding),
  stride(stride),
  Conv_Details(Conv_Details){


  this -> ConvolutionOutputDim = Dimension{
                                  .Height = (InputDim -> Height + 2 * padding - Conv_Details -> FilterHeight) / stride + 1,
                                  .Width = (InputDim -> Width + 2 * padding - Conv_Details -> FilterWidth) / stride + 1,
                                  .Depth = Conv_Details -> FilterDensity
                                };


  // The unrolled Input matrix has dimensions((C * k * k) x (H_out * W_out) x 1)
  this -> OutputDim = Dimension{
                        .Height = (InputDim -> Depth) * (Conv_Details -> FilterHeight) * (Conv_Details -> FilterWidth),
                        .Width = (this -> ConvolutionOutputDim.Height) * (this -> ConvolutionOutputDim.Width),
                        .Depth = 1
                      };

  // Allocate an array of Matrix pointers
  this -> InputUnrolled = new Matrix*[this -> numberOfStreams];

  // Allocate an array of Matrix pointers
  this -> FilterUnrolled = new Matrix*[this -> numberOfStreams];

  for (int i = 0; i < this -> numberOfStreams; i++) {
    // The unrolled Input matrix has dimensions((C * k * k) x (H_out * W_out) x 1)
    this -> InputUnrolled[i] = new Matrix(
                              this -> OutputDim.Height,
                              this -> OutputDim.Width,
                              this -> OutputDim.Depth,
                              NULL,
                              DefineOnDevice
                            );


    // Unrolled filter has dimesnios (M x (C * k * k) x 1)
    this -> FilterUnrolled[i] = new Matrix(
                                this -> ConvolutionOutputDim.Depth,
                                this -> Conv_Details -> FilterDepth * this -> Conv_Details -> FilterHeight * this -> Conv_Details -> FilterWidth,
                                1,
                                this -> Conv_Details -> ConvWeights,
                                DefineOnDevice
                              );

  }
}


InputUnrolling::~InputUnrolling() {


}


Matrix** InputUnrolling:: operator()() {

  /*
    This overload is returning the unrolled filter ready for matrix multiplication.
    It's already unrolled in the constructor.
  */
  return this -> FilterUnrolled;

}


Matrix** InputUnrolling:: operator()(Matrix** Device_Input) {

  /* Note: All the function input matrices are device matrices. */

  int nbx = (int)ceil((float)(this -> ConvolutionOutputDim.Width) / Tile_GEMM);
  int nby = (int)ceil((float)(this -> ConvolutionOutputDim.Height)  / Tile_GEMM);
  int nbz = Device_Input[0] -> depth;

  if (nbx == 0) nbx = 1;
  if (nby == 0) nby = 1;

  dim3 dim_Grid2(nbx, nby, nbz);
  dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

  for (int i = 0; i < this -> numberOfStreams; i++) {

    // You need to use cudaDeviceSynchronize if the kernel isn't working
    INPUT_UNROLLING <<< dim_Grid2, dim_Block2, 0, this -> streams[i] >>> (

      this -> stride,
      this -> Conv_Details -> FilterHeight,

      Device_Input[i] -> elements,
      Device_Input[i] -> height,
      Device_Input[i] -> width,
      Device_Input[i] -> depth,

      this -> InputUnrolled[i] -> elements,
      this -> InputUnrolled[i] -> height,
      this -> InputUnrolled[i] -> width,
      this -> InputUnrolled[i] -> depth,

      this -> ConvolutionOutputDim.Height,
      this -> ConvolutionOutputDim.Width

    );
  }

    
  return this -> InputUnrolled;

}
