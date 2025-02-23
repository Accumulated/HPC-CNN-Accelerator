#include "CommonInclude.h"
#include "InputUnrolling.h"

InputUnrolling::InputUnrolling(int FilterSize, int stride)
    : FilterSize(FilterSize), Stride(stride) {
    // Constructor implementation


}

InputUnrolling::~InputUnrolling() {
  // Destructor implementation


}


Matrix* InputUnrolling:: operator()(Matrix* Device_Input, int ExpectedConv_OutputHeight, int ExpectedConv_OutputWidth) {

  /* Note: All the function input matrices are device matrices. */

  int nbx = (int)ceil((float)ExpectedConv_OutputWidth / Tile_GEMM);
  int nby = (int)ceil((float)ExpectedConv_OutputHeight / Tile_GEMM);
  int nbz = Device_Input -> depth;

  if (nbx == 0) nbx = 1;
  if (nby == 0) nby = 1;

  dim3 dim_Grid2(nbx, nby, nbz);
  dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

  // You need to use cudaDeviceSynchronize if the kernel isn't working
  INPUT_UNROLLING <<< dim_Grid2, dim_Block2 >>> (
    this -> Stride,
    this -> FilterSize,

    Device_Input -> elements,
    Device_Input -> height,
    Device_Input -> width,
    Device_Input -> depth,

    this -> Output -> elements,
    this -> Output -> height,
    this -> Output -> width,
    this -> Output -> depth,

    ExpectedConv_OutputHeight,
    ExpectedConv_OutputWidth

  );

  return this -> Output;
}
