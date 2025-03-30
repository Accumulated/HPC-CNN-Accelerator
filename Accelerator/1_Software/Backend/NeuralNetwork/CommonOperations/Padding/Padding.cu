#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "Padding.h"


PaddingLayer::PaddingLayer(Dimension *InputDim, int Boundary){

      this -> Boundary = Boundary;

      // Allocate an array of Matrix pointers
      this -> Output = new Matrix*[this -> numberOfStreams];

      for (int i = 0; i < this -> numberOfStreams; i++) {

      this -> Output[i] = new Matrix(
                                    InputDim -> Height + 2 * Boundary,
                                    InputDim -> Width + 2 * Boundary,
                                    InputDim -> Depth,
                                    NULL,
                                    DefineOnDevice
                              );

      }
}


PaddingLayer::~PaddingLayer()
{

      delete this -> Output;

}


Matrix** PaddingLayer::operator()(Matrix** D_Input){
    /*
      Note: Matrix coming is a device elemente matrix;
            Original Matrix is a Device input that needs padding
            padded_Matrix is the return of this function;

    */
      int nbx = (int)ceil((float)(this -> Output[0] -> width) / Tile_GEMM);
      int nby = (int)ceil((float)(this -> Output[0] -> height) / Tile_GEMM);
      int nbz = this -> Output[0] -> depth;

      if (nbx == 0) nbx = 1;

      if (nby == 0) nby = 1;

      dim3 dim_Grid2(nbx, nby, nbz);
      dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

      for (int i = 0; i < this -> numberOfStreams; i++) {

            // Pass to the copying strided kernel to complete the padding process
            Complete_Padding_Process <<< dim_Grid2, dim_Block2, 0, this -> streams[i] >>> (
                  this -> Output[i] -> elements,
                  this -> Output[i] -> height,
                  this -> Output[i] -> width,
                  this -> Output[i] -> depth,
                  D_Input[i] -> elements,
                  D_Input[i] -> height,
                  D_Input[i] -> width,
                  D_Input[i] -> depth,
                  this -> Boundary
            );
      }

    
      return this -> Output;
}
