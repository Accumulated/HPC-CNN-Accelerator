#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "Padding.h"


PaddingLayer::PaddingLayer(Dimension *InputDim, int Boundary){


    this -> Boundary = Boundary;


    this -> Output = new Matrix(
                              InputDim -> Height + 2 * Boundary,
                              InputDim -> Width + 2 * Boundary,
                              InputDim -> Depth,
                              NULL,
                              DefineOnDevice
                        );


}


PaddingLayer::~PaddingLayer()
{

      delete this -> Output;

}


Matrix* PaddingLayer::operator()(Matrix* D_Input){
    /*
      Note: Matrix coming is a device elemente matrix;
            Original Matrix is a Device input that needs padding
            padded_Matrix is the return of this function;

    */


      int nbx = (int)ceil((float)(this -> Output -> width) / Tile_GEMM);
      int nby = (int)ceil((float)(this -> Output -> height) / Tile_GEMM);
      int nbz = this -> Output -> depth;

      if (nbx == 0) nbx = 1;

      if (nby == 0) nby = 1;

      dim3 dim_Grid2(nbx, nby, nbz);
      dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

      // Pass to the copying strided kernel to complete the padding process
      Complete_Padding_Process <<< dim_Grid2, dim_Block2 >>> (
            this -> Output -> elements,
            this -> Output -> height,
            this -> Output -> width,
            this -> Output -> depth,
            D_Input -> elements,
            D_Input -> height,
            D_Input -> width,
            D_Input -> depth,
            this -> Boundary
      );

      return this -> Output;
}
