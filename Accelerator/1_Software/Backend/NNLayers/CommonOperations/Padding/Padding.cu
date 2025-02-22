#include "CommonInclude.h"
#include "Padding.h"


Padding::Padding()
{
/*
      Set_DeviceMatrix(Original_Matrix_Before->height + 2 * Boundary,
                        Original_Matrix_Before->width + 2 * Boundary,
                        Original_Matrix_Before->depth,
                        padded_Matrix,
                        "Padded Matrix is allocated in device memory.");



*/
}


Padding::~Padding()
{

}

Matrix* Padding::operator()(Matrix* D_Input, int Boundary){
    /*
      Note: Matrix coming is a device elemente matrix;
            Original Matrix is a Device input that needs padding
            padded_Matrix is the return of this function;

      Warning: Padded_Matrix has a different size than the Original
                non padded matrix and it's not allocated in device yet.
                The allocateion is done inside this function.
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
            Boundary
      );
}
