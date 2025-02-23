#include <tuple>
#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "BatchNorm.h"
#include "Conv2d.h"
#include "SQueeze.h"
#include "MBConv.h"


MBConv:: MBConv(const MBConv_Abstraction* MBConvDetails, Dimension* InputDim): MBConvDetails(MBConvDetails){


  Dimension* MovingDimension = InputDim;

  if(MBConvDetails -> Expansion.Conv.ConvWeights){

    Conv2d *Conv1 = new Conv2d(
      CONV_1x1,                                         /*ConvType*/
      1,                                                /*stride*/
      0,                                                /*padding*/
      NO_ACTIVATION,                                    /*ActivationTypes*/
      &(MBConvDetails -> Expansion.Conv),                /*ConvDetails*/
      MovingDimension
    );

    layers.push_back(Conv1);

    /* Get the expected output dimension from this layer*/
    MovingDimension = Conv1 -> Conv2d_GetOutputDim();

  }

  if(MBConvDetails -> Expansion.BatchNormDetails.Mean &&
     MBConvDetails -> Expansion.BatchNormDetails.Variance &&
     MBConvDetails -> Expansion.BatchNormDetails.Weights &&
     MBConvDetails -> Expansion.BatchNormDetails.Bias){

      BatchNorm *BN1 = new BatchNorm(
        &(MBConvDetails -> Expansion.BatchNormDetails),  /*BatchNormDetails*/
        SWISH_ACTIVATION,                                 /*activation*/
        MovingDimension
      );

      layers.push_back(BN1);

      /* Get the expected output dimension from this layer*/
      MovingDimension = BN1 -> BN_GetOutputDim();

  }


  if(MBConvDetails -> DepthWise.Conv.ConvWeights){

    Conv2d *Conv2 = new Conv2d(
                          CONV_DW,                                          /*ConvType*/
                          MBConvDetails -> Stride,                          /*stride*/
                          MBConvDetails -> Padding,                         /*padding*/
                          NO_ACTIVATION,                                    /*ActivationTypes*/
                          &(MBConvDetails -> DepthWise.Conv),               /*ConvDetails*/
                          MovingDimension
                      );

    layers.push_back(Conv2);

    /* Get the expected output dimension from this layer*/
    MovingDimension = Conv2 -> Conv2d_GetOutputDim();

  }

  if(MBConvDetails -> DepthWise.BatchNormDetails.Mean &&
     MBConvDetails -> DepthWise.BatchNormDetails.Variance &&
     MBConvDetails -> DepthWise.BatchNormDetails.Weights &&
     MBConvDetails -> DepthWise.BatchNormDetails.Bias){

       BatchNorm *BN2 = new BatchNorm(
                          &(MBConvDetails -> DepthWise.BatchNormDetails),  /*BatchNormDetails*/
                          SWISH_ACTIVATION,                              /*activation*/
                          MovingDimension
                        );

       layers.push_back(BN2);

      /* Get the expected output dimension from this layer*/
      MovingDimension = BN2 -> BN_GetOutputDim();
  }



  if(MBConvDetails -> SQueezeExcite.SQ1.Conv.ConvWeights &&
      MBConvDetails -> SQueezeExcite.SQ2.Conv.ConvWeights){

    /* Define a Squeeze and Excite layer for this MBConv block. */
    SQueeze *SE = new SQueeze(&(MBConvDetails -> SQueezeExcite), MovingDimension);

    layers.push_back(SE);

    MovingDimension = SE -> SQ_GetOutputDim();
  }

  if(MBConvDetails -> Project.Conv.ConvWeights){

    Conv2d *Conv3 = new Conv2d(
                      CONV_1x1,                                          /*ConvType*/
                      1,                                                /*stride*/
                      0,                                                /*padding*/
                      NO_ACTIVATION,                                    /*ActivationTypes*/
                      &(MBConvDetails -> Project.Conv),                  /*ConvDetails*/
                      MovingDimension
                    );

    layers.push_back(Conv3);

    /* Get the expected output dimension from this layer*/
    MovingDimension = Conv3 -> Conv2d_GetOutputDim();

  }

  if(MBConvDetails -> Project.BatchNormDetails.Mean &&
     MBConvDetails -> Project.BatchNormDetails.Variance &&
     MBConvDetails -> Project.BatchNormDetails.Weights &&
     MBConvDetails -> Project.BatchNormDetails.Bias){

      BatchNorm *BN3 = new BatchNorm(
        &(MBConvDetails -> Project.BatchNormDetails),  /*BatchNormDetails*/
        SWISH_ACTIVATION,                              /*activation*/
        MovingDimension
      );

      layers.push_back(BN3);
      MovingDimension = BN3 -> BN_GetOutputDim();

  }

  this -> OutputDim = Dimension{
    .Height = MovingDimension -> Height,
    .Width = MovingDimension -> Width,
    .Depth = MovingDimension -> Depth,
  };


}


Dimension* MBConv::MBConv_GetOutputDim(){

  return &(this -> OutputDim);

}


MBConv::~MBConv(){

  for (auto layer : layers) {
    delete layer;
  }

  layers.clear();
}


Matrix* MBConv:: MBConv_SKIPID(Matrix *D_Input)
{
  int nbx = (int)ceil((float)(this -> Output -> width) / Tile_GEMM);
  int nby = (int)ceil((float)(this -> Output -> height) / Tile_GEMM);
  int nbz = this -> Output -> depth;

  if (nbx == 0) nbx = 1;

  if (nby == 0) nby = 1;

  // This is the only kernel that runs 3d Grid;
  // Each block in z dimension controls 1 channel
  dim3 dim_Grid2(nbx, nby, nbz);
  dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);

  Identity_Skip <<<dim_Grid2, dim_Block2 >>> (

    this -> Output -> elements,
    this -> Output -> height,
    this -> Output -> width,
    this -> Output -> depth,
    D_Input -> elements

  );

  return this -> Output;
}



Matrix* MBConv:: operator()(Matrix* Input)
{

  Matrix *MBConvOut = Input;

  for (auto layer : layers) {
    MBConvOut = (*layer)(MBConvOut);
  }


  // Skip identity layer
  if(this -> MBConvDetails -> Skip)
  {
    MBConvOut = MBConv_SKIPID(MBConvOut);
  }

  return MBConvOut;

}
