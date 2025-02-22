#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "MBConv.h"
#include "Conv2d.h"
#include "BatchNorm.h"
#include "SQueeze.h"


MBConv::MBConv(
    float* ExpansionConvWeights, float* DepthWiseConvWeights,
    float* SqueezeExciteWeights_1, float* SqueezeExciteWeights_2,
    float* ProjectWeights,

    int ExpansionFilterDensity, int DepthWiseFilterDensity,
    int SqueezeExcite1FilterDensity, int SqueezeExcite2FilterDensity,
    int ProjectFilterDensity,

    int input_channels, int output_channels, int FilterSizeDW,
    int Stride, int padding, SkipID SkipMe,

    float* SqueezeExciteBias_1, float* SqueezeExciteBias_2,

    float* ExpansionBatchNorm_Mean, float* ExpansionBatchNorm_Variance,
    float* ExpansionBatchNorm_Weights, float* ExpansionBatchNorm_Bias,

    float* DepthWiseBatchNorm_Mean, float* DepthWiseBatchNorm_Variance,
    float* DepthWiseBatchNorm_Weights, float* DepthWiseBatchNorm_Bias,

    float* ProjectBatchNorm_Mean, float* ProjectBatchNorm_Variance,
    float* ProjectBatchNorm_Weights, float* ProjectBatchNorm_Bias
  ){

    /* Expansion Layer conv2d + BN */
    if(ExpansionConvWeights){
      /* Define an expansion layer for this MBConv block. */
      Conv2d *Conv1 = new Conv2d(
        CONV_1x1,                     /*ConvType*/
        1,                            /*kernel_size*/
        1,                            /*stride*/
        0,                            /*padding*/
        NO_ACTIVATION,                /*ActivationTypes*/
        ExpansionConvWeights,         /*weight*/
        NULL                          /*bias*/
      );

      layers.push_back(Conv1);
    }

    if(ExpansionBatchNorm_Mean && ExpansionBatchNorm_Variance &&
      ExpansionBatchNorm_Weights && ExpansionBatchNorm_Bias){
     /* Define a batch normalization layer for the expansion layer. */
     BatchNorm *BN1 = new BatchNorm(
       ExpansionBatchNorm_Mean,      /*mean*/
       ExpansionBatchNorm_Variance,  /*variance*/
       ExpansionBatchNorm_Weights,   /*weights*/
       ExpansionBatchNorm_Bias,       /*bias*/
       SWISH_ACTIVATION              /*activation*/
     );

     layers.push_back(BN1);
   }

    /* Depthwise sub-layer - Conv2d and BatchNorm */
    if(DepthWiseConvWeights){
      /* Define a depthwise layer for this MBConv block. */
      Conv2d *Conv2 = new Conv2d(
        CONV_KxK,                     /*ConvType*/
        FilterSizeDW,                 /*kernel_size*/
        Stride,                       /*stride*/
        padding,                      /*padding*/
        NO_ACTIVATION,                /*ActivationTypes*/
        DepthWiseConvWeights,         /*weight*/
        NULL                          /*bias*/
      );

      layers.push_back(Conv2);
    }


    if(DepthWiseBatchNorm_Mean && DepthWiseBatchNorm_Variance &&
      DepthWiseBatchNorm_Weights && DepthWiseBatchNorm_Bias){
     /* Define a batch normalization layer for the depthwise layer. */
     BatchNorm *BN2 = new BatchNorm(
       DepthWiseBatchNorm_Mean,      /*mean*/
       DepthWiseBatchNorm_Variance,  /*variance*/
       DepthWiseBatchNorm_Weights,   /*weights*/
       DepthWiseBatchNorm_Bias,       /*bias*/
       SWISH_ACTIVATION              /*activation*/
     );

     layers.push_back(BN2);
   }


    if(SqueezeExciteWeights_1 && SqueezeExciteWeights_2){
      /* Define a Squeeze and Excite layer for this MBConv block. */
      SQueeze *SE = new SQueeze(
        SqueezeExciteWeights_1,       /*Filter1*/
        SqueezeExciteWeights_2,       /*Filter2*/
        SqueezeExcite1FilterDensity,  /*FilterDensity1*/
        SqueezeExcite2FilterDensity,  /*FilterDensity2*/
        SqueezeExcite2FilterDensity,  /*input_channels*/
        SqueezeExcite1FilterDensity,  /*output_channels*/
        SqueezeExciteBias_1,          /*bias1*/
        SqueezeExciteBias_2           /*bias2*/
      );

      layers.push_back(SE);
    }


    if(ProjectWeights){
      /* Define a projection layer for this MBConv block. */
      Conv2d *Conv3 = new Conv2d(
        CONV_1x1,                     /*ConvType*/
        1,                            /*kernel_size*/
        1,                            /*stride*/
        0,                            /*padding*/
        NO_ACTIVATION,                /*ActivationTypes*/
        ProjectWeights,               /*weight*/
        NULL                          /*bias*/
      );

      layers.push_back(Conv3);
    }


    if(ProjectBatchNorm_Mean && ProjectBatchNorm_Variance &&
       ProjectBatchNorm_Weights && ProjectBatchNorm_Bias){
      /* Define a batch normalization layer for the projection layer. */
      BatchNorm *BN3 = new BatchNorm(
        ProjectBatchNorm_Mean,        /*mean*/
        ProjectBatchNorm_Variance,    /*variance*/
        ProjectBatchNorm_Weights,     /*weights*/
        ProjectBatchNorm_Bias,         /*bias*/
        NO_ACTIVATION                 /*activation*/
      );

      layers.push_back(BN3);
    }


}

MBConv::~MBConv(){



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



// 5 Filters needed to run the 4 layers sequentially
Matrix* MBConv:: operator()(Matrix* Input
  /*
    Matrix* F1, Matrix* F2, Matrix* F3, Matrix* F4, Matrix* F5,
    int FD1, int FD2, int FD3, int FD4, int FD5,
    int input_channels, int output_channels, int FilterSizeDW,
    int Stride, int padding, int skip,
    Matrix *bias1, Matrix *bias2,
    Matrix *ExpansionConvWeights,     Matrix *MBConv_expansion_conv_BN_variance,
    Matrix *MBConv_expansion_conv_BN_weights,  Matrix *MBConv_expansion_conv_BN_bias,
    Matrix *MBConv_depthwise_conv_BN_mean,     Matrix *MBConv_depthwise_conv_BN_variance,
    Matrix *MBConv_depthwise_conv_BN_weights,  Matrix *MBConv_depthwise_conv_BN_bias,
    Matrix *MBConv_project_conv_BN_mean,       Matrix *MBConv_project_conv_BN_variance,
    Matrix *MBConv_project_conv_BN_weights,    Matrix *MBConv_project_conv_BN_bias*/)
{

    Matrix *ptr_mat;

    // 1st layer: 1x1 Conv2d, stride = 1, padding = 0, K = 1
    Conv2d_Layer(Input, this -> ExpansionConvWeights, &tmp1, 1, 0,
                  input_channels, FD1, FD1,
                  CONV_1x1,
                  NO_ACTIVATION, 0, NULL);

    BN_ALL_PRE_DEFINED(&tmp1, SWISH_ACTIVATION,
                        ExpansionConvWeights,    MBConv_expansion_conv_BN_variance,
                        MBConv_expansion_conv_BN_weights, MBConv_expansion_conv_BN_bias);

   // 2nd Layer: KxK DWconv, stride = s, padding = p, K = k

    // Height and width changes, Only depth remains still
    int OutputHeight = (ptr_mat -> height + 2 * padding - FilterSizeDW)/Stride + 1;
    int OutputWidth = (ptr_mat -> width + 2 * padding - FilterSizeDW)/Stride + 1;
    int OutputDepth = ptr_mat -> depth;

    // Set and allocate tmp2 matrix; it's a transistion between expansion and squeeze
    Matrix tmp2;
    Set_DeviceMatrix(OutputHeight, OutputWidth, OutputDepth, &tmp2,
                    "Output_2 is allocated in device memory");

    Conv2d_Layer(ptr_mat, this -> DepthWiseConvWeights, &tmp2,
                 Stride, padding, FD1, FD2, FD2, CONV_DW,
                 NO_ACTIVATION, 0, NULL);


    BN_ALL_PRE_DEFINED(&tmp2, SWISH_ACTIVATION,
                       MBConv_depthwise_conv_BN_mean,     MBConv_depthwise_conv_BN_variance,
                       MBConv_depthwise_conv_BN_weights,  MBConv_depthwise_conv_BN_bias);

    // 3rd Layer: squeeze and excitation

    /*
      Squeeze excite layer doesn't change the final output dimensions;
      SE_OUT can be removed; Do so later
    */
    Matrix *SE_OUT;
    Squeeze_and_Excite(&tmp2, SE_OUT, this -> SqueezeExciteWeights_1, this -> SquuezeExciteWeights_2,
                        FD4, FD3, FD4, FD3,
                        bias1, bias2);

    // 4th Layer: 1x1 Conv2d
    // MBConv output pointer is set and finally updated after this layer execution
    Set_DeviceMatrix(tmp2.height, tmp2.width, FD5, MBConvOut,
                     "Matrix final output is allocated in device memory");


    // 1x1 Conv2d layer
    Conv2d_Layer(&tmp2, this -> ProjectWeights, MBConvOut, 1, 0, FD4, FD5, FD5, CONV_1x1,
                 NO_ACTIVATION, 0, NULL);


    // BatchNorm layer
    BN_ALL_PRE_DEFINED(MBConvOut, NO_ACTIVATION,
                       this ->  MBConv_project_conv_BN_mean,     MBConv_project_conv_BN_variance,
                       MBConv_project_conv_BN_weights,  MBConv_project_conv_BN_bias);

    // Skip identity layer
    if(skip)
    {
      MBConv_SKIP_IDENTITY(MBConvOut, Input);
    }

}
