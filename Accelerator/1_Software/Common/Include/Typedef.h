#pragma once


typedef enum {
    CONV_1x1 = 0,
    CONV_KxK = 1,
    CONV_DW = 2,
} SupportConvolutionOPs;


typedef enum {
    DefineOnHost,
    DefineOnDevice,
} MatrixType;


typedef enum{
    SKIP_NOT_SUPPORTED = 0,
    SKIP_SUPPORTED = 1
}SkipID;


typedef enum {
    NO_ACTIVATION = 0,
    SWISH_ACTIVATION = 1,
    SIGMOID_ACTIVATION = 2,
} ActivationTypes;


typedef enum {
    BIASED = 0,
    NOT_BIASED = 1
} ErrorTypes;



typedef struct BatchNorm_Weights{
    const float* Mean;
    const float* Variance;
    const float* Weights;
    const float* Bias;
}BatchNorm_Weights;


typedef struct ConvDetails{

    const float* ConvWeights;
    const float* Bias;
    int FilterDensity;
    int FilterHeight;
    int FilterWidth;
    int FilterDepth;

}ConvDetails;


typedef struct Basiclayer{

    ConvDetails Conv;
    BatchNorm_Weights BatchNormDetails;

}Basiclayer;


typedef struct Dimension{
    int Height;
    int Width;
    int Depth;
}Dimension;


#define Tile_GEMM                   16
#define BIASED                      0
#define THREAD_GRANULARITY_BLOCKS	2
