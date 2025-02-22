#pragma once

typedef enum {
    CONV_1x1,
    CONV_KxK,
    CONV_DW
} SupportConvolutionOPs;


typedef enum {
    DefineOnHost,
    DefineOnDevice,
} MatrixType;


typedef enum{
    SKIP_SUPPORTED,
    SKIP_NOT_SUPPORTED
}SkipID;

typedef enum {
    NO_ACTIVATION,
    SWISH_ACTIVATION,
    SIGMOID_ACTIVATION
} ActivationTypes;


typedef enum {
    BIASED,
    NOT_BIASED
} ErrorTypes;




#define Tile_GEMM                   16
#define BIASED                      0
#define THREAD_GRANULARITY_BLOCKS	2
