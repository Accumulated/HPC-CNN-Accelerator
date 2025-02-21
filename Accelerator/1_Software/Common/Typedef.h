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



typedef struct {
    int width;
    int height;
    int depth;
    float* elements;
}
Matrix;
