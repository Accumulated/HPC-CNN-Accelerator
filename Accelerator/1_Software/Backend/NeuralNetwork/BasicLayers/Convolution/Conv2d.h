#pragma once

#include "Padding.h"
#include "IBasicLayer.h"

class Conv2d: public IBasicLayer {


public:

    Conv2d(SupportConvolutionOPs ConvType,
            int stride,
            int padding,
            ActivationTypes activation_type,
            const ConvDetails * Details,
            Dimension* InputDim);

    ~Conv2d();

    Matrix* operator()(Matrix *D_input);
    Dimension* Conv2d_GetOutputDim();

    int kernel_size;
    int stride;
    int padding;
    Dimension* InputDim;
    Dimension OutputDim;
    ActivationTypes activation_type;
    Matrix* weight;
    Matrix* bias;
    Matrix* Output;
    PaddingLayer* pad;
    SupportConvolutionOPs ConvType;

};
