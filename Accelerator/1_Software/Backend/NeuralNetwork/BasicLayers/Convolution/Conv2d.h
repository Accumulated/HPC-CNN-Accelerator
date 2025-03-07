#pragma once

#include "Padding.h"
#include "InputUnrolling.h"
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
    Dimension* InputDim = nullptr;
    Dimension OutputDim;
    ActivationTypes activation_type;
    Matrix* weight = nullptr;
    Matrix* bias = nullptr;
    Matrix* Output = nullptr;
    PaddingLayer* pad = nullptr;
    InputUnrolling* Conv_InputUnrolling = nullptr;
    SupportConvolutionOPs ConvType;
};
