#pragma once

#include "IBasicLayer.h"

class Conv2d: public IBasicLayer {


public:
    Matrix *Output;

    Conv2d(SupportConvolutionOPs ConvType,
            int kernel_size,
            int stride,
            int padding,
            ActivationTypes activation_type,
            float* weight,
            float* bias);

    ~Conv2d();

    Matrix* operator()(Matrix *D_input);

private:
    int kernel_size;
    int stride;
    int padding;
    ActivationTypes activation_type;
    Matrix* weight;
    Matrix* bias;
    SupportConvolutionOPs ConvType;

};
