#pragma once

#include "IBasicLayer.h"

class FCLayer: public IBasicLayer {


public:

    FCLayer(const ConvDetails * Details,
            Dimension* InputDim,
            ActivationTypes activation_type);

    ~FCLayer();

    Matrix** operator()(Matrix **D_input);
    Dimension* FCLayer_GetOutputDim();

    Dimension* InputDim = nullptr;
    Dimension OutputDim;
    ActivationTypes activation_type;
    Matrix* weight = nullptr;
    Matrix* bias = nullptr;
    Matrix** Output = nullptr;
};
