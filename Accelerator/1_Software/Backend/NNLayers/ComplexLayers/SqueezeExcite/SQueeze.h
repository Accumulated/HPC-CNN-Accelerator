#pragma once

#include "IBasicLayer.h"

class SQueeze: public IBasicLayer{


public:

    SQueeze(
        float *Filter1, float *Filter2,
        int FilterDensity1, int FilterDensity2,
        int input_channels, int output_channels,
        float* First_bias, float* Second_bias
    );

    ~SQueeze();

    Matrix* operator()(Matrix* D_input);

    Matrix* Output;
    Matrix* FilterBank1;
    Matrix* FilterBank2;
    Matrix* FirstConvOutput;
    Matrix* SecondConvOutput;
};