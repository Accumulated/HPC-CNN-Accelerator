#pragma once

#include "MBConv.h"
#include "IBasicLayer.h"

class SQueeze: public IBasicLayer{


public:

    SQueeze(const MBConv_SqueezeExcite* Details, Dimension* InputDim);

    ~SQueeze();

    Matrix** operator()(Matrix** D_input);
    Dimension* SQ_GetOutputDim();

    std::vector<IBasicLayer*> layers;
    Dimension* InputDim;
    Dimension OutputDim;
    const MBConv_SqueezeExcite* Details;

};