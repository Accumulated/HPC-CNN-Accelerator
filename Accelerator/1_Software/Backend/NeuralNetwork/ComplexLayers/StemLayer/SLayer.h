#pragma once

#include "IBasicLayer.h"

class SLayer : public IBasicLayer {

public:
    SLayer(const Basiclayer* SLayerDetails, Dimension* InputDim);
    ~SLayer();

    Matrix* operator()(Matrix* D_input);
    Dimension* SLayer_GetOutputDim();

private:
    const Basiclayer* SLayerDetails;
    std::vector<IBasicLayer*> layers;
    Dimension OutputDim;
};
