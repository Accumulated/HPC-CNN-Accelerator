#pragma once

#include "IBasicLayer.h"

typedef struct HLayerAbstraction{

    Basiclayer FirstStage;
    ConvDetails FCLayer;
}HLayerAbstraction;


class HLayer : public IBasicLayer {

public:

    HLayer(const HLayerAbstraction* HLayerDetails, Dimension* InputDim);
    ~HLayer();

    Matrix** operator()(Matrix** D_input);

    std::vector<IBasicLayer*> layers;
    const HLayerAbstraction* HLayerDetails;
    Dimension OutputDim;
};