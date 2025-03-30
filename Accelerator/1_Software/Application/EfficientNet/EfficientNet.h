#pragma once 

#include "CommonInclude.h"
#include "IBasicLayer.h"


class EfficientNet: public IBasicLayer {
public:
    EfficientNet();
    ~EfficientNet();
    Matrix** operator()(Matrix **D_input);
    Matrix** operator()();

private:
    Matrix** Input;
    Dimension InputDim;
    Dimension* M;
    std::vector<IBasicLayer*> NNModel;

};

