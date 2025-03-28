#pragma once 

#include "CommonInclude.h"
#include "IBasicLayer.h"


class EfficientNet {
public:
    EfficientNet();
    ~EfficientNet();
    void run();

private:
    Matrix* Input;
    Dimension InputDim;
    Dimension* M;
    std::vector<IBasicLayer*> NNModel;

};

