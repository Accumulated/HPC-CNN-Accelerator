#pragma once

#include "IBasicLayer.h"


class PaddingLayer: public IBasicLayer{

public:

    Matrix** Output;
    int Boundary;


    PaddingLayer(Dimension *InputDim, int Boundary);
    ~PaddingLayer();

    Matrix** operator()(Matrix** D_Input);

};