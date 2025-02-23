#pragma once

#include "IBasicLayer.h"



class BatchNorm: public IBasicLayer{


public:

    Matrix *Output;
    Matrix *mean;
    Matrix *variance;
    Matrix *weights;
    Matrix *bias;
    ActivationTypes activation;
    Dimension* InputDim;
    Dimension OutputDim;


    BatchNorm(const BatchNorm_Weights* Details, ActivationTypes activation, Dimension* InputDim);

    Matrix* operator()(Matrix *D_input);


};