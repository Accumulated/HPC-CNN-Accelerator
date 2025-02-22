#pragma once

#include "IBasicLayer.h"


typedef struct BatchNorm_Weights{
    float* Mean;
    float* Variance;
    float* Weights;
    float* Bias;
}BatchNorm_Weights;


class BatchNorm: public IBasicLayer{


public:

    Matrix *Output;
    Matrix *mean;
    Matrix *variance;
    Matrix *weights;
    Matrix *bias;
    ActivationTypes activation;

    BatchNorm(  float* mean,
                float* variance,
                float* weights,
                float* bias,
                ActivationTypes activation
            );

    Matrix* operator()(Matrix *D_input);


};