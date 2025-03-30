#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "BatchNorm.h"
#include "Conv2d.h"
#include "SLayer.h"

SLayer::SLayer(const Basiclayer* SLayerDetails, Dimension* InputDim) : SLayerDetails(SLayerDetails) {

    Dimension* MovingDimension = InputDim;

    if (SLayerDetails->Conv.ConvWeights) {

        Conv2d* Conv1 = new Conv2d(
            CONV_KxK,                                 /* ConvType */
            2,                                        /* stride */
            1,                                        /* padding */
            NO_ACTIVATION,                            /* ActivationTypes */
            &(SLayerDetails->Conv),                   /* ConvDetails */
            MovingDimension
        );

        layers.push_back(Conv1);

        /* Get the expected output dimension from this layer */
        MovingDimension = Conv1->Conv2d_GetOutputDim();

    }


    if(SLayerDetails -> BatchNormDetails.Mean &&
       SLayerDetails -> BatchNormDetails.Bias &&
       SLayerDetails -> BatchNormDetails.Variance &&
       SLayerDetails -> BatchNormDetails.Weights){

        BatchNorm* BN1 = new BatchNorm(
            &(SLayerDetails->BatchNormDetails),           /* BatchNormDetails */
            SWISH_ACTIVATION,
            MovingDimension
        );

        layers.push_back(BN1);

        /* Get the expected output dimension from this layer */
        MovingDimension = BN1->BN_GetOutputDim();

    }

    this -> OutputDim = Dimension{
        .Height = MovingDimension -> Height,
        .Width = MovingDimension -> Width,
        .Depth = MovingDimension -> Depth,
      };
}

SLayer::~SLayer() {
    for (IBasicLayer* layer : layers) {
        delete layer;
    }
}

Matrix** SLayer::operator()(Matrix** D_input) {

    Matrix** output = D_input;
    for (IBasicLayer* layer : layers) {
        output = (*layer)(output);
    }
    return output;
}

Dimension* SLayer::SLayer_GetOutputDim() {
    return &(this -> OutputDim);
}