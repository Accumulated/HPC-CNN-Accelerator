#include "CommonInclude.h"
#include "IBasicLayer.h"
#include "BatchNorm.h"
#include "Conv2d.h"
#include "Reduce.h"
#include "HLayer.h"



HLayer:: HLayer(const HLayerAbstraction* HLayerDetails, Dimension* InputDim):
                HLayerDetails(HLayerDetails){

    Dimension* MovingDimension = InputDim;

    if(this -> HLayerDetails->FirstStage.Conv.ConvWeights){

        Conv2d *Conv1 = new Conv2d(
            CONV_1x1,                                         /*ConvType*/
            1,                                                /*stride*/
            0,                                                /*padding*/
            NO_ACTIVATION,                                    /*ActivationTypes*/
            &(HLayerDetails->FirstStage.Conv),                /*ConvDetails*/
            MovingDimension
          );

          layers.push_back(Conv1);

          /* Get the expected output dimension from this layer*/
          MovingDimension = Conv1 -> Conv2d_GetOutputDim();

    }

    if(HLayerDetails -> FirstStage.BatchNormDetails.Mean &&
        HLayerDetails -> FirstStage.BatchNormDetails.Variance &&
        HLayerDetails -> FirstStage.BatchNormDetails.Weights &&
        HLayerDetails -> FirstStage.BatchNormDetails.Bias){

         BatchNorm *BN1 = new BatchNorm(
           &(HLayerDetails -> FirstStage.BatchNormDetails),  /*BatchNormDetails*/
           SWISH_ACTIVATION,                                 /*activation*/
           MovingDimension
         );

         layers.push_back(BN1);

         /* Get the expected output dimension from this layer*/
         MovingDimension = BN1 -> BN_GetOutputDim();

     }

    ReductionSum *Reduction = new ReductionSum(MovingDimension);

    MovingDimension = Reduction -> RS_GetOutputDim();

    layers.push_back(Reduction);


    if(this -> HLayerDetails->FCLayer.ConvWeights){

        Conv2d *Conv2 = new Conv2d(
            CONV_1x1,                                         /*ConvType*/
            1,                                                /*stride*/
            0,                                                /*padding*/
            NO_ACTIVATION,                                    /*ActivationTypes*/
            &(HLayerDetails->FCLayer),                /*ConvDetails*/
            MovingDimension
          );

          layers.push_back(Conv2);

          /* Get the expected output dimension from this layer*/
          MovingDimension = Conv2 -> Conv2d_GetOutputDim();

    }

}

// Destructor definition
HLayer::~HLayer() {
    // Clean up resources if necessary
}


Matrix* HLayer:: operator()(Matrix* D_input){

    return D_input;
}
