#pragma once


#include "HLayer.h"
#include "Head_conv_parameters.h"



HLayerAbstraction HeadLayer{

    .FirstStage{

        .Conv{
            .ConvWeights = Head_conv2d_weights,
            .Bias = NULL,
            .FilterDensity = 1280,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 320,
        },

        .BatchNormDetails{
            .Mean = Head_BN_mean,
            .Variance = Head_BN_variance,
            .Weights = Head_BN_weights,
            .Bias = Head_BN_bias,
            .size = 1280,
        },
    },

    .FCLayer {
        .ConvWeights = Head_linear_weights,
        .Bias = Head_linear_bias,
        .FilterDensity = 1,
        .FilterHeight = 1280,
        .FilterWidth = 1000,
        .FilterDepth = 1,
    },


};