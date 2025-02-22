#pragma once

#include "MBConv.h"

#include "MBConv6_1_expansion_conv_parameters.h"
#include "MBConv6_1_depthwise_conv_parameters.h"
#include "MBConv6_1_squeeze_excitation_parameters.h"
#include "MBConv6_1_project_conv_parameters.h"

MBConv_Abstraction MBConv6_1_Layers{

    .Expansion{

        /* const const float MBConv6_1_expansion_conv_conv2d_weights  [1 * 1 * 96 * 16] */
        .ConvWeights = MBConv6_1_expansion_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 96,
        .FilterHeight = 1,
        .FilterWidth = 1,
        .FilterDepth = 16,
        .BatchNormDetails{
            .Mean = MBConv6_1_expansion_conv_BN_mean,
            .Variance = MBConv6_1_expansion_conv_BN_variance,
            .bias = MBConv6_1_expansion_conv_BN_bias,
            .weights = MBConv6_1_expansion_conv_BN_weights
        }

    },

    .DepthWise{

        /* const const float MBConv6_1_depthwise_conv_conv2d_weights  [3 * 3 * 1 * 96] */
        .ConvWeights = MBConv6_1_depthwise_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 1,
        .FilterHeight = 3,
        .FilterWidth = 3,
        .FilterDepth = 96,
        .BatchNormDetails{
            .Mean = MBConv6_1_depthwise_conv_BN_mean,
            .Variance = MBConv6_1_depthwise_conv_BN_variance,
            .bias = MBConv6_1_depthwise_conv_BN_bias,
            .weights = MBConv6_1_depthwise_conv_BN_weights
        }

    },

    .Project{

        /* const const float MBConv6_1_project_conv_conv2d_weights  [1 * 1 * 96 * 24] */
        .ConvWeights = MBConv6_1_project_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 24,
        .FilterHeight = 1,
        .FilterWidth = 1,
        .FilterDepth = 96,
        .BatchNormDetails{
            .Mean = MBConv6_1_project_conv_BN_mean,
            .Variance = MBConv6_1_project_conv_BN_variance,
            .bias = MBConv6_1_project_conv_BN_bias,
            .weights = MBConv6_1_project_conv_BN_weights
        },
    },

    .SQueezeExcite{
        .SQ1{
            /* const const float MBConv6_1_squeeze_excitation1_conv2d_weights  [1 * 1 * 96 * 4] */
            .ConvWeights = MBConv6_1_squeeze_excitation1_conv2d_weights,
            .Bias = MBConv6_1_squeeze_excitation1_conv2d_bias,
            .FilterDensity = 4,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 96,
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            }
        },
        .SQ2{
            /* const const float MBConv6_1_squeeze_excitation2_conv2d_weights  [1 * 1 * 4 * 96] */
            .ConvWeights = MBConv6_1_squeeze_excitation2_conv2d_weights,
            .Bias = MBConv6_1_squeeze_excitation2_conv2d_bias,
            .FilterDensity = 96,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 4,
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            },
        },
    },
};