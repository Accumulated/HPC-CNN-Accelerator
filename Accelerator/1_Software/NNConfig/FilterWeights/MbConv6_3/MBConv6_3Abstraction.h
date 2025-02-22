#pragma once

#include "MBConv.h"

#include "MBConv6_3_expansion_conv_parameters.h"
#include "MBConv6_3_depthwise_conv_parameters.h"
#include "MBConv6_3_squeeze_excitation_parameters.h"
#include "MBConv6_3_project_conv_parameters.h"

MBConv_Abstraction MBConv6_3_Layers{

    .Expansion{

        /* const const float MBConv6_3_expansion_conv_conv2d_weights  [1 * 1 * 24 * 144] */
        .ConvWeights = MBConv6_3_expansion_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 144,
        .FilterHeight = 1,
        .FilterWidth = 1,
        .FilterDepth = 24,
        .BatchNormDetails{
            .Mean = MBConv6_3_expansion_conv_BN_mean,
            .Variance = MBConv6_3_expansion_conv_BN_variance,
            .bias = MBConv6_3_expansion_conv_BN_bias,
            .weights = MBConv6_3_expansion_conv_BN_weights
        }

    },

    .DepthWise{

        /* const const float MBConv6_3_depthwise_conv_conv2d_weights  [144 * 1 * 5 * 5] */
        .ConvWeights = MBConv6_3_depthwise_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 1,
        .FilterHeight = 5,
        .FilterWidth = 5,
        .FilterDepth = 144,
        .BatchNormDetails{
            .Mean = MBConv6_3_depthwise_conv_BN_mean,
            .Variance = MBConv6_3_depthwise_conv_BN_variance,
            .bias = MBConv6_3_depthwise_conv_BN_bias,
            .weights = MBConv6_3_depthwise_conv_BN_weights
        }

    },

    .Project{

        /* const const float MBConv6_3_project_conv_conv2d_weights  [1 * 1 * 144 * 40] */
        .ConvWeights = MBConv6_3_project_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 40,
        .FilterHeight = 1,
        .FilterWidth = 1,
        .FilterDepth = 144,
        .BatchNormDetails{
            .Mean = MBConv6_3_project_conv_BN_mean,
            .Variance = MBConv6_3_project_conv_BN_variance,
            .bias = MBConv6_3_project_conv_BN_bias,
            .weights = MBConv6_3_project_conv_BN_weights
        }

    },

    .SQueezeExcite{
        .SQ1{
            /* const const float MBConv6_3_squeeze_excitation1_conv2d_weights  [1 * 1 * 144 * 6] */
            .ConvWeights = MBConv6_3_squeeze_excitation1_conv2d_weights,
            .Bias = MBConv6_3_squeeze_excitation1_conv2d_bias,
            .FilterDensity = 6,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 144,
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            }
        },

        .SQ2{
            /* const const float MBConv6_3_squeeze_excitation2_conv2d_weights  [1 * 1 * 6 * 144] */
            .ConvWeights = MBConv6_3_squeeze_excitation2_conv2d_weights,
            .Bias = MBConv6_3_squeeze_excitation2_conv2d_bias,
            .FilterDensity = 144,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 6,
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            }
        }
    }
};