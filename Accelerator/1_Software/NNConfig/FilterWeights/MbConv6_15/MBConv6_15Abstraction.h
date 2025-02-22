#pragma once

#include "MBConv.h"

#include "MBConv6_15_expansion_conv_parameters.h"
#include "MBConv6_15_depthwise_conv_parameters.h"
#include "MBConv6_15_squeeze_excitation_parameters.h"
#include "MBConv6_15_project_conv_parameters.h"

MBConv_Abstraction MBConv6_15_Layers{

    .Expansion{

        /* const float MBConv6_15_expansion_conv_conv2d_weights  [1 * 1 * 192 * 1152] */
        .ConvWeights = MBConv6_15_expansion_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 1152,
        .FilterHeight = 1,
        .FilterWidth = 1,
        .FilterDepth = 192,
        .BatchNormDetails{
            .Mean = MBConv6_15_expansion_conv_BN_mean,
            .Variance = MBConv6_15_expansion_conv_BN_variance,
            .bias = MBConv6_15_expansion_conv_BN_bias,
            .weights = MBConv6_15_expansion_conv_BN_weights
        }

    },

    .DepthWise{

        /* const float MBConv6_15_depthwise_conv_conv2d_weights  [3 * 3 * 1152 * 1] */
        .ConvWeights = MBConv6_15_depthwise_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 1,
        .FilterHeight = 3,
        .FilterWidth = 3,
        .FilterDepth = 1152,
        .BatchNormDetails{
            .Mean = MBConv6_15_depthwise_conv_BN_mean,
            .Variance = MBConv6_15_depthwise_conv_BN_variance,
            .bias = MBConv6_15_depthwise_conv_BN_bias,
            .weights = MBConv6_15_depthwise_conv_BN_weights
        }

    },

    .Project{

        /* const float MBConv6_15_project_conv_conv2d_weights  [1 * 1 * 1152 * 320] */
        .ConvWeights = MBConv6_15_project_conv_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 320,
        .FilterHeight = 1,
        .FilterWidth = 1,
        .FilterDepth = 1152,
        .BatchNormDetails{
            .Mean = MBConv6_15_project_conv_BN_mean,
            .Variance = MBConv6_15_project_conv_BN_variance,
            .bias = MBConv6_15_project_conv_BN_bias,
            .weights = MBConv6_15_project_conv_BN_weights
        }

    },

    .SQueezeExcite{
        .SQ1{
            /* const float MBConv6_15_squeeze_excitation1_conv2d_weights  [1 * 1 * 1152 * 48] */
            .ConvWeights = MBConv6_15_squeeze_excitation1_conv2d_weights,
            .Bias = MBConv6_15_squeeze_excitation1_conv2d_bias,
            .FilterDensity = 48,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 1152,
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            }
        },

        .SQ2{
            /* const float MBConv6_15_squeeze_excitation2_conv2d_weights  [1 * 1 * 48 * 1152] */
            .ConvWeights = MBConv6_15_squeeze_excitation2_conv2d_weights,
            .Bias = MBConv6_15_squeeze_excitation2_conv2d_bias,
            .FilterDensity = 1152,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 48,
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            }
        }
    }
};