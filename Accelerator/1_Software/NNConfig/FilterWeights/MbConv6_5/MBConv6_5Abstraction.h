#pragma once

#include "MBConv.h"

#include "MBConv6_5_expansion_conv_parameters.h"
#include "MBConv6_5_depthwise_conv_parameters.h"
#include "MBConv6_5_squeeze_excitation_parameters.h"
#include "MBConv6_5_project_conv_parameters.h"

MBConv_Abstraction MBConv6_5_Layers{

    .Expansion{
        .Conv{
            /* const float MBConv6_5_expansion_conv_conv2d_weights  [1 * 1 * 40 * 240] */
            .ConvWeights = MBConv6_5_expansion_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 240,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 40,
        },
        .BatchNormDetails{
            .Mean = MBConv6_5_expansion_conv_BN_mean,
            .Variance = MBConv6_5_expansion_conv_BN_variance,
            .Bias = MBConv6_5_expansion_conv_BN_bias,
            .Weights = MBConv6_5_expansion_conv_BN_weights
        }
    },

    .DepthWise{
        .Conv{
            /* const float MBConv6_5_depthwise_conv_conv2d_weights  [3 * 3 * 240 * 1] */
            .ConvWeights = MBConv6_5_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 3,
            .FilterWidth = 3,
            .FilterDepth = 240,
        },
        .BatchNormDetails{
            .Mean = MBConv6_5_depthwise_conv_BN_mean,
            .Variance = MBConv6_5_depthwise_conv_BN_variance,
            .Bias = MBConv6_5_depthwise_conv_BN_bias,
            .Weights = MBConv6_5_depthwise_conv_BN_weights
        }
    },

    .Project{
        .Conv{
            /* const float MBConv6_5_project_conv_conv2d_weights  [1 * 1 * 240 * 80] */
            .ConvWeights = MBConv6_5_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 80,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 240,
        },
        .BatchNormDetails{
            .Mean = MBConv6_5_project_conv_BN_mean,
            .Variance = MBConv6_5_project_conv_BN_variance,
            .Bias = MBConv6_5_project_conv_BN_bias,
            .Weights = MBConv6_5_project_conv_BN_weights
        }
    },

    .SQueezeExcite{
        .SQ1{
            .Conv{
                /* const float MBConv6_5_squeeze_excitation1_conv2d_weights  [1 * 1 * 240 * 10] */
                .ConvWeights = MBConv6_5_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv6_5_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 10,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 240,
            },
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .Bias = nullptr,
                .Weights = nullptr,
            }
        },
        .SQ2{
            .Conv{
                /* const float MBConv6_5_squeeze_excitation2_conv2d_weights  [1 * 1 * 10 * 240] */
                .ConvWeights = MBConv6_5_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv6_5_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 240,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 10,
            },
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .Bias = nullptr,
                .Weights = nullptr,
            }
        }
    },

    .Stride = 2,
    .Padding = 1,
    .Skip = SKIP_NOT_SUPPORTED
};