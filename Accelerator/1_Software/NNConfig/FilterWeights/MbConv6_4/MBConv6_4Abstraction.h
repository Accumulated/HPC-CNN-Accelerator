#pragma once

#include "MBConv.h"

#include "MBConv6_4_expansion_conv_parameters.h"
#include "MBConv6_4_depthwise_conv_parameters.h"
#include "MBConv6_4_squeeze_excitation_parameters.h"
#include "MBConv6_4_project_conv_parameters.h"

MBConv_Abstraction MBConv6_4_Layers{

    .Expansion{
        .Conv{
            /* const float MBConv6_4_expansion_conv_conv2d_weights  [1 * 1 * 40 * 240] */
            .ConvWeights = MBConv6_4_expansion_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 240,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 40,
        },
        .BatchNormDetails{
            .Mean = MBConv6_4_expansion_conv_BN_mean,
            .Variance = MBConv6_4_expansion_conv_BN_variance,
            .Weights = MBConv6_4_expansion_conv_BN_weights,
            .Bias = MBConv6_4_expansion_conv_BN_bias,
            .size = 240
        }
    },

    .DepthWise{
        .Conv{
            /* const float MBConv6_4_depthwise_conv_conv2d_weights  [5 * 5 * 240 * 1] */
            .ConvWeights = MBConv6_4_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 5,
            .FilterWidth = 5,
            .FilterDepth = 240,
        },
        .BatchNormDetails{
            .Mean = MBConv6_4_depthwise_conv_BN_mean,
            .Variance = MBConv6_4_depthwise_conv_BN_variance,
            .Weights = MBConv6_4_depthwise_conv_BN_weights,
            .Bias = MBConv6_4_depthwise_conv_BN_bias,
            .size = 240
        }
    },

    .Project{
        .Conv{
            /* const float MBConv6_4_project_conv_conv2d_weights  [1 * 1 * 240 * 40] */
            .ConvWeights = MBConv6_4_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 40,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 240,
        },
        .BatchNormDetails{
            .Mean = MBConv6_4_project_conv_BN_mean,
            .Variance = MBConv6_4_project_conv_BN_variance,
            .Weights = MBConv6_4_project_conv_BN_weights,
            .Bias = MBConv6_4_project_conv_BN_bias,
            .size = 40
        }
    },

    .SQueezeExcite{
        .SQ1{
            .Conv{
                /* const float MBConv6_4_squeeze_excitation1_conv2d_weights  [1 * 1 * 240 * 10] */
                .ConvWeights = MBConv6_4_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv6_4_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 10,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 240,
            },
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .Weights = nullptr,
                .Bias = nullptr,
            }
        },
        .SQ2{
            .Conv{
                /* const float MBConv6_4_squeeze_excitation2_conv2d_weights  [1 * 1 * 10 * 240] */
                .ConvWeights = MBConv6_4_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv6_4_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 240,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 10,
            },
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .Weights = nullptr,
                .Bias = nullptr,
            }
        }
    },

    .Stride = 1,
    .Padding = 2,
    .Skip = SKIP_SUPPORTED
};