#pragma once

#include "MBConv.h"

#include "MBConv6_8_expansion_conv_parameters.h"
#include "MBConv6_8_depthwise_conv_parameters.h"
#include "MBConv6_8_squeeze_excitation_parameters.h"
#include "MBConv6_8_project_conv_parameters.h"

MBConv_Abstraction MBConv6_8_Layers{

    .Expansion{
        .Conv{
            /* const float MBConv6_8_expansion_conv_conv2d_weights  [1 * 1 * 80 * 480] */
            .ConvWeights = MBConv6_8_expansion_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 480,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 80,
        },
        .BatchNormDetails{
            .Mean = MBConv6_8_expansion_conv_BN_mean,
            .Variance = MBConv6_8_expansion_conv_BN_variance,
            .Weights = MBConv6_8_expansion_conv_BN_weights,
            .Bias = MBConv6_8_expansion_conv_BN_bias,
            .size = 480,
        }
    },

    .DepthWise{
        .Conv{
            /* const float MBConv6_8_depthwise_conv_conv2d_weights  [5 * 5 * 480 * 1] */
            .ConvWeights = MBConv6_8_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 5,
            .FilterWidth = 5,
            .FilterDepth = 480,
        },
        .BatchNormDetails{
            .Mean = MBConv6_8_depthwise_conv_BN_mean,
            .Variance = MBConv6_8_depthwise_conv_BN_variance,
            .Weights = MBConv6_8_depthwise_conv_BN_weights,
            .Bias = MBConv6_8_depthwise_conv_BN_bias,
            .size = 480,
        }
    },

    .Project{
        .Conv{
            /* const float MBConv6_8_project_conv_conv2d_weights  [1 * 1 * 480 * 112] */
            .ConvWeights = MBConv6_8_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 112,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 480,
        },
        .BatchNormDetails{
            .Mean = MBConv6_8_project_conv_BN_mean,
            .Variance = MBConv6_8_project_conv_BN_variance,
            .Weights = MBConv6_8_project_conv_BN_weights,
            .Bias = MBConv6_8_project_conv_BN_bias,
            .size = 112,
        }
    },

    .SQueezeExcite{
        .SQ1{
            .Conv{
                /* const float MBConv6_8_squeeze_excitation1_conv2d_weights  [1 * 1 * 480 * 20] */
                .ConvWeights = MBConv6_8_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv6_8_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 20,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 480,
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
                /* const float MBConv6_8_squeeze_excitation2_conv2d_weights  [1 * 1 * 20 * 480] */
                .ConvWeights = MBConv6_8_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv6_8_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 480,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 20,
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
    .Skip = SKIP_NOT_SUPPORTED
};