#pragma once

#include "MBConv.h"

#include "MBConv6_6_expansion_conv_parameters.h"
#include "MBConv6_6_depthwise_conv_parameters.h"
#include "MBConv6_6_squeeze_excitation_parameters.h"
#include "MBConv6_6_project_conv_parameters.h"

MBConv_Abstraction MBConv6_6_Layers{

    .Expansion{
        .Conv{
            /* const float MBConv6_6_expansion_conv_conv2d_weights  [1 * 1 * 80 * 480] */
            .ConvWeights = MBConv6_6_expansion_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 480,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 80,
        },
        .BatchNormDetails{
            .Mean = MBConv6_6_expansion_conv_BN_mean,
            .Variance = MBConv6_6_expansion_conv_BN_variance,
            .Bias = MBConv6_6_expansion_conv_BN_bias,
            .Weights = MBConv6_6_expansion_conv_BN_weights
        }
    },

    .DepthWise{
        .Conv{
            /* const float MBConv6_6_depthwise_conv_conv2d_weights  [3 * 3 * 480 * 1] */
            .ConvWeights = MBConv6_6_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 3,
            .FilterWidth = 3,
            .FilterDepth = 480,
        },
        .BatchNormDetails{
            .Mean = MBConv6_6_depthwise_conv_BN_mean,
            .Variance = MBConv6_6_depthwise_conv_BN_variance,
            .Bias = MBConv6_6_depthwise_conv_BN_bias,
            .Weights = MBConv6_6_depthwise_conv_BN_weights
        }
    },

    .Project{
        .Conv{
            /* const float MBConv6_6_project_conv_conv2d_weights  [1 * 1 * 480 * 80] */
            .ConvWeights = MBConv6_6_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 80,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 480,
        },
        .BatchNormDetails{
            .Mean = MBConv6_6_project_conv_BN_mean,
            .Variance = MBConv6_6_project_conv_BN_variance,
            .Bias = MBConv6_6_project_conv_BN_bias,
            .Weights = MBConv6_6_project_conv_BN_weights
        }
    },

    .SQueezeExcite{
        .SQ1{
            .Conv{
                /* const float MBConv6_6_squeeze_excitation1_conv2d_weights  [1 * 1 * 480 * 20] */
                .ConvWeights = MBConv6_6_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv6_6_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 20,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 480,
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
                /* const float MBConv6_6_squeeze_excitation2_conv2d_weights  [1 * 1 * 20 * 480] */
                .ConvWeights = MBConv6_6_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv6_6_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 480,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 20,
            },
            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .Bias = nullptr,
                .Weights = nullptr,
            }
        }
    },

    .Stride = 1,
    .Padding = 1,
    .Skip = SKIP_SUPPORTED
};