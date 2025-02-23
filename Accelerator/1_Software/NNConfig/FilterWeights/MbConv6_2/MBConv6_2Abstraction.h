#pragma once

#include "MBConv.h"

#include "MBConv6_2_expansion_conv_parameters.h"
#include "MBConv6_2_depthwise_conv_parameters.h"
#include "MBConv6_2_squeeze_excitation_parameters.h"
#include "MBConv6_2_project_conv_parameters.h"

MBConv_Abstraction MBConv6_2_Layers{

    .Expansion{
        .Conv{
            /* const float MBConv6_2_expansion_conv_conv2d_weights  [1 * 1 * 24 * 144] */
            .ConvWeights = MBConv6_2_expansion_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 144,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 24,
        },
        .BatchNormDetails{
            .Mean = MBConv6_2_expansion_conv_BN_mean,
            .Variance = MBConv6_2_expansion_conv_BN_variance,
            .Bias = MBConv6_2_expansion_conv_BN_bias,
            .Weights = MBConv6_2_expansion_conv_BN_weights
        }
    },

    .DepthWise{
        .Conv{
            /* const float MBConv6_2_depthwise_conv_conv2d_weights  [3 * 3 * 1 * 144] */
            .ConvWeights = MBConv6_2_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 3,
            .FilterWidth = 3,
            .FilterDepth = 144,
        },
        .BatchNormDetails{
            .Mean = MBConv6_2_depthwise_conv_BN_mean,
            .Variance = MBConv6_2_depthwise_conv_BN_variance,
            .Bias = MBConv6_2_depthwise_conv_BN_bias,
            .Weights = MBConv6_2_depthwise_conv_BN_weights
        }
    },

    .Project{
        .Conv{
            /* const float MBConv6_2_project_conv_conv2d_weights  [1 * 1 * 144 * 24] */
            .ConvWeights = MBConv6_2_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 24,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 144,
        },
        .BatchNormDetails{
            .Mean = MBConv6_2_project_conv_BN_mean,
            .Variance = MBConv6_2_project_conv_BN_variance,
            .Bias = MBConv6_2_project_conv_BN_bias,
            .Weights = MBConv6_2_project_conv_BN_weights
        }
    },

    .SQueezeExcite{
        .SQ1{
            .Conv{
                /* const float MBConv6_2_squeeze_excitation1_conv2d_weights  [1 * 1 * 144 * 6] */
                .ConvWeights = MBConv6_2_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv6_2_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 6,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 144,
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
                /* const float MBConv6_2_squeeze_excitation2_conv2d_weights  [1 * 1 * 6 * 144] */
                .ConvWeights = MBConv6_2_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv6_2_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 144,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 6,
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
