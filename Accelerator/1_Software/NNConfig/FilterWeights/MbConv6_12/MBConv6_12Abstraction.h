#pragma once

#include "MBConv.h"

#include "MBConv6_12_expansion_conv_parameters.h"
#include "MBConv6_12_depthwise_conv_parameters.h"
#include "MBConv6_12_squeeze_excitation_parameters.h"
#include "MBConv6_12_project_conv_parameters.h"

MBConv_Abstraction MBConv6_12_Layers{

    .Expansion{
        .Conv{
            /* const float MBConv6_12_expansion_conv_conv2d_weights  [1 * 1 * 192 * 1152] */
            .ConvWeights = MBConv6_12_expansion_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1152,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 192,
        },
        .BatchNormDetails{
            .Mean = MBConv6_12_expansion_conv_BN_mean,
            .Variance = MBConv6_12_expansion_conv_BN_variance,
            .Weights = MBConv6_12_expansion_conv_BN_weights,
            .Bias = MBConv6_12_expansion_conv_BN_bias,
            .size = 1152,
        }
    },

    .DepthWise{
        .Conv{
            /* const float MBConv6_12_depthwise_conv_conv2d_weights  [5 * 5 * 1152 * 1] */
            .ConvWeights = MBConv6_12_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 5,
            .FilterWidth = 5,
            .FilterDepth = 1152,
        },
        .BatchNormDetails{
            .Mean = MBConv6_12_depthwise_conv_BN_mean,
            .Variance = MBConv6_12_depthwise_conv_BN_variance,
            .Weights = MBConv6_12_depthwise_conv_BN_weights,
            .Bias = MBConv6_12_depthwise_conv_BN_bias,
            .size = 1152,
        }
    },

    .Project{
        .Conv{
            /* const float MBConv6_12_project_conv_conv2d_weights  [1 * 1 * 1152 * 192] */
            .ConvWeights = MBConv6_12_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 192,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 1152,
        },
        .BatchNormDetails{
            .Mean = MBConv6_12_project_conv_BN_mean,
            .Variance = MBConv6_12_project_conv_BN_variance,
            .Weights = MBConv6_12_project_conv_BN_weights,
            .Bias = MBConv6_12_project_conv_BN_bias,
            .size = 192,
        }
    },

    .SQueezeExcite{
        .SQ1{
            .Conv{
                /* const float MBConv6_12_squeeze_excitation1_conv2d_weights  [1 * 1 * 1152 * 48] */
                .ConvWeights = MBConv6_12_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv6_12_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 48,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 1152,
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
                /* const float MBConv6_12_squeeze_excitation2_conv2d_weights  [1 * 1 * 48 * 1152] */
                .ConvWeights = MBConv6_12_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv6_12_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 1152,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 48,
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