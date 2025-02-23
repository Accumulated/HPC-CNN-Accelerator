#pragma once

#include "MBConv.h"

#include "MBConv6_10_expansion_conv_parameters.h"
#include "MBConv6_10_depthwise_conv_parameters.h"
#include "MBConv6_10_squeeze_excitation_parameters.h"
#include "MBConv6_10_project_conv_parameters.h"

MBConv_Abstraction MBConv6_10_Layers{

    .Expansion{
        .Conv{
            /* const float MBConv6_10_expansion_conv_conv2d_weights  [1 * 1 * 112 * 672] */
            .ConvWeights = MBConv6_10_expansion_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 672,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 112,
        },
        .BatchNormDetails{
            .Mean = MBConv6_10_expansion_conv_BN_mean,
            .Variance = MBConv6_10_expansion_conv_BN_variance,
            .Weights = MBConv6_10_expansion_conv_BN_weights,
            .Bias = MBConv6_10_expansion_conv_BN_bias,
            .size = 672,
        }
    },

    .DepthWise{
        .Conv{
            /* const float MBConv6_10_depthwise_conv_conv2d_weights  [5 * 5 * 672 * 1] */
            .ConvWeights = MBConv6_10_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 5,
            .FilterWidth = 5,
            .FilterDepth = 672,
        },
        .BatchNormDetails{
            .Mean = MBConv6_10_depthwise_conv_BN_mean,
            .Variance = MBConv6_10_depthwise_conv_BN_variance,
            .Weights = MBConv6_10_depthwise_conv_BN_weights,
            .Bias = MBConv6_10_depthwise_conv_BN_bias,
            .size = 672,
        }
    },

    .Project{
        .Conv{
            /* const float MBConv6_10_project_conv_conv2d_weights  [1 * 1 * 672 * 112] */
            .ConvWeights = MBConv6_10_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 112,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 672,
        },
        .BatchNormDetails{
            .Mean = MBConv6_10_project_conv_BN_mean,
            .Variance = MBConv6_10_project_conv_BN_variance,
            .Weights = MBConv6_10_project_conv_BN_weights,
            .Bias = MBConv6_10_project_conv_BN_bias,
            .size = 112,
        }
    },

    .SQueezeExcite{
        .SQ1{
            .Conv{
                /* const float MBConv6_10_squeeze_excitation1_conv2d_weights  [1 * 1 * 672 * 28] */
                .ConvWeights = MBConv6_10_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv6_10_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 28,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 672,
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
                /* const float MBConv6_10_squeeze_excitation2_conv2d_weights  [1 * 1 * 28 * 672] */
                .ConvWeights = MBConv6_10_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv6_10_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 672,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 28,
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