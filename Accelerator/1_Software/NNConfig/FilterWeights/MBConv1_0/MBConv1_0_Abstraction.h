#pragma once

#include "MBConv.h"
#include "MBConv1_0_depthwise_conv_parameters.h"
#include "MBConv1_0_project_conv_parameters.h"
#include "MBConv1_0_squeeze_excitation_parameters.h"

MBConv_Abstraction MBConv1_0_Layers{

    .DepthWise{

        .Conv{

            /* const float MBConv1_0_depthwise_conv_conv2d_weights  [3 * 3 * 1 * 32] */
            .ConvWeights = MBConv1_0_depthwise_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 1,
            .FilterHeight = 3,
            .FilterWidth = 3,
            .FilterDepth = 32,
        },

        .BatchNormDetails{
            .Mean = MBConv1_0_depthwise_conv_BN_mean,
            .Variance = MBConv1_0_depthwise_conv_BN_variance,
            .bias = MBConv1_0_depthwise_conv_BN_bias,
            .weights = MBConv1_0_depthwise_conv_BN_weights
        }

    },

    .Project{

        .Conv{
            /* const float MBConv1_0_project_conv_conv2d_weights  [1 * 1 * 32 * 16] */
            .ConvWeights = MBConv1_0_project_conv_conv2d_weights,
            .Bias = nullptr,
            .FilterDensity = 16,
            .FilterHeight = 1,
            .FilterWidth = 1,
            .FilterDepth = 32,
        },

        .BatchNormDetails{
            .Mean = MBConv1_0_project_conv_BN_mean,
            .Variance = MBConv1_0_project_conv_BN_variance,
            .bias = MBConv1_0_project_conv_BN_bias,
            .weights = MBConv1_0_project_conv_BN_weights
        }

    },

    .Expansion{

        .Conv{
            /* This sub-layer doesn't exist in this MBConv layer */
            .ConvWeights = nullptr,
            .Bias = nullptr,
            .FilterDensity = 0,
            .FilterHeight = 0,
            .FilterWidth = 0,
            .FilterDepth = 0,
        },
        .BatchNormDetails{
            .Mean = nullptr,
            .Variance = nullptr,
            .bias = nullptr,
            .weights = nullptr
        }

    },

    .SQueezeExcite{

        .SQ1{

            .Conv{
                /* const float MBConv1_0_squeeze_excitation1_conv2d_weights [1 * 1 * 32 * 8] */
                .ConvWeights = MBConv1_0_squeeze_excitation1_conv2d_weights,
                .Bias = MBConv1_0_squeeze_excitation1_conv2d_bias,
                .FilterDensity = 8,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 32,
            },

            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            }
        },
        .SQ2{

            .Conv{
                /* const float MBConv1_0_squeeze_excitation2_conv2d_weights [1 * 1 * 8 * 32] */
                .ConvWeights = MBConv1_0_squeeze_excitation2_conv2d_weights,
                .Bias = MBConv1_0_squeeze_excitation2_conv2d_bias,
                .FilterDensity = 32,
                .FilterHeight = 1,
                .FilterWidth = 1,
                .FilterDepth = 8,
            },

            .BatchNormDetails{
                .Mean = nullptr,
                .Variance = nullptr,
                .bias = nullptr,
                .weights = nullptr
            }
        },
    },

    .Stride = 1,
    .Padding = 1,
    .Skip = SKIP_NOT_SUPPORTED
};
