#include "CommonInclude.h"
#include "Stem_conv_parameters.h"
#include "SLayer.h"

Basiclayer StemLayer{
    .Conv{
        /* const float MBConv6_1_squeeze_excitation1_conv2d_weights  [1 * 1 * 96 * 4] */
        .ConvWeights = Stem_conv2d_weights,
        .Bias = nullptr,
        .FilterDensity = 32,
        .FilterHeight = 3,
        .FilterWidth = 3,
        .FilterDepth = 3,
    },
    .BatchNormDetails{
        .Mean = Stem_BN_mean,
        .Variance = Stem_BN_variance,
        .Weights = Stem_BN_weights,
        .Bias = Stem_BN_bias,
        .size = 32
    }
};