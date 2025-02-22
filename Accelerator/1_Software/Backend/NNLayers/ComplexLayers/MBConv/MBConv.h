#pragma once


typedef struct MBConv_Basiclayer{

    const float* ConvWeights;
    const float* Bias;
    int FilterDensity;
    int FilterHeight;
    int FilterWidth;
    int FilterDepth;


    BatchNorm_Weights BatchNormDetails;


}MBConv_Basiclayer;


typedef struct MBConv_SqueezeExcite{

    MBConv_Basiclayer SQ1;
    MBConv_Basiclayer SQ2;

}MBConv_SqueezeExcite;


typedef struct MBConv_Abstraction{


    MBConv_Basiclayer Expansion;
    MBConv_Basiclayer DepthWise;
    MBConv_Basiclayer Project;
    MBConv_SqueezeExcite SQueezeExcite;


}MBConv_Abstraction;



class MBConv: public IBasicLayer{


public:

    MBConv(
        /* Convolution weights */
        const float* ExpansionConvWeights, const float* DepthWiseConvWeights,
        const float* SqueezeExciteWeights_1, const float* SquuezeExciteWeights_2,
        const float* ProjectWeights,

        /* Filter density information for each layer */
        int ExpansionFilterDensity, int DepthWiseFilterDensity,
        int SqueezeExcite1FilterDensity, int SqueezeExcite2FilterDensity,
        int ProjectFilterDensity,

        /* I/P and O/P channels for the whole MBConv layer */
        int input_channels, int output_channels, int FilterSizeDW,

        /* Stride and padding are used for the DWConv sublayer. */
        int Stride, int padding,

        /* Choose whether to have a skip identity layer or not. */
        SkipID SkipMe,

        /* Bias matrices for squeeze excitation layers. */
        float* SqueezeExciteBias_1, float* SqueezeExciteBias_2,

        /* BatchNorm parameters for each layer */
        const float* ExpansionBatchNorm_Mean, const float* ExpansionBatchNorm_Variance,
        const float* ExpansionBatchNorm_Weights, const float* ExpansionBatchNorm_Bias,

        const float* DepthWiseBatchNorm_Mean, const float* DepthWiseBatchNorm_Variance,
        const float* DepthWiseBatchNorm_Weights, const float* DepthWiseBatchNorm_Bias,

        const float* ProjectBatchNorm_Mean, const float* ProjectBatchNorm_Variance,
        const float* ProjectBatchNorm_Weights, const float* ProjectBatchNorm_Bias);

    ~MBConv();

    Matrix* operator()(Matrix* D_input);
    Matrix* MBConv_SKIPID(Matrix *D_Input);


    std::vector<IBasicLayer*> layers;

    Matrix* ExpansionConvWeights;
    Matrix* DepthWiseConvWeights;
    Matrix* SqueezeExciteWeights_1;
    Matrix* SquuezeExciteWeights_2;
    Matrix* ProjectWeights;

    Matrix* Output;

};