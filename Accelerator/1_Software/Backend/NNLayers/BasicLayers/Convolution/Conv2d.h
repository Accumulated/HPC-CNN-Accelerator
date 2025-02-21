#pragma once

class Conv2d {


public:
    Conv2d(SupportConvolutionOPs ConvType,
            int kernel_size,
            int stride,
            int padding,
            float* weight,
            float* bias,
            float* input,
            float* output);

    ~Conv2d();

    void operator()();

private:
    int kernel_size;
    int stride;
    int padding;
    Matrix weight;
    Matrix bias;
    Matrix input;
    Matrix output;
    SupportConvolutionOPs ConvType;

};
