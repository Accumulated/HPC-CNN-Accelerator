#pragma once



class InputUnrolling{

public:

    InputUnrolling(Dimension* InputDim, const ConvDetails* Conv_Details, int padding, int stride);

    ~InputUnrolling();

    Matrix* operator()(Matrix* Device_Input);
    Matrix* operator()();

    Matrix* FilterUnrolled;
    Matrix* InputUnrolled;
    Dimension OutputDim;
    Dimension ConvolutionOutputDim;
    Dimension* InputDim;
    int padding;
    int stride;
    const ConvDetails* Conv_Details;

};
