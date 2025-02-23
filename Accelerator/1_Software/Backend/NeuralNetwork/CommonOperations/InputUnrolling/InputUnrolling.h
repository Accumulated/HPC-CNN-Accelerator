#pragma once



class InputUnrolling{

public:

    InputUnrolling(int FilterSize, int stride);

    ~InputUnrolling();

    Matrix* operator()(Matrix* Device_Input, int ExpectedConv_OutputHeight, int ExpectedConv_OutputWidth);

    int FilterSize = 0;
    int Stride = 0;
    Matrix* Output;

};
